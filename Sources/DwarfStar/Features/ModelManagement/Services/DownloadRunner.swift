import Foundation
import DS4Engine

private enum DownloadUIEvent: Sendable {
    case snapshot(progress: ModelDownloadProgress?, state: ModelDownloadState)
}

/// Thread-safe relay used directly from URLSession callbacks. Every progress
/// snapshot carries the latest phase, so a bounded AsyncStream may coalesce
/// byte events without ever losing the transition to SHA verification.
private final class DownloadCallbackRelay: @unchecked Sendable {
    private let lock = NSLock()
    private var state: ModelDownloadState = .checkingLocalFile
    private let continuation: AsyncStream<DownloadUIEvent>.Continuation

    init(_ continuation: AsyncStream<DownloadUIEvent>.Continuation) {
        self.continuation = continuation
    }

    func yield(progress: ModelDownloadProgress) {
        lock.lock()
        let current = state
        lock.unlock()
        _ = continuation.yield(.snapshot(progress: progress, state: current))
    }

    func yield(state: ModelDownloadState) {
        lock.lock()
        self.state = state
        lock.unlock()
        _ = continuation.yield(.snapshot(progress: nil, state: state))
    }
}

enum CatalogInstallState: Equatable {
    case notDownloaded
    case partial
    case invalidLocalFile
    case installed

    var title: String {
        switch self {
        case .notDownloaded: "Non scaricato"
        case .partial: "Parziale"
        case .invalidLocalFile: "File non valido"
        case .installed: "Installato"
        }
    }

    var symbol: String {
        switch self {
        case .notDownloaded: "icloud.and.arrow.down"
        case .partial: "arrow.clockwise.circle"
        case .invalidLocalFile: "exclamationmark.octagon.fill"
        case .installed: "checkmark.circle.fill"
        }
    }
}

enum CatalogUpdateState: Equatable {
    case unknown
    case checking
    case current
    case available
    case failed(String)
}

struct InvalidCatalogArtifact {
    let path: String
    let reason: String
}

struct CatalogInstallation {
    let state: CatalogInstallState
    let installedArtifacts: Int
    let artifactCount: Int
    let localBytes: Int64
    let partialBytes: Int64
    let pathsByTargetID: [String: String]
    let invalidArtifacts: [String: InvalidCatalogArtifact]

    static let missing = CatalogInstallation(
        state: .notDownloaded,
        installedArtifacts: 0,
        artifactCount: 0,
        localBytes: 0,
        partialBytes: 0,
        pathsByTargetID: [:],
        invalidArtifacts: [:]
    )
}

/// Main-actor adapter for the native Engine downloader. It owns presentation
/// state only: the supported catalog, HTTP/resume policy and integrity checks
/// remain in DS4Engine.
@MainActor
@Observable
final class DownloadRunner {
    private enum RunnerError: LocalizedError {
        case incompletePackage
        case incompleteRemoteMetadata(String)

        var errorDescription: String? {
            switch self {
            case .incompletePackage:
                "Il download è terminato, ma uno o più file attesi non sono presenti."
            case .incompleteRemoteMetadata(let file):
                "Hugging Face non ha restituito SHA-256 e dimensione per \(file)."
            }
        }
    }

    struct ActiveProgress {
        let entryID: String
        var artifactName: String
        var artifactIndex: Int
        var artifactCount: Int
        var completedBytes: Int64
        var totalBytes: Int64
        var overallFraction: Double
        var phase: String

        var byteSummary: String {
            guard totalBytes > 0 else { return "Preparazione…" }
            let done = ByteCountFormatter.string(fromByteCount: completedBytes, countStyle: .file)
            let total = ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file)
            return "\(done) / \(total)"
        }
    }

    private(set) var installations: [String: CatalogInstallation] = [:]
    private(set) var active: ActiveProgress?
    private(set) var errors: [String: String] = [:]
    private(set) var notices: [String: String] = [:]
    private(set) var availableBytes: Int64 = 0
    private(set) var updates: [String: CatalogUpdateState] = [:]

    private var searchDirectories: [URL] = []
    private var destination = AppEnvironment.modelDownloadDirectory
    private var task: Task<Void, Never>?
    private var remoteEntries: [String: ModelCatalogEntry] = [:]

    var isRunning: Bool { task != nil }

    func configure(searchDirectories: [URL], destination: URL) {
        self.destination = destination.standardizedFileURL
        self.searchDirectories = Self.uniqueDirectories([destination] + searchDirectories)
        refresh()
    }

    func installation(for entry: ModelCatalogEntry) -> CatalogInstallation {
        installations[entry.id.rawValue] ?? .missing
    }

    func error(for entry: ModelCatalogEntry) -> String? {
        errors[entry.id.rawValue]
    }

    func notice(for entry: ModelCatalogEntry) -> String? {
        notices[entry.id.rawValue]
    }

    func isActive(_ entry: ModelCatalogEntry) -> Bool {
        active?.entryID == entry.id.rawValue
    }

    func updateState(for entry: ModelCatalogEntry) -> CatalogUpdateState {
        updates[entry.id.rawValue] ?? .unknown
    }

    /// Compare catalog artifacts with the live Hugging Face manifest. The API
    /// exposes the LFS SHA-256 and exact size, so no model payload is downloaded.
    func checkForUpdates() {
        guard task == nil else { return }
        refresh()
        let installedEntries = ModelCatalogRegistry.entries.filter {
            installation(for: $0).state == .installed
        }
        guard !installedEntries.isEmpty else { return }
        for entry in installedEntries { updates[entry.id.rawValue] = .checking }

        task = Task { [weak self] in
            guard let self else { return }
            defer { self.task = nil }
            for entry in installedEntries {
                do {
                    let refreshed = try await self.remoteEntry(for: entry)
                    self.remoteEntries[entry.id.rawValue] = refreshed
                    self.updates[entry.id.rawValue] = (self.matchesInstalledReceipt(refreshed)
                        || refreshed.artifacts == entry.artifacts)
                        ? .current : .available
                } catch {
                    self.updates[entry.id.rawValue] = .failed(error.localizedDescription)
                }
            }
        }
    }

    func installUpdate(_ entry: ModelCatalogEntry,
                       onSelectableModel: @escaping @MainActor (String) -> Bool) {
        guard let remote = remoteEntries[entry.id.rawValue], task == nil else { return }
        acquire(remote, replacing: true, onSelectableModel: onSelectableModel)
    }

    func refresh() {
        var next: [String: CatalogInstallation] = [:]
        for entry in ModelCatalogRegistry.entries {
            let effective = remoteEntries[entry.id.rawValue] ?? effectiveInstalledEntry(entry)
            next[entry.id.rawValue] = inspect(effective)
        }
        installations = next
        availableBytes = Self.availableCapacity(at: destination)
    }

    /// Acquire every artifact in a catalog entry. Existing non-empty files in
    /// any known model directory are reused in place when their pinned size, if
    /// present, also matches; only missing files are downloaded into Application
    /// Support. A selectable single-file model is handed back to the caller
    /// after either the reuse or download path succeeds.
    func acquire(_ entry: ModelCatalogEntry,
                 onSelectableModel: @escaping @MainActor (String) -> Bool) {
        acquire(entry, replacing: false, onSelectableModel: onSelectableModel)
    }

    private func acquire(_ entry: ModelCatalogEntry, replacing: Bool,
                         onSelectableModel: @escaping @MainActor (String) -> Bool) {
        guard task == nil else { return }
        let key = entry.id.rawValue
        errors[key] = nil
        notices[key] = nil

        let initial = inspect(entry)
        installations[key] = initial
        guard initial.state != .invalidLocalFile else { return }
        if initial.state == .installed && !replacing {
            if let path = selectablePath(for: entry, installation: initial) {
                notices[key] = onSelectableModel(path)
                    ? "Già installato: è stato selezionato senza scaricarlo di nuovo."
                    : "Il file è installato, ma la validazione del runtime ne ha impedito la selezione."
            } else if entry.isSplitFragmentPackage {
                notices[key] = "Tutte le parti consecutive del GGUF sono già installate."
            } else if entry.artifacts.count > 1 {
                notices[key] = "Tutti gli shard del pacchetto distribuito sono già installati."
            } else {
                notices[key] = entry.runtimeAvailability.unavailableReason
                    ?? "L'artefatto è già installato, ma non è selezionabile dal runtime corrente."
            }
            return
        }

        active = ActiveProgress(
            entryID: key,
            artifactName: "",
            artifactIndex: 0,
            artifactCount: entry.artifacts.count,
            completedBytes: 0,
            totalBytes: 0,
            overallFraction: 0,
            phase: "Preparazione"
        )

        task = Task { [weak self] in
            guard let self else { return }
            defer {
                self.active = nil
                self.task = nil
                self.refresh()
            }
            do {
                var downloaded = false
                for (index, target) in entry.artifacts.enumerated() {
                    try Task.checkCancellation()
                    let catalogTarget = ModelCatalogRegistry.entry(entry.id)?.artifacts
                        .first(where: { $0.id == target.id })
                    let replaceArtifact = replacing && catalogTarget != target
                    if !replaceArtifact, self.existingFile(for: target) != nil {
                        self.updateSkippedArtifact(entry: entry, target: target, index: index)
                        continue
                    }
                    downloaded = true
                    try await self.acquireArtifact(target, entry: entry, index: index,
                                                   replacing: replaceArtifact)
                }
                self.updates[key] = .current

                let installed = self.inspect(entry)
                self.installations[key] = installed
                guard installed.state == .installed else {
                    throw RunnerError.incompletePackage
                }
                if let path = self.selectablePath(for: entry, installation: installed) {
                    if onSelectableModel(path) {
                        self.notices[key] = downloaded
                            ? "Download completato. Il modello è stato selezionato."
                            : "Già installato: il modello è stato selezionato."
                    } else {
                        self.notices[key] = "Download completato, ma la validazione del runtime ne ha impedito la selezione."
                    }
                } else if entry.isSplitFragmentPackage {
                    self.notices[key] = "Download delle parti completato. Il futuro lettore Kimi le esporrà come un unico GGUF senza duplicarle."
                } else if entry.artifacts.count > 1 {
                    self.notices[key] = "Download del pacchetto distribuito completato. Gli shard non sono selezionabili come un GGUF locale."
                } else {
                    self.notices[key] = entry.runtimeAvailability.unavailableReason
                        ?? "Download completato. Questo artefatto non è selezionabile dal runtime corrente."
                }
            } catch is CancellationError {
                self.notices[key] = "Download annullato. Il file parziale è conservato per riprendere più tardi."
            } catch {
                self.errors[key] = error.localizedDescription
            }
        }
    }

    func cancel() {
        guard task != nil else { return }
        active?.phase = "Annullamento…"
        task?.cancel()
    }

    // MARK: - Download bridge

    private func acquireArtifact(_ target: ModelTarget,
                                 entry: ModelCatalogEntry,
                                 index: Int,
                                 replacing: Bool = false) async throws {
        let count = max(entry.artifacts.count, 1)
        active = ActiveProgress(
            entryID: entry.id.rawValue,
            artifactName: target.file,
            artifactIndex: index + 1,
            artifactCount: count,
            completedBytes: 0,
            totalBytes: 0,
            overallFraction: Double(index) / Double(count),
            phase: "Download"
        )

        // Buffer only the latest few callbacks and consume at UI cadence. Model
        // files are hundreds of GB and the engine can report every MiB; queuing
        // one MainActor task per callback would itself become a bottleneck.
        let (stream, continuation) = AsyncStream<DownloadUIEvent>.makeStream(
            bufferingPolicy: .bufferingNewest(4)
        )
        let relay = DownloadCallbackRelay(continuation)
        let directory = destination
        let token = HFTokenStore.load()
        let final = try ModelDownloader.destinationURL(for: target, in: directory)
        let backup = directory.appendingPathComponent(target.file + ".previous")
        if replacing, FileManager.default.fileExists(atPath: final.path) {
            guard !FileManager.default.fileExists(atPath: backup.path) else {
                throw RunnerError.incompletePackage
            }
            try FileManager.default.moveItem(at: final, to: backup)
        }
        async let result: ModelDownloadResult = {
            defer { continuation.finish() } // also finish on error/cancellation
            do {
                let value = try await ModelDownloader.acquire(
                    target: target, ggufDirectory: directory, token: token,
                    onProgress: { relay.yield(progress: $0) },
                    onState: { relay.yield(state: $0) })
                if replacing { try? FileManager.default.removeItem(at: backup) }
                return value
            } catch {
                if replacing,
                   !FileManager.default.fileExists(atPath: final.path),
                   FileManager.default.fileExists(atPath: backup.path) {
                    try? FileManager.default.moveItem(at: backup, to: final)
                }
                throw error
            }
        }()

        for await event in stream {
            switch event {
            case .snapshot(let progress, let state):
                // `active` is an optional value type tracked by @Observable.
                // Mutating one of its fields opens `_modify` on the complete
                // optional. Reading `active` again on the right-hand side of
                // that mutation trips Swift's exclusivity runtime. Apply the
                // complete event to a local value and publish it once instead.
                guard var nextActive = active,
                      nextActive.entryID == entry.id.rawValue,
                      nextActive.artifactName == target.file else { continue }

                nextActive.phase = Self.friendlyPhase(state)
                switch state {
                case .verifying:
                    nextActive.overallFraction = max(
                        nextActive.overallFraction,
                        (Double(index) + 0.9) / Double(count)
                    )
                case .finalizing:
                    nextActive.overallFraction = max(
                        nextActive.overallFraction,
                        (Double(index) + 0.99) / Double(count)
                    )
                case .completed:
                    nextActive.overallFraction = Double(index + 1) / Double(count)
                default:
                    break
                }
                if let progress {
                    nextActive.completedBytes = progress.completedBytes
                    nextActive.totalBytes = progress.totalBytes ?? 0
                    let rawFraction = progress.fractionCompleted ?? 0
                    // The integrity pass re-reads the complete file and reports
                    // bytes from zero. Reserve the last 10% for SHA verification
                    // so a resumed download never makes the bar jump backwards.
                    let artifactFraction: Double
                    if case .verifying = state {
                        artifactFraction = 0.9 + rawFraction * 0.1
                    } else {
                        artifactFraction = rawFraction * 0.9
                    }
                    let next = (Double(index) + artifactFraction) / Double(count)
                    nextActive.overallFraction = max(nextActive.overallFraction, next)
                }
                active = nextActive
            }
            // Coalesce a burst into at most ~8 visible updates per second.
            try? await Task.sleep(for: .milliseconds(125))
        }
        _ = try await result
        rememberInstalled(target)
    }

    private static let receiptDefaultsKey = "ds4.modelArtifactReceipts.v1"

    private func rememberInstalled(_ target: ModelTarget) {
        guard let sha = target.sha256, let size = target.expectedSizeBytes else { return }
        var receipts = UserDefaults.standard.dictionary(forKey: Self.receiptDefaultsKey)
            as? [String: String] ?? [:]
        receipts[target.id] = "\(sha.lowercased())|\(size)"
        UserDefaults.standard.set(receipts, forKey: Self.receiptDefaultsKey)
    }

    private func matchesInstalledReceipt(_ entry: ModelCatalogEntry) -> Bool {
        let receipts = UserDefaults.standard.dictionary(forKey: Self.receiptDefaultsKey)
            as? [String: String] ?? [:]
        return entry.artifacts.allSatisfy { target in
            guard let sha = target.sha256, let size = target.expectedSizeBytes else { return false }
            return receipts[target.id] == "\(sha.lowercased())|\(size)"
        }
    }

    private func effectiveInstalledEntry(_ entry: ModelCatalogEntry) -> ModelCatalogEntry {
        let receipts = UserDefaults.standard.dictionary(forKey: Self.receiptDefaultsKey)
            as? [String: String] ?? [:]
        let artifacts = entry.artifacts.map { target -> ModelTarget in
            guard let value = receipts[target.id],
                  let separator = value.lastIndex(of: "|"),
                  let size = Int64(value[value.index(after: separator)...]) else { return target }
            return ModelTarget(
                id: target.id, file: target.file, approxGB: target.approxGB, note: target.note,
                sha256: String(value[..<separator]), expectedSizeBytes: size,
                role: target.role, source: target.source)
        }
        return ModelCatalogEntry(
            id: entry.id, displayName: entry.displayName, profile: entry.profile,
            summary: entry.summary, artifacts: artifacts,
            runtimeAvailability: entry.runtimeAvailability)
    }

    private struct HFManifest: Decodable {
        struct Sibling: Decodable {
            /// Current HF model manifests call the digest `sha256`; older API
            /// responses and some compatible hubs use `oid` instead.
            struct LFS: Decodable {
                let sha256: String?
                let oid: String?
                let size: Int64?
            }
            let rfilename: String
            let lfs: LFS?
            let size: Int64?
        }
        let siblings: [Sibling]
    }

    private func remoteEntry(for entry: ModelCatalogEntry) async throws -> ModelCatalogEntry {
        var manifests: [HuggingFaceSource: HFManifest] = [:]
        var artifacts: [ModelTarget] = []
        for target in entry.artifacts {
            let manifest: HFManifest
            if let cached = manifests[target.source] {
                manifest = cached
            } else {
                let repo = target.source.repository
                    .split(separator: "/").map(String.init)
                    .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? $0 }
                    .joined(separator: "/")
                let revision = target.source.revision
                    .addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? target.source.revision
                var request = URLRequest(url: URL(string: "https://huggingface.co/api/models/\(repo)/revision/\(revision)?blobs=true")!)
                if let token = HFTokenStore.load() {
                    request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
                }
                let (data, response) = try await URLSession.shared.data(for: request)
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    throw URLError(.badServerResponse)
                }
                manifest = try JSONDecoder().decode(HFManifest.self, from: data)
                manifests[target.source] = manifest
            }
            guard let remote = manifest.siblings.first(where: { $0.rfilename == target.file }) else {
                throw URLError(.fileDoesNotExist)
            }
            guard let digest = remote.lfs?.sha256 ?? remote.lfs?.oid,
                  let byteCount = remote.lfs?.size ?? remote.size else {
                throw RunnerError.incompleteRemoteMetadata(target.file)
            }
            artifacts.append(ModelTarget(
                id: target.id, file: target.file, approxGB: target.approxGB, note: target.note,
                sha256: digest, expectedSizeBytes: byteCount,
                role: target.role, source: target.source))
        }
        return ModelCatalogEntry(id: entry.id, displayName: entry.displayName,
            profile: entry.profile, summary: entry.summary, artifacts: artifacts,
            runtimeAvailability: entry.runtimeAvailability)
    }

    private func updateSkippedArtifact(entry: ModelCatalogEntry,
                                       target: ModelTarget,
                                       index: Int) {
        let count = max(entry.artifacts.count, 1)
        active = ActiveProgress(
            entryID: entry.id.rawValue,
            artifactName: target.file,
            artifactIndex: index + 1,
            artifactCount: count,
            completedBytes: 1,
            totalBytes: 1,
            overallFraction: Double(index + 1) / Double(count),
            phase: "Già presente"
        )
    }

    // MARK: - Local installation state

    private func inspect(_ entry: ModelCatalogEntry) -> CatalogInstallation {
        var paths: [String: String] = [:]
        var installedBytes: Int64 = 0
        var partialBytes: Int64 = 0
        var invalidArtifacts: [String: InvalidCatalogArtifact] = [:]

        for target in entry.artifacts {
            if let existing = existingFile(for: target) {
                paths[target.id] = existing.path
                installedBytes += Self.fileSize(existing)
            } else if let invalid = invalidManagedArtifact(for: target) {
                invalidArtifacts[target.id] = invalid
            } else {
                partialBytes += Self.fileSize(
                    destination.appendingPathComponent(target.file + ".part")
                )
            }
        }

        let installed = paths.count
        let count = entry.artifacts.count
        let state: CatalogInstallState
        if !invalidArtifacts.isEmpty {
            state = .invalidLocalFile
        } else if count > 0, installed == count {
            state = .installed
        } else if installed > 0 || partialBytes > 0 {
            state = .partial
        } else {
            state = .notDownloaded
        }
        return CatalogInstallation(
            state: state,
            installedArtifacts: installed,
            artifactCount: count,
            localBytes: installedBytes,
            partialBytes: partialBytes,
            pathsByTargetID: paths,
            invalidArtifacts: invalidArtifacts
        )
    }

    private func selectablePath(for entry: ModelCatalogEntry,
                                installation: CatalogInstallation) -> String? {
        guard entry.isSelectable, entry.artifacts.count == 1,
              let target = entry.artifacts.first else { return nil }
        return installation.pathsByTargetID[target.id]
    }

    private func existingFile(for target: ModelTarget) -> URL? {
        for directory in searchDirectories {
            let candidate = directory.appendingPathComponent(target.file, isDirectory: false)
            guard FileManager.default.isReadableFile(atPath: candidate.path) else { continue }
            let size = Self.fileSize(candidate)
            guard size > 0 else { continue }
            if let expected = target.expectedSizeBytes,
               size != expected {
                continue
            }
            return candidate
        }
        return nil
    }

    /// A non-empty final file in the managed destination is user-owned. If it
    /// cannot satisfy the pinned catalog metadata, expose it to the UI instead
    /// of overwriting it or offering a retry loop that can never succeed.
    private func invalidManagedArtifact(for target: ModelTarget) -> InvalidCatalogArtifact? {
        let candidate = destination.appendingPathComponent(target.file, isDirectory: false)
        if (try? FileManager.default.destinationOfSymbolicLink(atPath: candidate.path)) != nil {
            return .init(path: candidate.path, reason: "il percorso è un link simbolico")
        }
        guard FileManager.default.fileExists(atPath: candidate.path) else { return nil }
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: candidate.path),
              attributes[.type] as? FileAttributeType == .typeRegular else {
            return .init(path: candidate.path, reason: "il percorso non è un file regolare")
        }
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        guard size > 0 else { return nil } // the Engine safely removes empty sentinels
        if let expected = target.expectedSizeBytes, size != expected {
            let actualText = ByteCountFormatter.string(fromByteCount: size, countStyle: .file)
            let expectedText = ByteCountFormatter.string(fromByteCount: expected, countStyle: .file)
            return .init(
                path: candidate.path,
                reason: "dimensione \(actualText), attesi \(expectedText)"
            )
        }
        return nil
    }

    private static func fileSize(_ url: URL) -> Int64 {
        guard (try? FileManager.default.destinationOfSymbolicLink(atPath: url.path)) == nil else {
            return 0
        }
        let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values?.isRegularFile == true else { return 0 }
        return Int64(values?.fileSize ?? 0)
    }

    private static func uniqueDirectories(_ directories: [URL]) -> [URL] {
        var seen = Set<String>()
        return directories.compactMap { directory in
            let normalized = directory.standardizedFileURL
            return seen.insert(normalized.path).inserted ? normalized : nil
        }
    }

    private static func availableCapacity(at directory: URL) -> Int64 {
        let values = try? directory.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey,
            .volumeAvailableCapacityKey,
        ])
        if let important = values?.volumeAvailableCapacityForImportantUsage {
            return important
        }
        return Int64(values?.volumeAvailableCapacity ?? 0)
    }

    private static func friendlyPhase(_ state: ModelDownloadState) -> String {
        switch state {
        case .checkingLocalFile: "Controllo file locale"
        case .preparingRequest: "Connessione"
        case .resuming(let byte):
            "Ripresa da \(ByteCountFormatter.string(fromByteCount: byte, countStyle: .file))"
        case .downloading: "Download"
        case .verifying: "Verifica integrità"
        case .finalizing: "Finalizzazione"
        case .completed(.alreadyPresent): "Già presente"
        case .completed(.downloaded): "Completato"
        }
    }
}
