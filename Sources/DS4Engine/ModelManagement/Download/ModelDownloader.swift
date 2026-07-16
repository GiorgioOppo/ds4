import Foundation
import CryptoKit

public enum ModelLocalFileState: Sendable, Hashable {
    case missing
    case empty
    case present(byteCount: Int64)
}

/// Native, resumable downloader for the model catalog. It never invokes curl,
/// `hf`, or another process. The final GGUF only appears after a complete,
/// verified `.part` file is atomically renamed in the same directory.
public enum ModelDownloader {
    public static let repo = "antirez/deepseek-v4-gguf"

    /// Compatibility view for older callers. New UI code should use
    /// `DeepSeekV4ModelCatalog.entries`, which preserves package boundaries and
    /// runtime availability. MTP remains addressable as an optional accessory.
    public static var targets: [ModelTarget] {
        DeepSeekV4ModelCatalog.allArtifacts + [DeepSeekV4AccessoryCatalog.mtp]
    }

    public static func target(_ id: String) -> ModelTarget? {
        targets.first { $0.id == id }
    }

    public static func resolveURL(_ file: String) -> URL {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/\(repo)/resolve/main/\(file)"
        return components.url!
    }

    /// Resolve an HF token: explicit > HF_TOKEN env > ~/.cache/huggingface/token.
    /// The GUI reads `HFTokenStore` and supplies it as the explicit value, so
    /// command-line/library use never triggers a surprise Keychain prompt.
    public static func resolveToken(_ explicit: String?) -> String? {
        if let token = normalizedToken(explicit) { return token }
        if let token = normalizedToken(ProcessInfo.processInfo.environment["HF_TOKEN"]) {
            return token
        }
        let path = (NSHomeDirectory() as NSString)
            .appendingPathComponent(".cache/huggingface/token")
        if let value = try? String(contentsOfFile: path, encoding: .utf8) {
            return normalizedToken(value)
        }
        return nil
    }

    public enum DownloadError: Error, Sendable, CustomStringConvertible, LocalizedError {
        case unsafeFileName(String)
        case existingItemNotRegular(String)
        case partialItemNotRegular(String)
        case existingFileSizeMismatch(expected: Int64, actual: Int64)
        case http(Int)
        case invalidContentRange(String?)
        case unexpectedContentEncoding(String)
        case rangeNotSatisfiable(partialBytes: Int64, remoteBytes: Int64?)
        case remoteObjectChanged
        case incompleteDownload(expected: Int64, actual: Int64)
        case insufficientDiskSpace(required: Int64, available: Int64)
        case checksumMismatch(expected: String, got: String)
        case alreadyInProgress(String)

        public var description: String {
            switch self {
            case .unsafeFileName(let name):
                return "Nome file remoto non sicuro: \(name)"
            case .existingItemNotRegular(let path):
                return "Il percorso finale esiste ma non è un file regolare: \(path)"
            case .partialItemNotRegular(let path):
                return "Il percorso parziale esiste ma non è un file regolare: \(path)"
            case .existingFileSizeMismatch(let expected, let actual):
                return "Il modello locale ha dimensione \(actual) byte, attesi \(expected)."
            case .http(401):
                return "Hugging Face richiede un token valido per scaricare questo file."
            case .http(403):
                return "Il token Hugging Face non ha accesso a questo file (HTTP 403)."
            case .http(404):
                return "Il file non è più disponibile nel repository Hugging Face (HTTP 404)."
            case .http(let code):
                return "Download Hugging Face non riuscito (HTTP \(code))."
            case .invalidContentRange(let value):
                return "Risposta HTTP Range non valida: \(value ?? "header assente")"
            case .unexpectedContentEncoding(let value):
                return "Il server ha applicato una codifica HTTP non compatibile con il resume: \(value)."
            case .rangeNotSatisfiable(let partial, let remote):
                let remoteDescription = remote.map(String.init) ?? "sconosciuta"
                return "Il file .part è di \(partial) byte, dimensione remota \(remoteDescription); impossibile riprendere in sicurezza."
            case .remoteObjectChanged:
                return "Il file remoto è cambiato durante la ripresa; il .part è stato conservato."
            case .incompleteDownload(let expected, let actual):
                return "Download incompleto: ricevuti \(actual) byte su \(expected)."
            case .insufficientDiskSpace(let required, let available):
                return "Spazio insufficiente: servono circa \(required) byte, disponibili \(available)."
            case .checksumMismatch(let expected, let got):
                return "SHA-256 non valido: atteso \(expected), ottenuto \(got). Il file parziale è stato rimosso."
            case .alreadyInProgress(let path):
                return "Un download è già in corso per \(path)."
            }
        }

        public var errorDescription: String? { description }
    }

    /// Canonical API used by the GUI. A non-empty regular final file is never
    /// downloaded again: the result explicitly reports `.alreadyPresent`.
    public static func acquire(
        target: ModelTarget,
        ggufDirectory: URL,
        token: String? = nil,
        onProgress: @escaping @Sendable (ModelDownloadProgress) -> Void = { _ in },
        onState: @escaping @Sendable (ModelDownloadState) -> Void = { _ in }
    ) async throws -> ModelDownloadResult {
        let directory = ggufDirectory.standardizedFileURL
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let destination = try destinationURL(for: target, in: directory)

        guard await downloadGate.begin(destination.path) else {
            throw DownloadError.alreadyInProgress(destination.path)
        }
        do {
            let result = try await acquireLocked(
                target: target,
                directory: directory,
                destination: destination,
                token: token,
                onProgress: onProgress,
                onState: onState
            )
            await downloadGate.end(destination.path)
            return result
        } catch {
            await downloadGate.end(destination.path)
            throw error
        }
    }

    /// Source-compatible wrapper used by the pre-existing GUI runner.
    public static func download(
        target: ModelTarget,
        ggufDir: String,
        token: String? = nil,
        onProgress: @escaping @Sendable (Int64, Int64) -> Void,
        onStatus: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> URL {
        let result = try await acquire(
            target: target,
            ggufDirectory: URL(fileURLWithPath: ggufDir, isDirectory: true),
            token: token,
            onProgress: { progress in
                onProgress(progress.completedBytes, progress.totalBytes ?? 0)
            },
            onState: { state in onStatus(statusDescription(state)) }
        )
        return result.fileURL
    }

    /// Resolve and validate the final local path without touching the network.
    public static func destinationURL(for target: ModelTarget, in directory: URL) throws -> URL {
        let name = target.file
        guard !name.isEmpty,
              name != ".", name != "..",
              !name.contains("/"), !name.contains("\\"), !name.contains("\0"),
              (name as NSString).lastPathComponent == name else {
            throw DownloadError.unsafeFileName(name)
        }
        let base = directory.standardizedFileURL
        let destination = base.appendingPathComponent(name, isDirectory: false).standardizedFileURL
        guard destination.deletingLastPathComponent().path == base.path else {
            throw DownloadError.unsafeFileName(name)
        }
        return destination
    }

    /// Inspect a catalog target locally. This pure filesystem check is also used
    /// by the GUI to show Installed without issuing HEAD/network requests.
    public static func localState(target: ModelTarget, in directory: URL) throws
        -> ModelLocalFileState {
        let destination = try destinationURL(for: target, in: directory)
        if isSymbolicLink(destination) {
            throw DownloadError.existingItemNotRegular(destination.path)
        }
        guard FileManager.default.fileExists(atPath: destination.path) else { return .missing }
        let size = try regularFileSize(
            at: destination,
            invalid: DownloadError.existingItemNotRegular(destination.path)
        )
        return size > 0 ? .present(byteCount: size) : .empty
    }

    /// Minimum free bytes required for a same-volume `.part` download. Rename is
    /// atomic and needs no second copy; the margin covers filesystem metadata and
    /// catalog size rounding.
    public static func requiredFreeSpace(
        totalBytes: Int64,
        existingPartialBytes: Int64,
        minimumMarginBytes: Int64 = 1 << 30
    ) -> Int64 {
        let total = max(0, totalBytes)
        let remaining = max(0, total - max(0, existingPartialBytes))
        let proportionalMargin = total / 50 // 2%
        let margin = max(0, max(minimumMarginBytes, proportionalMargin))
        let (sum, overflow) = remaining.addingReportingOverflow(margin)
        return overflow ? Int64.max : sum
    }

    /// Stream a file from disk and return its lowercase-hex SHA-256. Memory use
    /// is bounded by `chunk`, and cancellation is checked between blocks.
    public static func sha256Hex(
        of url: URL,
        chunk: Int = 8 << 20,
        onProgress: (Int64) -> Void = { _ in }
    ) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        var hashed: Int64 = 0
        while let data = try handle.read(upToCount: max(1, chunk)), !data.isEmpty {
            hasher.update(data: data)
            hashed += Int64(data.count)
            onProgress(hashed)
            try Task.checkCancellation()
        }
        return hex(hasher.finalize())
    }

    // MARK: - Internal flow

    private static func acquireLocked(
        target: ModelTarget,
        directory: URL,
        destination: URL,
        token: String?,
        onProgress: @escaping @Sendable (ModelDownloadProgress) -> Void,
        onState: @escaping @Sendable (ModelDownloadState) -> Void
    ) async throws -> ModelDownloadResult {
        try Task.checkCancellation()
        onState(.checkingLocalFile)

        switch try localState(target: target, in: directory) {
        case .present(let byteCount):
            if let expected = target.expectedSizeBytes, byteCount != expected {
                throw DownloadError.existingFileSizeMismatch(expected: expected, actual: byteCount)
            }
            onProgress(.init(completedBytes: byteCount, totalBytes: byteCount))
            onState(.completed(.alreadyPresent))
            return .init(
                disposition: .alreadyPresent,
                fileURL: destination,
                byteCount: byteCount
            )
        case .empty:
            // A zero-byte final file can never be a valid GGUF. Removing only
            // this empty sentinel is safe and lets an existing `.part` resume.
            try FileManager.default.removeItem(at: destination)
        case .missing:
            break
        }

        let partial = URL(fileURLWithPath: destination.path + ".part")
        let resumeMetadata = URL(fileURLWithPath: partial.path + ".resume.json")
        if isSymbolicLink(resumeMetadata) {
            throw DownloadError.partialItemNotRegular(resumeMetadata.path)
        }
        let partialBytes = try partialFileSize(at: partial)

        // For a fresh download the catalog estimate catches an undersized disk
        // before opening the connection. For a resume, wait for the exact
        // Content-Range: a fully downloaded `.part` must still be able to receive
        // 416 and finalize even when very little free space remains.
        if partialBytes == 0 || target.expectedSizeBytes != nil {
            try preflightSpace(
                directory: directory,
                totalBytes: target.expectedSizeBytes
                    ?? Int64(target.approxGB) * 1_000_000_000,
                existingPartialBytes: partialBytes
            )
        }

        onState(.preparingRequest)
        var request = URLRequest(
            url: resolveURL(target.file),
            cachePolicy: .reloadIgnoringLocalCacheData,
            timeoutInterval: 120
        )
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        if partialBytes > 0 {
            request.setValue("bytes=\(partialBytes)-", forHTTPHeaderField: "Range")
            if let validator = ResumeMetadata.load(from: resumeMetadata)?.validator {
                request.setValue(validator, forHTTPHeaderField: "If-Range")
            }
        }
        if let resolvedToken = resolveToken(token) {
            request.setValue("Bearer \(resolvedToken)", forHTTPHeaderField: "Authorization")
        }

        let transfer = HTTPRangeFileTransfer(
            request: request,
            partialURL: partial,
            metadataURL: resumeMetadata,
            requestedOffset: partialBytes,
            expectedSizeBytes: target.expectedSizeBytes,
            expectedSHA256: target.sha256,
            onProgress: onProgress,
            onState: onState
        )
        let transferResult = try await transfer.start()

        try Task.checkCancellation()
        onState(.verifying)
        try verify(
            file: partial,
            target: target,
            computed: transferResult.computedSHA256,
            onProgress: onProgress
        )

        onState(.finalizing)
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw DownloadError.existingItemNotRegular(destination.path)
        }
        try FileManager.default.moveItem(at: partial, to: destination)
        try? FileManager.default.removeItem(at: resumeMetadata)

        let size = try regularFileSize(
            at: destination,
            invalid: DownloadError.existingItemNotRegular(destination.path)
        )
        onProgress(.init(completedBytes: size, totalBytes: size))
        onState(.completed(.downloaded))
        return .init(disposition: .downloaded, fileURL: destination, byteCount: size)
    }

    private static func partialFileSize(at url: URL) throws -> Int64 {
        if isSymbolicLink(url) {
            throw DownloadError.partialItemNotRegular(url.path)
        }
        guard FileManager.default.fileExists(atPath: url.path) else { return 0 }
        return try regularFileSize(
            at: url,
            invalid: DownloadError.partialItemNotRegular(url.path)
        )
    }

    private static func regularFileSize(at url: URL, invalid: Error) throws -> Int64 {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard attributes[.type] as? FileAttributeType == .typeRegular else { throw invalid }
        return (attributes[.size] as? NSNumber)?.int64Value ?? 0
    }

    private static func isSymbolicLink(_ url: URL) -> Bool {
        (try? FileManager.default.destinationOfSymbolicLink(atPath: url.path)) != nil
    }

    private static func preflightSpace(
        directory: URL,
        totalBytes: Int64,
        existingPartialBytes: Int64
    ) throws {
        let values = try? directory.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        )
        guard let available = values?.volumeAvailableCapacityForImportantUsage else { return }
        let required = requiredFreeSpace(
            totalBytes: totalBytes,
            existingPartialBytes: existingPartialBytes
        )
        guard available >= required else {
            throw DownloadError.insufficientDiskSpace(required: required, available: available)
        }
    }

    private static func verify(
        file: URL,
        target: ModelTarget,
        computed: String?,
        onProgress: @escaping @Sendable (ModelDownloadProgress) -> Void
    ) throws {
        guard let expected = target.sha256, !expected.isEmpty else { return }
        let total = try regularFileSize(
            at: file,
            invalid: DownloadError.partialItemNotRegular(file.path)
        )
        let actual: String
        if let computed {
            actual = computed
            onProgress(.init(completedBytes: total, totalBytes: total))
        } else {
            actual = try sha256Hex(of: file) { hashed in
                onProgress(.init(completedBytes: hashed, totalBytes: total))
            }
        }
        guard actual.caseInsensitiveCompare(expected) == .orderedSame else {
            try? FileManager.default.removeItem(at: file)
            try? FileManager.default.removeItem(
                at: URL(fileURLWithPath: file.path + ".resume.json")
            )
            throw DownloadError.checksumMismatch(
                expected: expected.lowercased(),
                got: actual.lowercased()
            )
        }
    }

    private static func normalizedToken(_ token: String?) -> String? {
        let trimmed = token?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func statusDescription(_ state: ModelDownloadState) -> String {
        switch state {
        case .checkingLocalFile: return "Controllo del modello locale…"
        case .preparingRequest: return "Preparazione del download…"
        case .resuming(let offset): return "Ripresa del download da \(offset) byte…"
        case .downloading: return "Download in corso…"
        case .verifying: return "Verifica SHA-256…"
        case .finalizing: return "Finalizzazione del modello…"
        case .completed(.alreadyPresent): return "Modello già presente."
        case .completed(.downloaded): return "Download completato."
        }
    }

    static func hex(_ digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }

    static func availableCapacity(at directory: URL) -> Int64? {
        (try? directory.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        ))?.volumeAvailableCapacityForImportantUsage
    }

    private static let downloadGate = ModelDownloadGate()
}

private actor ModelDownloadGate {
    private var activePaths = Set<String>()

    func begin(_ path: String) -> Bool {
        activePaths.insert(path).inserted
    }

    func end(_ path: String) {
        activePaths.remove(path)
    }
}

struct ResumeMetadata: Codable, Sendable {
    let validator: String?
    let totalBytes: Int64?

    static func load(from url: URL) -> ResumeMetadata? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Self.self, from: data)
    }

    func save(to url: URL) {
        guard let data = try? JSONEncoder().encode(self) else { return }
        try? data.write(to: url, options: .atomic)
    }
}
