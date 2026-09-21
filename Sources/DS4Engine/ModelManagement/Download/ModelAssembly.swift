import CryptoKit
import Darwin
import Foundation

extension ModelDownloader {
    public enum AssemblyError: Error, LocalizedError, Sendable {
        case invalidRecipe
        case unsafeFile(String)
        case incompatiblePrefix(String)
        case digestMismatch(String)
        case invalidSize(String)

        public var errorDescription: String? {
            switch self {
            case .invalidRecipe: "Ricetta di assemblaggio incompleta o incoerente."
            case .unsafeFile(let path): "L'assemblaggio richiede file regolari senza link: \(path)."
            case .incompatiblePrefix(let path): "Il prefisso parziale non corrisponde ai frammenti. Sposta o rimuovi \(path) prima di riprovare; i frammenti sono conservati."
            case .digestMismatch(let file): "SHA-256 non corrispondente per \(file). Nessun GGUF finale pubblicato; frammenti e parziale sono conservati."
            case .invalidSize(let file): "Dimensione non valida durante l'assemblaggio: \(file)."
            }
        }
    }

    /// Concatenate byte fragments with a fixed-size buffer. Existing partial
    /// bytes are compared against their sources before appending; every source
    /// digest and the final digest must match before atomic publication. No
    /// input fragment is removed. A resumed pass rechecks the complete prefix.
    public static func assemble(
        entry: ModelCatalogEntry, fragmentURLs: [URL], in directory: URL,
        chunkBytes: Int = 8 << 20,
        onProgress: @escaping @Sendable (ModelDownloadProgress) -> Void = { _ in },
        onState: @escaping @Sendable (ModelDownloadState) -> Void = { _ in }
    ) async throws -> ModelDownloadResult {
        guard let output = entry.assemblyOutput else { throw AssemblyError.invalidRecipe }
        let destination = try destinationURL(for: output, in: directory)
        guard await assemblyGate.begin(destination.path) else {
            throw DownloadError.alreadyInProgress(destination.path)
        }
        do {
            let work = Task.detached(priority: .utility) {
                try assembleSynchronously(entry: entry, fragmentURLs: fragmentURLs,
                    directory: directory, destination: destination, chunkBytes: chunkBytes,
                    onProgress: onProgress, onState: onState)
            }
            let result = try await withTaskCancellationHandler {
                try await work.value
            } onCancel: { work.cancel() }
            await assemblyGate.end(destination.path)
            return result
        } catch {
            await assemblyGate.end(destination.path)
            throw error
        }
    }

    private static func assembleSynchronously(
        entry: ModelCatalogEntry, fragmentURLs: [URL], directory: URL,
        destination: URL, chunkBytes: Int,
        onProgress: @escaping @Sendable (ModelDownloadProgress) -> Void,
        onState: @escaping @Sendable (ModelDownloadState) -> Void
    ) throws -> ModelDownloadResult {
        func validDigest(_ value: String?) -> Bool {
            guard let value, value.count == 64 else { return false }
            return value.utf8.allSatisfy { (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0) }
        }
        guard entry.isSplitFragmentPackage, fragmentURLs.count == entry.artifacts.count,
              Set(fragmentURLs.map { $0.standardizedFileURL.path }).count == fragmentURLs.count,
              let output = entry.assemblyOutput, output.role == .mainModel,
              let total = output.expectedSizeBytes, total > 0, validDigest(output.sha256),
              entry.expectedSizeBytes == total,
              entry.artifacts.allSatisfy({ ($0.expectedSizeBytes ?? 0) > 0 && validDigest($0.sha256) }),
              !fragmentURLs.contains(where: { $0.standardizedFileURL == destination }) else {
            throw AssemblyError.invalidRecipe
        }
        try Task.checkCancellation()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        onState(.checkingLocalFile)
        switch try localState(target: output, in: directory) {
        case .present(let size):
            guard size == total else { throw AssemblyError.invalidSize(output.file) }
            onState(.verifying)
            let digest = try sha256Hex(of: destination) { onProgress(.init(completedBytes: $0, totalBytes: total)) }
            guard digest == output.sha256!.lowercased() else { throw AssemblyError.digestMismatch(output.file) }
            onState(.completed(.alreadyPresent))
            return .init(disposition: .alreadyPresent, fileURL: destination, byteCount: total)
        case .empty: throw AssemblyError.invalidSize(output.file)
        case .missing: break
        }
        let partial = URL(fileURLWithPath: destination.path + ".assembling.part")
        guard !fragmentURLs.contains(where: { $0.standardizedFileURL == partial.standardizedFileURL }) else {
            throw AssemblyError.invalidRecipe
        }
        // O_NOFOLLOW closes the stat/open symlink race. Reject hard links too,
        // since appending to staging must not mutate another user-owned path.
        func openRegular(_ url: URL, flags: Int32, singleLink: Bool) throws -> (FileHandle, Int64) {
            let fd = Darwin.open(url.path, flags | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
            guard fd >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
            var info = stat()
            guard fstat(fd, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG,
                  !singleLink || info.st_nlink == 1 else {
                Darwin.close(fd); throw AssemblyError.unsafeFile(url.path)
            }
            let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
            _ = enableUncachedIO(for: handle)
            return (handle, info.st_size)
        }
        let (out, prefixBytes) = try openRegular(partial, flags: O_RDWR | O_CREAT, singleLink: true)
        defer { try? out.close() }
        guard prefixBytes >= 0, prefixBytes <= total else { throw AssemblyError.invalidSize(partial.lastPathComponent) }
        if prefixBytes < total, let available = availableCapacity(at: directory) {
            let required = requiredFreeSpace(totalBytes: total, existingPartialBytes: prefixBytes)
            guard available >= required else { throw DownloadError.insufficientDiskSpace(required: required, available: available) }
        }
        onState(.assembling)
        let blockSize = max(1, min(chunkBytes, maximumSHA256ChunkBytes))
        var complete: Int64 = 0, finalHash = SHA256()
        for (target, url) in zip(entry.artifacts, fragmentURLs) {
            try Task.checkCancellation()
            let (input, size) = try openRegular(url, flags: O_RDONLY, singleLink: false)
            defer { try? input.close() }
            guard size == target.expectedSizeBytes else { throw AssemblyError.invalidSize(target.file) }
            var fragmentHash = SHA256(), read: Int64 = 0
            while read < size {
                try Task.checkCancellation()
                try autoreleasepool {
                    guard let data = try input.read(upToCount: min(blockSize, Int(size - read))), !data.isEmpty else {
                        throw AssemblyError.invalidSize(target.file)
                    }
                    fragmentHash.update(data: data)
                    finalHash.update(data: data)
                    let prefixCount = Int(min(Int64(data.count), max(0, prefixBytes - complete)))
                    if prefixCount > 0 {
                        let previous = try out.read(upToCount: prefixCount)
                        guard previous == data.prefix(prefixCount) else {
                            throw AssemblyError.incompatiblePrefix(partial.path)
                        }
                    }
                    if prefixCount < data.count { try out.write(contentsOf: data.dropFirst(prefixCount)) }
                    read += Int64(data.count); complete += Int64(data.count)
                }
                onProgress(.init(completedBytes: complete, totalBytes: total))
            }
            // Detect a source that grew while we were copying the pinned size.
            guard (try input.read(upToCount: 1))?.isEmpty != false else { throw AssemblyError.invalidSize(target.file) }
            guard hex(fragmentHash.finalize()) == target.sha256!.lowercased() else {
                throw AssemblyError.digestMismatch(target.file)
            }
        }
        try Task.checkCancellation()
        guard complete == total else { throw AssemblyError.invalidSize(output.file) }
        guard hex(finalHash.finalize()) == output.sha256!.lowercased() else {
            throw AssemblyError.digestMismatch(output.file)
        }
        try out.synchronize()
        onState(.finalizing)
        try Task.checkCancellation()
        // FileManager refuses to replace an existing final file. Source and
        // destination are in the same directory, so publication is a rename.
        try FileManager.default.moveItem(at: partial, to: destination)
        onState(.completed(.downloaded))
        return .init(disposition: .downloaded, fileURL: destination, byteCount: total)
    }

    private static let assemblyGate = ModelAssemblyGate()
}

private actor ModelAssemblyGate {
    private var paths: Set<String> = []
    func begin(_ path: String) -> Bool { paths.insert(path).inserted }
    func end(_ path: String) { paths.remove(path) }
}
