import Foundation
import CryptoKit

// File identity for the coordinator-driven distribution: streaming SHA-256
// with a persistent (size, mtime)-validated cache — hashing a 100+ GB gguf
// happens ONCE per machine, later connects answer from the manifest in
// milliseconds — plus the worker-side managed store where transferred files
// land (hash recorded at reception, so verification never re-reads them).

/// Streaming SHA-256 with a JSON cache keyed by absolute path and validated
/// by (size, mtime): touching or replacing the file invalidates the entry.
public enum DistFileHash {
    private static let lock = NSLock()
    private static var cacheURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DwarfStar/dist-file-hashes.json")
    }

    private struct Entry: Codable {
        var size: UInt64; var mtime: Double; var sha256: String
        /// v8: chained checkpoint digests (hex, one per fileCheckpointBytes).
        /// Optional: cache files written by older builds load fine and the
        /// chain is recomputed (and remembered) once.
        var chain: [String]?
    }

    /// The file's SHA-256, from the cache when (size, mtime) still match, else
    /// computed by streaming the file (F_NOCACHE: hashing must not evict the
    /// hot page cache) and cached. nil if unreadable. `onLog` is called only
    /// synchronously (non-escaping — an optional closure parameter would be
    /// implicitly @escaping and reject the callers' non-escaping loggers).
    public static func cachedOrCompute(path: String,
                                       onLog: @Sendable (String) -> Void = { _ in }) -> Data? {
        guard let (size, mtime) = stat(path) else { return nil }
        if let hit = lookup(path: path), hit.size == size, hit.mtime == mtime,
           let data = Data(hexString: hit.sha256) {
            return data
        }
        onLog("hash SHA-256 di \((path as NSString).lastPathComponent) "
              + "(\(size / 1_048_576) MB, una tantum)…\n")
        guard let digest = compute(path: path) else { return nil }
        remember(path: path, size: size, mtime: mtime, sha256: digest)
        return digest
    }

    /// Like `cachedOrCompute`, but also returns the v8 chained checkpoint
    /// list (computed in the SAME streaming pass as the whole-file hash and
    /// cached alongside it — an old cache entry without chain pays one re-read).
    public static func cachedOrComputeWithChain(path: String,
                                                onLog: @Sendable (String) -> Void = { _ in })
        -> (sha256: Data, chain: [Data])? {
        guard let (size, mtime) = stat(path) else { return nil }
        if let hit = lookup(path: path), hit.size == size, hit.mtime == mtime,
           let data = Data(hexString: hit.sha256),
           let hexChain = hit.chain {
            let chain = hexChain.compactMap { Data(hexString: $0) }
            if chain.count == hexChain.count { return (data, chain) }
        }
        onLog("hash SHA-256 + checkpoint di \((path as NSString).lastPathComponent) "
              + "(\(size / 1_048_576) MB, una tantum)…\n")
        guard let (digest, chain) = computeWithChain(path: path) else { return nil }
        remember(path: path, size: size, mtime: mtime, sha256: digest, chain: chain)
        return (digest, chain)
    }

    /// Full streaming hash (8 MB reads, F_NOCACHE fd).
    static func compute(path: String) -> Data? {
        computeWithChain(path: path)?.sha
    }

    /// One streaming pass → (whole-file SHA-256, chained checkpoint digests).
    /// Chain: b_k = SHA256(bytes of block k, fileCheckpointBytes each);
    /// chain[0] = b_0, chain[k] = SHA256(chain[k-1] ‖ b_k) — every entry
    /// commits to the entire prefix ("hash concatenati").
    static func computeWithChain(path: String) -> (sha: Data, chain: [Data])? {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { return nil }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)
        var hasher = SHA256()                      // whole file
        var blockHasher = SHA256()                 // current checkpoint block
        var chain: [Data] = []
        var inBlock: UInt64 = 0
        func closeBlock() {
            let b = Data(blockHasher.finalize())
            blockHasher = SHA256()
            var link = SHA256()
            if let prev = chain.last { link.update(data: prev) }
            link.update(data: b)
            chain.append(Data(link.finalize()))
            inBlock = 0
        }
        let bufSize = 8 * 1024 * 1024
        var buf = [UInt8](repeating: 0, count: bufSize)
        while true {
            let n = buf.withUnsafeMutableBytes { read(fd, $0.baseAddress, bufSize) }
            if n < 0 { return nil }
            if n == 0 { break }
            buf.withUnsafeBytes { hasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: $0[0..<n])) }
            // I confini di blocco vanno rispettati ESATTAMENTE anche con read
            // corte (n < bufSize): coordinatore e worker ricalcolano la catena
            // su stream diversi e devono spezzare i blocchi negli stessi punti.
            var off = 0
            while off < n {
                let take = min(Int(Dist.fileCheckpointBytes - inBlock), n - off)
                buf.withUnsafeBytes {
                    blockHasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: $0[off..<off+take]))
                }
                inBlock += UInt64(take); off += take
                if inBlock == Dist.fileCheckpointBytes { closeBlock() }
            }
        }
        if inBlock > 0 { closeBlock() }            // trailing partial block
        return (Data(hasher.finalize()), chain)
    }

    static func stat(_ path: String) -> (size: UInt64, mtime: Double)? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = (attrs[.size] as? NSNumber)?.uint64Value,
              let mtime = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970 else { return nil }
        return (size, mtime)
    }

    private static func lookup(path: String) -> Entry? {
        lock.lock(); defer { lock.unlock() }
        return loadCache()[path]
    }

    private static func remember(path: String, size: UInt64, mtime: Double, sha256: Data,
                                 chain: [Data]? = nil) {
        lock.lock(); defer { lock.unlock() }
        var cache = loadCache()
        cache[path] = Entry(size: size, mtime: mtime, sha256: sha256.hexString,
                            chain: chain.map { $0.map(\.hexString) })
        try? FileManager.default.createDirectory(at: cacheURL.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(cache) { try? data.write(to: cacheURL) }
    }

    private static func loadCache() -> [String: Entry] {
        guard let data = try? Data(contentsOf: cacheURL),
              let cache = try? JSONDecoder().decode([String: Entry].self, from: data) else { return [:] }
        return cache
    }
}

/// Worker-side managed store for coordinator-distributed files. The manifest
/// records (size, sha256) AT RECEPTION — the hash was computed while the bytes
/// streamed in, so later verifications are a lookup, never a re-read.
public struct DistFileStore: Sendable {
    public let directory: URL
    private var manifestURL: URL { directory.appendingPathComponent("manifest.json") }

    /// The store used by the worker (Application Support/DwarfStar/dist-models).
    public static let shared = DistFileStore(
        directory: FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DwarfStar/dist-models", isDirectory: true))

    public init(directory: URL) { self.directory = directory }

    private struct Entry: Codable { var size: UInt64; var sha256: String }

    /// Strip any path components from an offered name: the wire is not trusted
    /// with paths ("../../etc" becomes "etc").
    public static func sanitize(_ name: String) -> String {
        let base = (name as NSString).lastPathComponent
        return (base.isEmpty || base == "." || base == "..") ? "file" : base
    }

    public func url(for name: String) -> URL {
        directory.appendingPathComponent(Self.sanitize(name))
    }

    /// True when the store holds `name` with EXACTLY this size and hash (per
    /// the reception manifest) and the file is still there at that size.
    public func has(name: String, size: UInt64, sha256: Data) -> Bool {
        let key = Self.sanitize(name)
        guard let entry = manifest()[key], entry.size == size,
              entry.sha256 == sha256.hexString,
              let (fileSize, _) = DistFileHash.stat(url(for: key).path),
              fileSize == size else { return false }
        return true
    }

    /// Record a completed reception (called after the streamed hash verified).
    public func remember(name: String, size: UInt64, sha256: Data) {
        var m = manifest()
        m[Self.sanitize(name)] = Entry(size: size, sha256: sha256.hexString)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(m) { try? data.write(to: manifestURL) }
    }

    private func manifest() -> [String: Entry] {
        guard let data = try? Data(contentsOf: manifestURL),
              let m = try? JSONDecoder().decode([String: Entry].self, from: data) else { return [:] }
        return m
    }
}

// MARK: - Hex helpers

extension Data {
    var hexString: String { map { String(format: "%02x", $0) }.joined() }

    init?(hexString: String) {
        guard hexString.count % 2 == 0 else { return nil }
        var bytes: [UInt8] = []
        bytes.reserveCapacity(hexString.count / 2)
        var idx = hexString.startIndex
        while idx < hexString.endIndex {
            let next = hexString.index(idx, offsetBy: 2)
            guard let b = UInt8(hexString[idx..<next], radix: 16) else { return nil }
            bytes.append(b)
            idx = next
        }
        self.init(bytes)
    }
}
