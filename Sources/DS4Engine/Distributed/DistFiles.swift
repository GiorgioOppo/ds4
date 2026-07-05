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

    private struct Entry: Codable { var size: UInt64; var mtime: Double; var sha256: String }

    /// The file's SHA-256, from the cache when (size, mtime) still match, else
    /// computed by streaming the file (F_NOCACHE: hashing must not evict the
    /// hot page cache) and cached. nil if unreadable.
    public static func cachedOrCompute(path: String,
                                       onLog: (@Sendable (String) -> Void)? = nil) -> Data? {
        guard let (size, mtime) = stat(path) else { return nil }
        if let hit = lookup(path: path), hit.size == size, hit.mtime == mtime,
           let data = Data(hexString: hit.sha256) {
            return data
        }
        onLog?("hash SHA-256 di \((path as NSString).lastPathComponent) "
               + "(\(size / 1_048_576) MB, una tantum)…\n")
        guard let digest = compute(path: path) else { return nil }
        remember(path: path, size: size, mtime: mtime, sha256: digest)
        return digest
    }

    /// Full streaming hash (8 MB reads, F_NOCACHE fd).
    static func compute(path: String) -> Data? {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { return nil }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)
        var hasher = SHA256()
        let bufSize = 8 * 1024 * 1024
        var buf = [UInt8](repeating: 0, count: bufSize)
        while true {
            let n = buf.withUnsafeMutableBytes { read(fd, $0.baseAddress, bufSize) }
            if n < 0 { return nil }
            if n == 0 { break }
            buf.withUnsafeBytes { hasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: $0[0..<n])) }
        }
        return Data(hasher.finalize())
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

    private static func remember(path: String, size: UInt64, mtime: Double, sha256: Data) {
        lock.lock(); defer { lock.unlock() }
        var cache = loadCache()
        cache[path] = Entry(size: size, mtime: mtime, sha256: sha256.hexString)
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
