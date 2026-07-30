import Foundation
import DS4Core
import DS4Metal
#if canImport(Darwin)
import Darwin
#endif

/// Prefix-indexed disk checkpoints for Laguna's native `LKV1` cache format.
///
/// The active KV remains in Metal shared memory while inference is running.
/// This store preserves completed prefixes across model/session reloads and
/// lets the engine restore them without repeating prefill. Entries are scoped
/// to one GGUF by `LagunaChatService`; the engine additionally validates the
/// exact file size and cache geometry before publishing restored rows.
public final class LagunaDiskKVStore: @unchecked Sendable, DS4Logging {
    public static let logTag = "lagunakv"

    public struct Options: Sendable {
        /// Laguna's exact F16 checkpoint costs about 192 KiB/token below the
        /// 512-token sliding window. Tiny checkpoints are therefore neither
        /// storage-efficient nor useful enough to retain.
        public var minTokens = 128
        /// Re-checkpoint only after this many new tokens beyond the longest
        /// stored prefix of the same conversation.
        public var storeIntervalTokens = 256
        public init() {}
    }

    public struct Hit: Sendable {
        public let tokens: [Int32]
        public let url: URL
        public let name: String
    }

    public let directory: URL
    public let options: Options
    let budgetTokens: Int
    let budgetBytes: UInt64

    /// Safety ceiling. Actual mature checkpoints are ~48 KiB/token plus a
    /// fixed ~72 MiB sliding-ring payload; 256 KiB/token also covers short
    /// prefixes without allowing malformed writers to fill the volume.
    static let bytesPerTokenCap: UInt64 = 262_144

    struct EntryMetadata: Codable {
        var tokens: Int
        var bytes: UInt64
        var hits: UInt32
        var createdAt: UInt64
        var lastUsed: UInt64
    }

    private let lock = NSLock()
    private var metadata: [String: EntryMetadata] = [:]

    public init(directory: URL, budgetTokens: Int,
                options: Options = Options()) throws {
        self.directory = directory
        self.options = options
        self.budgetTokens = max(0, budgetTokens)
        self.budgetBytes = budgetTokens > 0
            ? UInt64(budgetTokens) * Self.bytesPerTokenCap
            : 4_294_967_296
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        reconcileIndex()
    }

    // MARK: - Lookup

    /// Longest stored strict prefix of `ids`. Only the small header/token
    /// section is read here; Metal planes are streamed by the engine on hit.
    public func findLongestPrefix(of ids: [Int32]) -> Hit? {
        var best: Hit?
        for name in entryNames() {
            let url = directory.appendingPathComponent(name)
            guard let tokens = Self.scanTokens(url) else { continue }
            let bestCount = best?.tokens.count ?? 0
            guard tokens.count >= options.minTokens,
                  tokens.count < ids.count,
                  tokens.count > bestCount,
                  ids.starts(with: tokens) else { continue }
            best = Hit(tokens: tokens, url: url, name: name)
        }
        return best
    }

    public func bumpHit(_ name: String) {
        lock.lock(); defer { lock.unlock() }
        guard var meta = metadata[name] else { return }
        meta.hits &+= 1
        meta.lastUsed = UInt64(Date().timeIntervalSince1970)
        metadata[name] = meta
        saveIndexLocked()
    }

    // MARK: - Store

    public func shouldStore(tokens: [Int32]) -> Bool {
        guard tokens.count >= options.minTokens else { return false }
        let name = Self.entryName(tokens: tokens)
        guard !FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(name).path) else {
            return false
        }
        guard let previous = findLongestPrefix(of: tokens) else { return true }
        return tokens.count - previous.tokens.count
            >= options.storeIntervalTokens
    }

    /// The engine writes an `LKV1` file to the temporary URL. Publication is
    /// atomic and happens only after byte/token budget eviction succeeds.
    @discardableResult
    public func store(tokens: [Int32],
                      writer: (URL) throws -> Void) -> Bool {
        guard tokens.count >= options.minTokens else { return false }
        if budgetTokens > 0, tokens.count > budgetTokens { return false }
        let name = Self.entryName(tokens: tokens)
        let url = directory.appendingPathComponent(name)
        guard !FileManager.default.fileExists(atPath: url.path) else {
            return false
        }
        let started = Date()
        let tmp = directory.appendingPathComponent(name + ".tmp")
        do {
            try writer(tmp)
        } catch {
            try? FileManager.default.removeItem(at: tmp)
            log("checkpoint fallito: \(error)")
            return false
        }
        let attrs = try? FileManager.default
            .attributesOfItem(atPath: tmp.path)
        let bytes = (attrs?[.size] as? NSNumber)?.uint64Value ?? 0
        guard bytes > 0, bytes <= budgetBytes else {
            try? FileManager.default.removeItem(at: tmp)
            return false
        }
        evictToBudget(incomingBytes: bytes, incomingTokens: tokens)
        guard rename(tmp.path, url.path) == 0 else {
            try? FileManager.default.removeItem(at: tmp)
            return false
        }

        let now = UInt64(Date().timeIntervalSince1970)
        lock.lock()
        metadata[name] = EntryMetadata(
            tokens: tokens.count, bytes: bytes, hits: 0,
            createdAt: now, lastUsed: now)
        saveIndexLocked()
        lock.unlock()
        log(String(
            format: "checkpoint %d token (%.0f MB) scritto in %.2fs",
            tokens.count, Double(bytes) / 1_048_576,
            Date().timeIntervalSince(started)))
        return true
    }

    // MARK: - Eviction

    /// Same score and superseded-prefix preference as the DeepSeek/GLM
    /// stores: under pressure, an extended conversation removes its own
    /// shorter checkpoints before useful prefixes from other conversations.
    func evictToBudget(incomingBytes: UInt64,
                       incomingTokens: [Int32]) {
        struct Victim {
            let name: String
            let bytes: UInt64
            let tokens: UInt64
            let lastUsed: UInt64
            let score: Double
        }
        guard incomingBytes <= budgetBytes else { return }
        let now = UInt64(Date().timeIntervalSince1970)
        lock.lock()
        let snapshot = metadata
        lock.unlock()
        var victims: [Victim] = []
        var totalBytes: UInt64 = 0
        var totalTokens: UInt64 = 0
        for (name, meta) in snapshot {
            totalBytes += meta.bytes
            totalTokens += UInt64(meta.tokens)
            var score = KVCFile.evictionScore(
                KVCFile.Entry(
                    hits: meta.hits, tokens: UInt32(meta.tokens),
                    fileSize: meta.bytes, createdAt: meta.createdAt,
                    lastUsed: meta.lastUsed,
                    reason: KVCFile.Reason.continued.rawValue),
                now: now)
            if meta.tokens < incomingTokens.count {
                let url = directory.appendingPathComponent(name)
                if let stored = Self.scanTokens(url),
                   incomingTokens.starts(with: stored) {
                    let hits = Double(meta.hits)
                    let fraction = hits > 0 ? hits / (hits + 1) : 0
                    score *= 0.05 + 0.45 * fraction
                }
            }
            victims.append(Victim(
                name: name, bytes: meta.bytes,
                tokens: UInt64(meta.tokens), lastUsed: meta.lastUsed,
                score: score))
        }

        let byteTarget = budgetBytes - incomingBytes
        let tokenTarget: UInt64 = budgetTokens > 0
            ? UInt64(max(0, budgetTokens - incomingTokens.count)) : .max
        guard totalBytes > byteTarget
                || totalTokens > tokenTarget else { return }
        let order = victims.sorted {
            $0.score != $1.score
                ? $0.score < $1.score : $0.lastUsed < $1.lastUsed
        }
        for victim in order
        where totalBytes > byteTarget || totalTokens > tokenTarget {
            try? FileManager.default.removeItem(
                at: directory.appendingPathComponent(victim.name))
            lock.lock()
            metadata.removeValue(forKey: victim.name)
            saveIndexLocked()
            lock.unlock()
            totalBytes -= min(totalBytes, victim.bytes)
            totalTokens -= min(totalTokens, victim.tokens)
            log("evict \(victim.name) (\(victim.tokens) token)")
        }
    }

    // MARK: - Header scan / naming

    /// `LKV1` header: magic, version, GGUF bytes, layers, token count,
    /// row width, sliding capacity/layer count, reserved, then token ids.
    static func scanTokens(_ url: URL) -> [Int32]? {
        guard let handle = FileHandle(forReadingAtPath: url.path) else {
            return nil
        }
        defer { try? handle.close() }
        guard let header = try? handle.read(upToCount: 40),
              header.count == 40 else { return nil }
        func u32(_ offset: Int) -> UInt32 {
            UInt32(littleEndian: header.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
            })
        }
        guard u32(0) == LagunaResidentModel.kvMagic, u32(4) == 1 else {
            return nil
        }
        let count = Int(u32(20))
        guard count > 0, count <= 1_000_000,
              let tokenData = try? handle.read(upToCount: count * 4),
              tokenData.count == count * 4 else { return nil }
        return tokenData.withUnsafeBytes {
            Array($0.bindMemory(to: Int32.self))
        }
    }

    static func entryName(tokens: [Int32]) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(tokens.count * 4)
        for token in tokens {
            let value = UInt32(bitPattern: token)
            bytes.append(contentsOf: [
                UInt8(value & 0xff), UInt8((value >> 8) & 0xff),
                UInt8((value >> 16) & 0xff),
                UInt8((value >> 24) & 0xff),
            ])
        }
        return KVCFile.sha1Hex(bytes) + ".lagunakv"
    }

    private func entryNames() -> [String] {
        let all = (try? FileManager.default.contentsOfDirectory(
            atPath: directory.path)) ?? []
        return all.filter { $0.hasSuffix(".lagunakv") }
    }

    private var indexURL: URL {
        directory.appendingPathComponent("index.json")
    }

    private func reconcileIndex() {
        if let data = FileManager.default.contents(atPath: indexURL.path),
           let loaded = try? JSONDecoder().decode(
               [String: EntryMetadata].self, from: data) {
            metadata = loaded
        }
        let present = Set(entryNames())
        metadata = metadata.filter { present.contains($0.key) }
        for name in present where metadata[name] == nil {
            let url = directory.appendingPathComponent(name)
            guard let tokens = Self.scanTokens(url) else { continue }
            let attrs = try? FileManager.default
                .attributesOfItem(atPath: url.path)
            let bytes = (attrs?[.size] as? NSNumber)?.uint64Value ?? 0
            let created = (attrs?[.creationDate] as? Date) ?? Date()
            let stamp = UInt64(created.timeIntervalSince1970)
            metadata[name] = EntryMetadata(
                tokens: tokens.count, bytes: bytes, hits: 0,
                createdAt: stamp, lastUsed: stamp)
        }
        lock.lock()
        saveIndexLocked()
        lock.unlock()
    }

    private func saveIndexLocked() {
        guard let data = try? JSONEncoder().encode(metadata) else { return }
        try? data.write(to: indexURL, options: .atomic)
    }
}
