import Foundation
import DS4Core
import DS4Metal

/// Disk-backed KV store for the GLM 5.2 backend — the `DiskKVStore`
/// counterpart over the engine-native `GKV1` checkpoint format: completed
/// generations are checkpointed into a per-model directory keyed by their
/// exact token prefix; a later conversation that starts with a stored prefix
/// restores it and prefills only the rest. Multiple conversations therefore
/// keep their checkpoints side by side instead of evicting each other (the
/// old single `state.glmkv` behavior).
///
/// Layout: `<sha1(token ids)>.glmkv` entries written BY THE ENGINE
/// (`GLM52ResidentModel.saveKVCheckpoint`, ~96 KB/token) — the store frames
/// nothing and only parses the GKV1 header (magic, version, gguf size,
/// layer count, token list) for lookup. Hits/lastUsed live in a sidecar
/// `index.json`, reconciled with the directory at init, so the engine wire
/// format stays untouched. Eviction reuses the ported
/// `KVCFile.evictionScore` under a token budget (byte safety cap derived at
/// 128 KB/token), with the supersede-continued rule: under pressure, the
/// longer checkpoint of the same conversation replaces its own shorter
/// prefixes first.
public final class GLM52DiskKVStore: @unchecked Sendable, DS4Logging {
    public static let logTag = "glmkv"

    public struct Options: Sendable {
        /// Don't checkpoint tiny prefixes (same spirit as DeepSeek's 128;
        /// GLM chats are shorter, matching the historical `fed.count >= 64`).
        public var minTokens = 64
        /// Re-checkpoint only after this many NEW tokens since the longest
        /// stored prefix of the conversation.
        public var storeIntervalTokens = 256
        public init() {}
    }

    public let directory: URL
    public let options: Options
    let budgetTokens: Int
    let budgetBytes: UInt64
    /// Byte safety cap per budgeted token: ~96 KB/token real cost plus
    /// per-entry overhead headroom (DeepSeek uses 32 KB for its format).
    static let bytesPerTokenCap: UInt64 = 131_072

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

    public struct Hit: Sendable {
        public let tokens: [Int32]
        public let url: URL
        public let name: String
    }

    /// The stored entry with the LONGEST token prefix of `ids` (strictly
    /// shorter, ≥ minTokens). Header + token-list scans only; the tensor
    /// body is read by the engine's restore.
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

    /// Bump hits + lastUsed for a restored entry (index only — the engine
    /// file is never rewritten).
    public func bumpHit(_ name: String) {
        lock.lock(); defer { lock.unlock() }
        guard var meta = metadata[name] else { return }
        meta.hits &+= 1
        meta.lastUsed = UInt64(Date().timeIntervalSince1970)
        metadata[name] = meta
        saveIndexLocked()
    }

    // MARK: - Store

    /// Whether `tokens` deserves a new checkpoint: not already stored, and
    /// at least `storeIntervalTokens` beyond the longest stored prefix
    /// (a conversation with NO stored prefix always qualifies — the anchor).
    public func shouldStore(tokens: [Int32]) -> Bool {
        guard tokens.count >= options.minTokens else { return false }
        let name = Self.entryName(tokens: tokens)
        guard !FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(name).path) else {
            return false
        }
        guard let previous = findLongestPrefix(of: tokens) else { return true }
        return tokens.count - previous.tokens.count >= options.storeIntervalTokens
    }

    /// Checkpoint `tokens` with the engine as the writer (it produces the
    /// GKV1 body straight to the given URL). Mirrors the DeepSeek order:
    /// refuse an entry that alone exceeds the budget, evict to make room,
    /// then publish atomically. Returns false when refused or the write fails.
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
        let t0 = Date()
        let tmp = directory.appendingPathComponent(name + ".tmp")
        do {
            try writer(tmp)
        } catch {
            try? FileManager.default.removeItem(at: tmp)
            log("checkpoint fallito: \(error)")
            return false
        }
        let attrs = try? FileManager.default.attributesOfItem(atPath: tmp.path)
        let bytes = (attrs?[.size] as? NSNumber)?.uint64Value ?? 0
        guard bytes > 0, bytes <= budgetBytes else {
            try? FileManager.default.removeItem(at: tmp)
            return false
        }
        evictToBudget(incomingBytes: bytes, incomingTokens: tokens)
        do {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: tmp)
        } catch {
            try? FileManager.default.removeItem(at: tmp)
            return false
        }
        let now = UInt64(Date().timeIntervalSince1970)
        lock.lock()
        metadata[name] = EntryMetadata(tokens: tokens.count, bytes: bytes,
                                       hits: 0, createdAt: now, lastUsed: now)
        saveIndexLocked()
        lock.unlock()
        log(String(format: "checkpoint %d token (%.0f MB) scritto in %.2fs",
                   tokens.count, Double(bytes) / 1_048_576,
                   Date().timeIntervalSince(t0)))
        return true
    }

    /// Adopt the pre-store single-checkpoint file (`state.glmkv`) as a
    /// regular entry, preserving the caches it holds across the upgrade.
    public func adoptLegacyCheckpoint(at legacy: URL) {
        guard FileManager.default.fileExists(atPath: legacy.path),
              let tokens = Self.scanTokens(legacy),
              tokens.count >= options.minTokens else {
            try? FileManager.default.removeItem(at: legacy)
            return
        }
        let name = Self.entryName(tokens: tokens)
        let url = directory.appendingPathComponent(name)
        guard !FileManager.default.fileExists(atPath: url.path) else {
            try? FileManager.default.removeItem(at: legacy)
            return
        }
        guard (try? FileManager.default.moveItem(at: legacy, to: url)) != nil else {
            return
        }
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        let bytes = (attrs?[.size] as? NSNumber)?.uint64Value ?? 0
        let now = UInt64(Date().timeIntervalSince1970)
        lock.lock()
        metadata[name] = EntryMetadata(tokens: tokens.count, bytes: bytes,
                                       hits: 0, createdAt: now, lastUsed: now)
        saveIndexLocked()
        lock.unlock()
        log("checkpoint legacy adottato (\(tokens.count) token)")
    }

    // MARK: - Eviction

    /// Evict lowest-score entries until the directory fits both the byte
    /// and the token budget with `incoming` accounted for. Score = ported
    /// `KVCFile.evictionScore`, dampened ×(0.05 + 0.45·h/(h+1)) when the
    /// entry is a strict token-prefix of the incoming checkpoint (the
    /// supersede-continued rule). Ties evict the older lastUsed.
    func evictToBudget(incomingBytes: UInt64, incomingTokens: [Int32]) {
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
                KVCFile.Entry(hits: meta.hits, tokens: UInt32(meta.tokens),
                              fileSize: meta.bytes, createdAt: meta.createdAt,
                              lastUsed: meta.lastUsed,
                              reason: KVCFile.Reason.continued.rawValue),
                now: now)
            if meta.tokens < incomingTokens.count {
                let url = directory.appendingPathComponent(name)
                if let stored = Self.scanTokens(url),
                   incomingTokens.starts(with: stored) {
                    let hits = Double(meta.hits)
                    let hFrac = hits > 0 ? hits / (hits + 1.0) : 0.0
                    score *= 0.05 + 0.45 * hFrac
                }
            }
            victims.append(Victim(name: name, bytes: meta.bytes,
                                  tokens: UInt64(meta.tokens),
                                  lastUsed: meta.lastUsed, score: score))
        }
        let byteTarget = budgetBytes - incomingBytes
        let tokenTarget: UInt64 = budgetTokens > 0
            ? UInt64(max(0, budgetTokens - incomingTokens.count)) : .max
        guard totalBytes > byteTarget || totalTokens > tokenTarget else { return }
        let order = victims.sorted {
            $0.score != $1.score ? $0.score < $1.score : $0.lastUsed < $1.lastUsed
        }
        for victim in order where totalBytes > byteTarget || totalTokens > tokenTarget {
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

    /// Parse the engine's GKV1 header — magic, version, gguf byte size,
    /// layer count, token list. Model identity is NOT verified here (the
    /// directory is already scoped per GGUF; the engine re-validates on
    /// restore); nil for foreign or malformed files.
    static func scanTokens(_ url: URL) -> [Int32]? {
        guard let handle = FileHandle(forReadingAtPath: url.path) else {
            return nil
        }
        defer { try? handle.close() }
        guard let header = try? handle.read(upToCount: 24),
              header.count == 24 else { return nil }
        func u32(_ offset: Int) -> UInt32 {
            header.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
            }
        }
        guard u32(0) == GLM52ResidentModel.kvMagic, u32(4) == 1 else {
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
        for t in tokens {
            let v = UInt32(bitPattern: t)
            bytes.append(contentsOf: [UInt8(v & 0xff), UInt8((v >> 8) & 0xff),
                                      UInt8((v >> 16) & 0xff),
                                      UInt8((v >> 24) & 0xff)])
        }
        return KVCFile.sha1Hex(bytes) + ".glmkv"
    }

    private func entryNames() -> [String] {
        let all = (try? FileManager.default.contentsOfDirectory(
            atPath: directory.path)) ?? []
        return all.filter {
            $0.hasSuffix(".glmkv") && $0 != Self.legacyFileName
        }
    }

    static let legacyFileName = "state.glmkv"
    private var indexURL: URL { directory.appendingPathComponent("index.json") }

    /// Load `index.json` and reconcile with the directory: drop metadata for
    /// vanished files, adopt entries the index doesn't know (crash between
    /// rename and index write, or a hand-copied checkpoint) with zeroed hits
    /// and filesystem dates.
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
            let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
            let bytes = (attrs?[.size] as? NSNumber)?.uint64Value ?? 0
            let created = (attrs?[.creationDate] as? Date) ?? Date()
            let stamp = UInt64(created.timeIntervalSince1970)
            metadata[name] = EntryMetadata(tokens: tokens.count, bytes: bytes,
                                           hits: 0, createdAt: stamp,
                                           lastUsed: stamp)
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
