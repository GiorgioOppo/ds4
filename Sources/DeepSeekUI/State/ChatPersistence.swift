import Foundation

/// Disk IO for the v2 lazy-loading chat layout. Manages four classes
/// of files (`chat.json`, `turns/{turnID}.json`, `turns/{turnID}/
/// rounds/{roundID}.json`, `pending.json`) plus the binary
/// `chat.tokens` blob. Each class has its own debounced writer keyed
/// by the affected ID, so a per-token mutation only re-encodes the
/// active round file and the small `pending.json` snapshot — not the
/// whole transcript. Atomic via `FileManager.replaceItemAt`, so a
/// crash mid-write leaves the previous on-disk state untouched.
///
/// Round payloads are cached in a pinning LRU so toggling a
/// disclosure open / closed doesn't re-decode the file every time;
/// the active streaming round is held in a separate slot outside the
/// LRU to guarantee it's never evicted.
///
/// This type is intentionally decoupled from `ChatStore` — it
/// receives IDs and `Codable` values, never touches `@Published`
/// state. The store wires it up at PR 2; for PR 1 it's dead code
/// covered only by unit tests.
@MainActor
final class ChatPersistence {
    /// Debounce windows per file class. Pending is shorter because it
    /// absorbs per-token writes; the others coalesce on logical
    /// turn / round boundaries.
    struct DebounceConfig {
        var manifest: TimeInterval = 0.5
        var turn:     TimeInterval = 0.5
        var round:    TimeInterval = 0.5
        var pending:  TimeInterval = 0.2
    }

    private var manifestTasks: [UUID: Task<Void, Never>] = [:]
    private var turnTasks:     [TurnKey: Task<Void, Never>] = [:]
    private var roundTasks:    [RoundKey: Task<Void, Never>] = [:]
    private var pendingTasks:  [UUID: Task<Void, Never>] = [:]

    /// Snapshot the persistence layer hands out to the debounce task.
    /// Captured by value at schedule time so the async write can run
    /// without holding back the main actor.
    private var manifests: [UUID: ChatManifest] = [:]
    private var summaries: [TurnKey: TurnSummary] = [:]
    private var rounds:    [RoundKey: StoredRound] = [:]
    private var pending:   [UUID: PendingSnapshot?] = [:]

    let debounce: DebounceConfig

    init(debounce: DebounceConfig = DebounceConfig()) {
        self.debounce = debounce
    }

    // MARK: - schedulers (debounced writes)

    /// Stage `manifest` for write to `chat.json`. Subsequent calls
    /// with the same chat id REPLACE the staged value, so only the
    /// latest snapshot ever lands on disk.
    func scheduleManifestSave(_ manifest: ChatManifest) {
        let id = manifest.id
        manifests[id] = manifest
        manifestTasks[id]?.cancel()
        manifestTasks[id] = Task { [weak self, debounce] in
            try? await Task.sleep(
                nanoseconds: UInt64(debounce.manifest * 1_000_000_000))
            if Task.isCancelled { return }
            await self?.flushManifest(id: id)
        }
    }

    func scheduleTurnSummarySave(chatID: UUID, _ summary: TurnSummary) {
        let key = TurnKey(chatID: chatID, turnID: summary.id)
        summaries[key] = summary
        turnTasks[key]?.cancel()
        turnTasks[key] = Task { [weak self, debounce] in
            try? await Task.sleep(
                nanoseconds: UInt64(debounce.turn * 1_000_000_000))
            if Task.isCancelled { return }
            await self?.flushTurn(key: key)
        }
    }

    func scheduleRoundSave(chatID: UUID, turnID: UUID, _ round: StoredRound) {
        let key = RoundKey(chatID: chatID, turnID: turnID, roundID: round.id)
        rounds[key] = round
        roundTasks[key]?.cancel()
        roundTasks[key] = Task { [weak self, debounce] in
            try? await Task.sleep(
                nanoseconds: UInt64(debounce.round * 1_000_000_000))
            if Task.isCancelled { return }
            await self?.flushRound(key: key)
        }
    }

    /// Stage a pending-snapshot write. Pass `nil` to clear the
    /// snapshot (`.done` path): the next flush deletes `pending.json`
    /// instead of writing a value.
    func schedulePendingSave(chatID: UUID, snapshot: PendingSnapshot?) {
        pending[chatID] = snapshot
        pendingTasks[chatID]?.cancel()
        pendingTasks[chatID] = Task { [weak self, debounce] in
            try? await Task.sleep(
                nanoseconds: UInt64(debounce.pending * 1_000_000_000))
            if Task.isCancelled { return }
            await self?.flushPending(chatID: chatID)
        }
    }

    /// Synchronous, non-debounced manifest write. Used at chat
    /// creation so the v2 folder + `chat.json` exist immediately —
    /// without this, the next `flushSave` would see `isV2Chat == false`
    /// for the 500 ms debounce window and skip the dual-write.
    func writeManifestImmediate(_ manifest: ChatManifest) throws {
        // Drop any debounced write for the same id so we don't
        // race a stale snapshot over the freshly-written one.
        manifestTasks.removeValue(forKey: manifest.id)?.cancel()
        manifests.removeValue(forKey: manifest.id)
        let data = try Self.encoder().encode(manifest)
        let url = try PersistencePaths.chatManifestURL(id: manifest.id)
        try Self.atomicWrite(data, to: url)
    }

    /// Drain every staged write synchronously. Called on app quit and
    /// from tests that need to observe the on-disk state immediately
    /// after a schedule call.
    func flushAll() async {
        for id in manifestTasks.keys { manifestTasks[id]?.cancel() }
        for k in turnTasks.keys      { turnTasks[k]?.cancel() }
        for k in roundTasks.keys     { roundTasks[k]?.cancel() }
        for id in pendingTasks.keys  { pendingTasks[id]?.cancel() }

        for id in Array(manifests.keys) { await flushManifest(id: id) }
        for key in Array(summaries.keys) { await flushTurn(key: key) }
        for key in Array(rounds.keys)    { await flushRound(key: key) }
        for id in Array(pending.keys)    { await flushPending(chatID: id) }
    }

    // MARK: - flush implementations

    private func flushManifest(id: UUID) async {
        guard let manifest = manifests.removeValue(forKey: id) else { return }
        manifestTasks.removeValue(forKey: id)
        do {
            let data = try Self.encoder().encode(manifest)
            let url = try PersistencePaths.chatManifestURL(id: id)
            try Self.atomicWrite(data, to: url)
        } catch {
            // Persist errors are non-fatal — callers can re-schedule.
            // We deliberately swallow here so a transient FS hiccup
            // doesn't take the chat down.
        }
    }

    private func flushTurn(key: TurnKey) async {
        guard let summary = summaries.removeValue(forKey: key) else { return }
        turnTasks.removeValue(forKey: key)
        do {
            let data = try Self.encoder().encode(summary)
            let url = try PersistencePaths.turnSummaryURL(
                chatID: key.chatID, turnID: key.turnID)
            try Self.atomicWrite(data, to: url)
        } catch { }
    }

    private func flushRound(key: RoundKey) async {
        guard let round = rounds.removeValue(forKey: key) else { return }
        roundTasks.removeValue(forKey: key)
        do {
            let data = try Self.encoder().encode(round)
            let url = try PersistencePaths.roundURL(
                chatID: key.chatID,
                turnID: key.turnID,
                roundID: key.roundID)
            try Self.atomicWrite(data, to: url)
        } catch { }
    }

    private func flushPending(chatID: UUID) async {
        let snapshotOpt = pending.removeValue(forKey: chatID)
        pendingTasks.removeValue(forKey: chatID)
        do {
            let url = try PersistencePaths.chatPendingURL(id: chatID)
            // snapshot == nil  → clear the file.
            // snapshot == .some(nil) here means the dict value was
            // explicitly set to nil — same semantics as "clear".
            if let snapshot = snapshotOpt.flatMap({ $0 }) {
                let data = try Self.encoder().encode(snapshot)
                try Self.atomicWrite(data, to: url)
            } else if FileManager.default.fileExists(atPath: url.path) {
                try? FileManager.default.removeItem(at: url)
            }
        } catch { }
    }

    // MARK: - reads (synchronous, fast — used at launch and on lazy load)

    func readManifest(id: UUID) throws -> ChatManifest {
        let url = try PersistencePaths.chatManifestURL(id: id)
        let data = try Data(contentsOf: url)
        return try Self.decoder().decode(ChatManifest.self, from: data)
    }

    func readTurnSummary(chatID: UUID, turnID: UUID) throws -> TurnSummary {
        let url = try PersistencePaths.turnSummaryURL(
            chatID: chatID, turnID: turnID)
        let data = try Data(contentsOf: url)
        return try Self.decoder().decode(TurnSummary.self, from: data)
    }

    func readRound(chatID: UUID, turnID: UUID, roundID: UUID) throws -> StoredRound {
        let url = try PersistencePaths.roundURL(
            chatID: chatID, turnID: turnID, roundID: roundID)
        let data = try Data(contentsOf: url)
        return try Self.decoder().decode(StoredRound.self, from: data)
    }

    func readPending(chatID: UUID) -> PendingSnapshot? {
        guard let url = try? PersistencePaths.chatPendingURL(id: chatID),
              FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url),
              let snap = try? Self.decoder().decode(
                PendingSnapshot.self, from: data)
        else { return nil }
        return snap
    }

    // MARK: - binary `chat.tokens`

    /// Magic prefix written at the head of `chat.tokens`. Lets us
    /// detect file corruption (or accidental writes from a different
    /// codec) without parsing the body.
    static let tokensMagic: [UInt8] = [0x44, 0x53, 0x54, 0x4B] // "DSTK"

    /// Write `tokens` as a little-endian `Int32` payload prefixed by
    /// the magic + a 4-byte count. Not debounced — called at most
    /// once per turn `.done` so the cost is amortised.
    func writeChatTokens(_ chatID: UUID, tokens: [Int32]) throws {
        var data = Data()
        data.append(contentsOf: Self.tokensMagic)
        var count = UInt32(tokens.count).littleEndian
        withUnsafeBytes(of: &count) { data.append(contentsOf: $0) }
        for var t in tokens {
            t = t.littleEndian
            withUnsafeBytes(of: &t) { data.append(contentsOf: $0) }
        }
        let url = try PersistencePaths.chatTokensURL(id: chatID)
        try Self.atomicWrite(data, to: url)
    }

    /// Read back `chat.tokens`. Returns nil when absent (e.g. remote
    /// chat, or fresh chat with no turns yet) or when the magic
    /// prefix doesn't match — callers fall back to a cold re-encode.
    func readChatTokens(_ chatID: UUID) -> [Int32]? {
        guard let url = try? PersistencePaths.chatTokensURL(id: chatID),
              let data = try? Data(contentsOf: url),
              data.count >= 8 else { return nil }
        let magic = Array(data.prefix(4))
        guard magic == Self.tokensMagic else { return nil }
        let count: UInt32 = data.subdata(in: 4..<8).withUnsafeBytes { ptr in
            UInt32(littleEndian: ptr.load(as: UInt32.self))
        }
        let expectedBytes = 8 + Int(count) * 4
        guard data.count == expectedBytes else { return nil }
        var out: [Int32] = []
        out.reserveCapacity(Int(count))
        var offset = 8
        for _ in 0..<Int(count) {
            let val: Int32 = data.subdata(in: offset..<(offset + 4))
                .withUnsafeBytes { ptr in
                    Int32(littleEndian: ptr.load(as: Int32.self))
                }
            out.append(val)
            offset += 4
        }
        return out
    }

    /// Wipe the entire on-disk footprint of one chat. Used by
    /// `ChatStore.delete(_:)` for v2 chats — a single FS call removes
    /// the manifest, all turn summaries, all rounds, the pending
    /// snapshot, the KV cache, and the tokens blob.
    func deleteChat(id: UUID) throws {
        // Cancel any pending writes for this chat first so a debounce
        // task can't recreate the folder after we remove it.
        manifestTasks.removeValue(forKey: id)?.cancel()
        pendingTasks.removeValue(forKey: id)?.cancel()
        for key in Array(turnTasks.keys) where key.chatID == id {
            turnTasks.removeValue(forKey: key)?.cancel()
            summaries.removeValue(forKey: key)
        }
        for key in Array(roundTasks.keys) where key.chatID == id {
            roundTasks.removeValue(forKey: key)?.cancel()
            rounds.removeValue(forKey: key)
        }
        manifests.removeValue(forKey: id)
        pending.removeValue(forKey: id)

        // Avoid `chatDir(id:)` here — it would create the folder
        // we're about to delete on chats that never had one (i.e.
        // legacy chats that `ChatStore.delete` also routes through
        // this method).
        let dir = try PersistencePaths.conversationsDir()
            .appendingPathComponent(id.uuidString, isDirectory: true)
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
        }
    }

    // MARK: - encoders / atomic write

    private static func encoder() -> JSONEncoder {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        // Pretty + sorted keys keep diffs deterministic on disk for
        // debugging; the cost is negligible relative to file IO.
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        return e
    }

    private static func decoder() -> JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }

    /// Write `data` to `url` so that either the new contents are
    /// fully visible or the previous on-disk state is preserved.
    /// Uses `replaceItemAt`, which honours the underlying FS's
    /// inode-preserving semantics where supported (APFS does).
    static func atomicWrite(_ data: Data, to url: URL) throws {
        // Ensure the parent dir exists — for a fresh round write the
        // `turns/{turnID}/rounds/` folder may not have been created
        // yet if the caller skipped `PersistencePaths.roundsDir`.
        let parent = url.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: parent.path) {
            try FileManager.default.createDirectory(
                at: parent, withIntermediateDirectories: true)
        }
        let tmp = url
            .deletingLastPathComponent()
            .appendingPathComponent(
                "." + url.lastPathComponent + ".tmp.\(UUID().uuidString)")
        try data.write(to: tmp, options: .atomic)
        defer { try? FileManager.default.removeItem(at: tmp) }
        if FileManager.default.fileExists(atPath: url.path) {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: tmp)
        } else {
            try FileManager.default.moveItem(at: tmp, to: url)
        }
    }
}

/// Composite key for a turn summary — `(chatID, turnID)`. The
/// `RoundKey` analog for rounds lives in `ChatManifest.swift` because
/// it also feeds the round LRU cache used by views.
struct TurnKey: Hashable {
    let chatID: UUID
    let turnID: UUID
}

// MARK: - Round LRU

/// Pinning LRU keyed by `RoundKey`. Capacity-bounded eviction skips
/// pinned entries; the streaming hot path pins the active round so a
/// burst of disclosure opens elsewhere can't evict the in-flight
/// generation buffer mid-token.
///
/// Order is tracked via a simple Int64 counter so we avoid the
/// linked-list bookkeeping for ~64-entry caches where O(N) eviction
/// scans are still microseconds.
struct RoundLRUCache {
    private var values: [RoundKey: StoredRound] = [:]
    private var ages:   [RoundKey: Int64] = [:]
    private var pins:   Set<RoundKey> = []
    private var clock:  Int64 = 0
    let capacity: Int

    init(capacity: Int = 64) {
        self.capacity = max(1, capacity)
    }

    mutating func get(_ key: RoundKey) -> StoredRound? {
        guard let v = values[key] else { return nil }
        clock &+= 1
        ages[key] = clock
        return v
    }

    mutating func put(_ key: RoundKey, _ value: StoredRound) {
        values[key] = value
        clock &+= 1
        ages[key] = clock
        evictIfNeeded()
    }

    mutating func pin(_ key: RoundKey) {
        pins.insert(key)
    }

    mutating func unpin(_ key: RoundKey) {
        pins.remove(key)
    }

    mutating func remove(_ key: RoundKey) {
        values.removeValue(forKey: key)
        ages.removeValue(forKey: key)
        pins.remove(key)
    }

    var count: Int { values.count }
    func contains(_ key: RoundKey) -> Bool { values[key] != nil }
    func isPinned(_ key: RoundKey) -> Bool { pins.contains(key) }
    /// Snapshot of every key currently cached, in unspecified
    /// order. Useful for batch eviction on chat delete: walk the
    /// snapshot, filter by `chatID`, call `remove(_:)` per match.
    /// The returned array doesn't keep the cache live — safe to
    /// iterate while mutating.
    var allKeys: [RoundKey] { Array(values.keys) }

    private mutating func evictIfNeeded() {
        while values.count > capacity {
            let candidate = ages
                .lazy
                .filter { !self.pins.contains($0.key) }
                .min(by: { $0.value < $1.value })?.key
            guard let victim = candidate else {
                // Every entry pinned — cannot evict. The cache will
                // grow past `capacity` until something is unpinned.
                return
            }
            remove(victim)
        }
    }
}
