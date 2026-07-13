import Foundation
import DS4Core
import DS4Metal

extension DiskKVStore {
    // MARK: Store

    /// Hands a snapshot to a (typically background) store call with UNIQUE
    /// ownership: `take()` empties the box, so the writer is the only owner and
    /// can drop each layer's slabs the moment they are on disk — RAM falls
    /// DURING the write instead of holding the whole checkpoint until the end.
    public final class SnapshotBox: @unchecked Sendable {
        private let lock = NSLock()
        private var snapshot: KVSnapshot?
        public init(_ snapshot: KVSnapshot) { self.snapshot = snapshot }
        func take() -> KVSnapshot? {
            lock.lock(); defer { lock.unlock() }
            let s = snapshot; snapshot = nil; return s
        }
    }

    /// Checkpoint from a SnapshotBox (unique ownership → progressive per-layer
    /// release while writing). This is the path the engine uses for its
    /// background checkpoint task.
    @discardableResult
    public func store(tokens: [Int], modelName: String, box: SnapshotBox,
                      reason: KVCFile.Reason = .continued) -> Bool {
        guard var snap = box.take() else { return false }
        return store(tokens: tokens, modelName: modelName, taking: &snap, reason: reason)
    }

    /// Checkpoint `tokens`+`snapshot` (dedup by content name). Mirrors the C
    /// store_live_prefix order: refuse an entry that alone exceeds the budget,
    /// EVICT FIRST to make room for the incoming bytes (with the supersede-
    /// continued scoring), then write atomically. `reason` "cold" marks the
    /// first checkpoint of a conversation (anchor, 2× protected in eviction).
    @discardableResult
    public func store(tokens: [Int], modelName: String, snapshot: KVSnapshot,
                      reason: KVCFile.Reason = .continued) -> Bool {
        var snap = snapshot
        return store(tokens: tokens, modelName: modelName, taking: &snap, reason: reason)
    }

    /// Core writer. `snapshot` is inout so each layer's slot can be emptied as
    /// soon as it is written: when the caller's copy is the only reference
    /// (SnapshotBox path), that frees the slab immediately.
    private func store(tokens: [Int], modelName: String, taking snapshot: inout KVSnapshot,
                       reason: KVCFile.Reason) -> Bool {
        guard tokens.count >= options.minTokens, snapshot.nKeys == tokens.count else { return false }
        let url = directory.appendingPathComponent(entryName(tokens: tokens, modelName: modelName))
        guard !FileManager.default.fileExists(atPath: url.path) else { return false }
        let t0 = Date()

        // Small prefix (name + token ids + payload header) built in RAM; the
        // large per-layer float slabs are STREAMED to the file below — no more
        // whole-checkpoint Data in memory (hundreds of MB at long contexts).
        var prefix = Data()
        appendU32(&prefix, UInt32(Data(modelName.utf8).count)); prefix.append(Data(modelName.utf8))
        appendU32(&prefix, UInt32(tokens.count))
        for t in tokens { appendU32(&prefix, UInt32(truncatingIfNeeded: t)) }
        let ph = DSV4PayloadHeader(
            savedContextSize: UInt32(contextSize), prefillChunk: 512,
            rawKVCapacity: UInt32(contextSize), rawSlidingWindow: 128,
            compressedKVCapacity: 0, checkpointTokenCount: UInt32(tokens.count),
            layerCount: UInt32(snapshot.layers.count), rawHeadKVDim: UInt32(snapshot.headDim),
            indexerHeadDim: 128, vocabSize: 0,
            liveRawRows: UInt32(snapshot.layers.first.map { snapshot.nKeys - $0.rawStart } ?? 0))
        prefix.append(contentsOf: ph.serialize())

        // Exact payload size up front (same wire format as the old one-shot
        // Data build) so header, budget checks and eviction see the true bytes.
        func compBytes(_ c: CompSnapshot?) -> Int {
            guard let c else { return 1 }
            return 1 + 4 + 4 + c.stateKv.count * 4 + c.stateScore.count * 4 + 4 + c.cacheRows.count * 4
        }
        var payload = prefix.count
        for layer in snapshot.layers {
            payload += 8 + layer.raw.count * 4 + compBytes(layer.comp) + compBytes(layer.idx)
        }
        let now = UInt64(Date().timeIntervalSince1970)
        let header = KVCFile.fillHeader(KVCFile.Header(
            quantBits: quantBits, reason: reason.rawValue, extFlags: 0, modelId: 0,
            tokens: UInt32(tokens.count), hits: 0, ctxSize: UInt32(contextSize),
            createdAt: now, lastUsed: now, payloadBytes: UInt64(payload)))
        let total = UInt64(header.count + payload)
        guard total <= budgetBytes else { return false }   // can never fit
        if budgetTokens > 0, tokens.count > budgetTokens { return false }
        evictToBudget(incomingBytes: total, incomingTokens: tokens, incomingModel: modelName)

        // Stream with F_NOCACHE: the checkpoint's dirty pages must not evict
        // the HOT page cache (dense weights / expert bundle) — that eviction
        // made the NEXT turn re-read everything from disk (gather bandwidth
        // collapse in the GUI, which checkpoints; the demo doesn't). .tmp +
        // rename keeps the entry atomic like Data(.atomic) did.
        let tmp = url.path + ".tmp"
        let fd = open(tmp, O_CREAT | O_TRUNC | O_WRONLY, 0o644)
        guard fd >= 0 else { return false }
        _ = fcntl(fd, F_NOCACHE, 1)
        var ok = true
        func writeAll(_ p: UnsafeRawPointer, _ n: Int) {
            guard ok else { return }
            var off = 0
            while off < n {
                let w = write(fd, p + off, n - off)
                if w <= 0 { ok = false; return }
                off += w
            }
        }
        func writeData(_ d: Data) { d.withUnsafeBytes { writeAll($0.baseAddress!, $0.count) } }
        func writeFloats(_ a: [Float]) { a.withUnsafeBufferPointer { writeAll($0.baseAddress!, $0.count * 4) } }
        func writeComp(_ c: CompSnapshot?) {
            guard let c else { var z: UInt8 = 0; writeAll(&z, 1); return }
            var d = Data(); d.append(1)
            appendU32(&d, UInt32(c.count)); appendU32(&d, UInt32(c.stateKv.count))
            writeData(d)
            writeFloats(c.stateKv); writeFloats(c.stateScore)
            var n = Data(); appendU32(&n, UInt32(c.cacheRows.count)); writeData(n)
            writeFloats(c.cacheRows)
        }
        writeData(Data(header))
        writeData(prefix)
        // Indexed loop (NOT for-in, which would retain every slab for the whole
        // write): each layer's slot is emptied right after its bytes are out,
        // so a uniquely-owned snapshot releases memory as the write progresses.
        for i in 0..<snapshot.layers.count {
            let layer = snapshot.layers[i]
            var lh = Data()
            appendU32(&lh, UInt32(layer.rawStart)); appendU32(&lh, UInt32(layer.raw.count))
            writeData(lh)
            writeFloats(layer.raw)
            writeComp(layer.comp)
            writeComp(layer.idx)     // NSA indexer compressor (ratio-4 layers)
            snapshot.layers[i] = KVLayerSnapshot(rawStart: 0, raw: [], comp: nil)
        }
        close(fd)
        if ok { ok = (rename(tmp, url.path) == 0) }
        if !ok { unlink(tmp); return false }
        FileHandle.standardError.write(Data(String(
            format: "DS4 diskkv: checkpoint %d token (%.0f MB) scritto in %.2fs (F_NOCACHE)\n",
            tokens.count, Double(total) / 1_048_576, Date().timeIntervalSince(t0)).utf8))
        return true
    }

    /// Evict lowest-score entries until the directory fits `budget − incomingBytes`
    /// AND (when a token budget is set) `budgetTokens − incomingTokens.count`.
    /// Score = ported `KVCFile.evictionScore` ×(0.05 + 0.45·h/(h+1)) when the entry
    /// is a CONTINUED strict token-prefix of the incoming checkpoint (the C
    /// supersede-continued rule: the longer checkpoint of the same conversation
    /// replaces the shorter one under pressure). Ties evict the older lastUsed.
    func evictToBudget(incomingBytes: UInt64, incomingTokens: [Int], incomingModel: String) {
        struct Victim { let url: URL; let size: UInt64; let tokens: UInt64; let lastUsed: UInt64; let score: Double }
        var victims: [Victim] = []
        var total: UInt64 = 0
        var totalTokens: UInt64 = 0
        let now = UInt64(Date().timeIntervalSince1970)
        for url in entryURLs() {
            guard let h = readHeader(url) else { continue }
            let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
            let sz = (attrs?[.size] as? NSNumber)?.uint64Value ?? 0
            total += sz
            totalTokens += UInt64(h.tokens)
            var score = KVCFile.evictionScore(
                KVCFile.Entry(hits: h.hits, tokens: h.tokens, fileSize: sz,
                              createdAt: h.createdAt, lastUsed: h.lastUsed, reason: h.reason),
                now: now)
            if h.reason == KVCFile.Reason.continued.rawValue,
               Int(h.tokens) < incomingTokens.count,
               let scan = scanEntry(url), scan.model == incomingModel,
               incomingTokens.starts(with: scan.tokens) {
                let hits = Double(h.hits)
                let hFrac = hits > 0 ? hits / (hits + 1.0) : 0.0
                score *= 0.05 + 0.45 * hFrac
            }
            victims.append(Victim(url: url, size: sz, tokens: UInt64(h.tokens),
                                  lastUsed: h.lastUsed, score: score))
        }
        guard incomingBytes <= budgetBytes else { return }
        let target = budgetBytes - incomingBytes
        let tokenTarget: UInt64 = budgetTokens > 0
            ? UInt64(max(0, budgetTokens - incomingTokens.count)) : .max
        guard total > target || totalTokens > tokenTarget else { return }
        let order = victims.sorted {
            $0.score != $1.score ? $0.score < $1.score : $0.lastUsed < $1.lastUsed
        }
        for v in order where total > target || totalTokens > tokenTarget {
            try? FileManager.default.removeItem(at: v.url)
            total -= min(total, v.size)
            totalTokens -= min(totalTokens, v.tokens)
        }
    }

}
