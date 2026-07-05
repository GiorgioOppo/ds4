import Foundation
import DS4Core
import DS4Metal

/// Disk-backed KV cache, modelled on ds4_kvstore.c: completed-generation
/// checkpoints are written to a directory keyed by their exact token prefix;
/// a later conversation (or a stateless HTTP request re-sending the transcript)
/// that starts with a stored prefix RESTORES it and prefills only the rest.
///
/// File layout (Swift-defined body behind the ported KVC/DSV4 headers):
///   [48B KVC header]  (KVCFile: magic, quant, tokens, ctx, hits, timestamps)
///   [u32 nameLen][model name utf8]
///   [u32 nTokens][nTokens × u32 token ids]
///   [52B DSV4PayloadHeader]
///   per layer: [u32 rawStart][u32 rawFloats][raw f32…]
///              [u8 hasComp]( [u32 count][u32 stateLen][stateKv f32…]
///                            [stateScore f32…][u32 cacheFloats][cache f32…] )
/// Eviction uses the ported `KVCFile.evictionScore` under a byte budget;
/// hits/lastUsed are bumped in-place on every restore (the 48B header only).
///
/// RAM discipline, both directions: restore STREAMS the file one layer at a
/// time into the decoder (each batch is parsed, imported, and freed before the
/// next is read — peak = one layer, never the whole checkpoint), and store
/// writes from a uniquely-owned snapshot whose layers are dropped as they hit
/// the disk. Both sides read/write with F_NOCACHE so checkpoint bytes never
/// displace the hot page cache (dense weights / expert bundle).
public final class DiskKVStore: @unchecked Sendable {
    public struct Options: Sendable {
        /// Don't checkpoint tiny prefixes (C default is 512; local chats have
        /// shorter useful prefixes, so we default lower).
        public var minTokens = 128
        /// Re-checkpoint only after this many NEW tokens since the last store.
        public var storeIntervalTokens = 256
        public init() {}
    }

    public let directory: URL
    public let options: Options
    private let budgetBytes: UInt64
    /// Total TOKEN budget across all stored entries (0 = byte budget only).
    /// Tokens are the natural unit here — "keep up to 1M tokens of checkpoints"
    /// — and per-token bytes vary with the model, so eviction counts tokens.
    private let budgetTokens: Int
    private let quantBits: UInt8
    private let contextSize: Int

    public init(directory: URL, budgetMB: Int, quantBits: UInt8, contextSize: Int,
                budgetTokens: Int = 0, options: Options = Options()) throws {
        self.directory = directory
        self.budgetTokens = max(0, budgetTokens)
        // Token-budgeted stores derive a generous byte SAFETY cap from the token
        // budget (~32 KB/token upper bound incl. per-entry overhead) so an
        // unexpected entry mix still can't grow the directory without bound.
        self.budgetBytes = budgetTokens > 0
            ? UInt64(budgetTokens) * 32_768
            : UInt64(max(64, budgetMB)) * 1_048_576
        self.quantBits = quantBits == 2 ? 2 : 4    // header validity wants {2,4}
        self.contextSize = contextSize
        self.options = options
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    // MARK: Lookup

    /// A matched entry: the token prefix plus WHERE it lives. The tensor body is
    /// NOT loaded here — restore streams it into the decoder one layer at a time.
    public struct Hit {
        public let tokens: [Int]
        let url: URL
    }

    /// Find the stored entry with the LONGEST token prefix of `ids` (strictly
    /// shorter than `ids`, ≥ minTokens, same model, fits this context). Cheap:
    /// scans headers + token lists only; the body is read by `restore`.
    public func findLongestPrefix(of ids: [Int], modelName: String) -> Hit? {
        var best: Hit?
        for url in entryURLs() {
            guard let scan = scanEntry(url) else { continue }
            // Local, NOT inline: `count < contextSize, … count > (…)` would be
            // parsed as a generic specialization `count<…>(…)` by Swift.
            let bestCount = best?.tokens.count ?? 0
            guard scan.model == modelName,
                  scan.tokens.count >= options.minTokens,
                  scan.tokens.count < ids.count,
                  scan.tokens.count < contextSize,
                  scan.tokens.count > bestCount,
                  ids.starts(with: scan.tokens) else { continue }
            best = Hit(tokens: scan.tokens, url: url)
        }
        return best
    }

    /// ALL stored entry lengths that are strict token-prefixes of `ids` (for the
    /// distributed restore negotiation: the coordinator intersects each worker's
    /// set and picks the longest length EVERY shard can restore). Sorted
    /// descending; header/token scans only, no tensor bodies.
    public func storedPrefixLengths(of ids: [Int], modelName: String) -> [Int] {
        var lengths: [Int] = []
        for url in entryURLs() {
            guard let scan = scanEntry(url) else { continue }
            guard scan.model == modelName,
                  scan.tokens.count >= options.minTokens,
                  scan.tokens.count < ids.count,
                  scan.tokens.count < contextSize,
                  ids.starts(with: scan.tokens) else { continue }
            lengths.append(scan.tokens.count)
        }
        return lengths.sorted(by: >)
    }

    /// Stream-restore `hit` into the decoder, ONE layer at a time: each layer's
    /// float slabs are parsed, written into the KV buffers, and released before
    /// the next layer is read — peak RAM is one layer, not the ~3× whole-file
    /// cost of the old load-everything path. Reads use F_NOCACHE (like store),
    /// so a multi-GB restore can't evict the hot page cache either. Bumps the
    /// hit counters on success; a corrupt entry is discarded like the C does.
    public func restore(_ hit: Hit, into decoder: StreamingDecoder) -> Bool {
        streamRestore(hit.url, into: decoder)
    }

    /// Restore the entry stored under the EXACT `tokens` (a content-keyed cache:
    /// the key is the file/project content prefix, not a conversation prefix).
    /// Same streaming path as `restore(_:into:)`. false if absent/corrupt/mismatched.
    public func restore(forTokens tokens: [Int], modelName: String,
                        into decoder: StreamingDecoder) -> Bool {
        let url = directory.appendingPathComponent(entryName(tokens: tokens, modelName: modelName))
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        return streamRestore(url, into: decoder)
    }

    private func streamRestore(_ url: URL, into decoder: StreamingDecoder) -> Bool {
        do {
            try streamSnapshot(url,
                onMeta: { nKeys, headDim, layerCount in
                    // Shape gate BEFORE any tensor body is read.
                    try decoder.beginImportKV(nKeys: nKeys, headDim: headDim, layerCount: layerCount)
                },
                onLayer: { i, layer in try decoder.importKVLayer(layer, at: i) })
            bumpHit(url)
            return true
        } catch is KVSnapshotError {
            return false            // valid entry, wrong shape for this decoder: keep it
        } catch {
            // Corrupt/truncated entry: discard it like the C does on load failure.
            try? FileManager.default.removeItem(at: url)
            return false
        }
    }

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

    // MARK: File scanning / parsing

    private func entryURLs() -> [URL] {
        let all = (try? FileManager.default.contentsOfDirectory(at: directory,
                                                                includingPropertiesForKeys: nil)) ?? []
        return all.filter { KVCFile.shaHexName($0.lastPathComponent) != nil }
    }

    private func entryName(tokens: [Int], modelName: String) -> String {
        var bytes = Array(modelName.utf8)
        for t in tokens {
            let v = UInt32(truncatingIfNeeded: t)
            bytes.append(contentsOf: [UInt8(v & 0xff), UInt8((v >> 8) & 0xff),
                                      UInt8((v >> 16) & 0xff), UInt8((v >> 24) & 0xff)])
        }
        return KVCFile.sha1Hex(bytes) + ".kv"
    }

    /// Cheap scan: header + model name + token list (no tensor body).
    private func scanEntry(_ url: URL) -> (model: String, tokens: [Int])? {
        guard let fh = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? fh.close() }
        func read(_ n: Int) -> [UInt8]? {
            guard n >= 0, let d = try? fh.read(upToCount: n), d.count == n else { return nil }
            return [UInt8](d)
        }
        guard let head = read(KVCFile.fixedHeader), KVCFile.parseHeader(head) != nil,
              let nameLenB = read(4) else { return nil }
        let nameLen = Int(KVCFile.leGet32(nameLenB, 0))
        guard nameLen < 4096, let nameB = read(nameLen),
              let countB = read(4) else { return nil }
        let count = Int(KVCFile.leGet32(countB, 0))
        guard count > 0, count < 1_000_000, let tokB = read(count * 4) else { return nil }
        var tokens = [Int](); tokens.reserveCapacity(count)
        for i in 0..<count { tokens.append(Int(KVCFile.leGet32(tokB, i * 4))) }
        return (String(decoding: nameB, as: UTF8.self), tokens)
    }

    private func readHeader(_ url: URL) -> KVCFile.Header? {
        guard let fh = try? FileHandle(forReadingFrom: url),
              let d = try? fh.read(upToCount: KVCFile.fixedHeader) else { return nil }
        try? fh.close()
        return KVCFile.parseHeader([UInt8](d))
    }

    /// Bump hits + lastUsed in the 48-byte header, in place.
    private func bumpHit(_ url: URL) {
        guard var h = readHeader(url) else { return }
        h.hits &+= 1
        h.lastUsed = UInt64(Date().timeIntervalSince1970)
        guard let fh = try? FileHandle(forWritingTo: url) else { return }
        try? fh.write(contentsOf: Data(KVCFile.fillHeader(h)))
        try? fh.close()
    }

    /// Parse error of the streaming reader (truncated file, bogus lengths).
    enum StreamError: Error { case corrupt }

    /// Stream-parse one entry: header/meta first (via `onMeta`, which may throw
    /// to abort BEFORE any tensor body is read — the shape gate), then ONE layer
    /// at a time to `onLayer`. Each layer's slabs are freed as soon as the
    /// callback returns; nothing accumulates. The fd reads with F_NOCACHE so the
    /// checkpoint's pages never displace the hot page cache (dense weights /
    /// expert bundle) — same rule as the store path. Throws StreamError.corrupt
    /// on a malformed file; rethrows whatever the callbacks throw.
    func streamSnapshot(_ url: URL,
                        onMeta: (_ nKeys: Int, _ headDim: Int, _ layerCount: Int) throws -> Void,
                        onLayer: (_ index: Int, _ layer: KVLayerSnapshot) throws -> Void) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw StreamError.corrupt }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw StreamError.corrupt }
        var remaining = Int(st.st_size)

        func readAll(_ p: UnsafeMutableRawPointer, _ n: Int) throws {
            guard n <= remaining else { throw StreamError.corrupt }
            var off = 0
            while off < n {
                let r = read(fd, p + off, n - off)
                guard r > 0 else { throw StreamError.corrupt }
                off += r
            }
            remaining -= n
        }
        func readBytes(_ n: Int) throws -> [UInt8] {
            guard n > 0 else { throw StreamError.corrupt }   // no zero-length fields in this format
            var b = [UInt8](repeating: 0, count: n)
            try b.withUnsafeMutableBytes { try readAll($0.baseAddress!, n) }
            return b
        }
        func u32() throws -> Int {
            Int(KVCFile.leGet32(try readBytes(4), 0))
        }
        func skip(_ n: Int) throws {
            guard n >= 0, n <= remaining, lseek(fd, off_t(n), SEEK_CUR) >= 0 else {
                throw StreamError.corrupt
            }
            remaining -= n
        }
        /// One tensor slab, read straight into its final allocation (no Data
        /// intermediate). The `remaining` guard rejects bogus lengths before
        /// they can turn into a giant allocation.
        func floats(_ n: Int) throws -> [Float] {
            guard n >= 0, n * 4 <= remaining else { throw StreamError.corrupt }
            return try [Float](unsafeUninitializedCapacity: n) { buf, count in
                if n > 0 { try readAll(UnsafeMutableRawPointer(buf.baseAddress!), n * 4) }
                count = n
            }
        }
        func readComp() throws -> CompSnapshot? {
            let has = try readBytes(1)[0]
            if has != 1 { return nil }
            let cCount = try u32(), stateLen = try u32()
            let kv = try floats(stateLen), score = try floats(stateLen)
            let cache = try floats(try u32())
            return CompSnapshot(count: cCount, stateKv: kv, stateScore: score, cacheRows: cache)
        }

        guard KVCFile.parseHeader(try readBytes(KVCFile.fixedHeader)) != nil else {
            throw StreamError.corrupt
        }
        try skip(try u32())                    // model name (matched by the caller's scan)
        let count = try u32()                  // nKeys
        guard count > 0, count < 1_000_000 else { throw StreamError.corrupt }
        try skip(count * 4)                    // token ids (the Hit already carries them)
        guard let ph = DSV4PayloadHeader(try readBytes(DSV4PayloadHeader.u32Fields * 4)) else {
            throw StreamError.corrupt
        }
        try onMeta(count, Int(ph.rawHeadKVDim), Int(ph.layerCount))
        for i in 0..<Int(ph.layerCount) {
            let rawStart = try u32()
            let raw = try floats(try u32())
            let comp = try readComp()
            let idx = try readComp()           // NSA indexer compressor (ratio-4 layers)
            try onLayer(i, KVLayerSnapshot(rawStart: rawStart, raw: raw, comp: comp, idx: idx))
        }
    }

    /// Full in-RAM parse of one entry (tests / tools): the streaming reader with
    /// an accumulator. The engine restore path never materializes this.
    func loadSnapshot(_ url: URL) -> KVSnapshot? {
        var nKeys = 0, headDim = 0
        var layers: [KVLayerSnapshot] = []
        do {
            try streamSnapshot(url,
                onMeta: { n, h, _ in nKeys = n; headDim = h },
                onLayer: { _, layer in layers.append(layer) })
        } catch { return nil }
        return KVSnapshot(nKeys: nKeys, headDim: headDim, layers: layers)
    }

    // MARK: little-endian append helpers

    private func appendU32(_ d: inout Data, _ v: UInt32) {
        d.append(contentsOf: [UInt8(v & 0xff), UInt8((v >> 8) & 0xff),
                              UInt8((v >> 16) & 0xff), UInt8((v >> 24) & 0xff)])
    }
}
