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
    private let quantBits: UInt8
    private let contextSize: Int

    public init(directory: URL, budgetMB: Int, quantBits: UInt8, contextSize: Int,
                options: Options = Options()) throws {
        self.directory = directory
        self.budgetBytes = UInt64(max(64, budgetMB)) * 1_048_576
        self.quantBits = quantBits == 2 ? 2 : 4    // header validity wants {2,4}
        self.contextSize = contextSize
        self.options = options
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    // MARK: Lookup

    public struct Hit {
        public let tokens: [Int]
        public let snapshot: KVSnapshot
    }

    /// Find the stored entry with the LONGEST token prefix of `ids` (strictly
    /// shorter than `ids`, ≥ minTokens, same model, fits this context), load its
    /// snapshot and bump its hit counters.
    public func findLongestPrefix(of ids: [Int], modelName: String) -> Hit? {
        var bestURL: URL?
        var bestTokens: [Int] = []
        for url in entryURLs() {
            guard let scan = scanEntry(url) else { continue }
            guard scan.model == modelName,
                  scan.tokens.count >= options.minTokens,
                  scan.tokens.count < ids.count,
                  scan.tokens.count < contextSize,
                  scan.tokens.count > bestTokens.count,
                  ids.starts(with: scan.tokens) else { continue }
            bestURL = url
            bestTokens = scan.tokens
        }
        guard let url = bestURL else { return nil }
        guard let snapshot = loadSnapshot(url) else {
            // Corrupt/truncated entry: discard it like the C does on load failure.
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        bumpHit(url)
        return Hit(tokens: bestTokens, snapshot: snapshot)
    }

    // MARK: Store

    /// Checkpoint `tokens`+`snapshot` (dedup by content name). Mirrors the C
    /// store_live_prefix order: refuse an entry that alone exceeds the budget,
    /// EVICT FIRST to make room for the incoming bytes (with the supersede-
    /// continued scoring), then write atomically. `reason` "cold" marks the
    /// first checkpoint of a conversation (anchor, 2× protected in eviction).
    @discardableResult
    public func store(tokens: [Int], modelName: String, snapshot: KVSnapshot,
                      reason: KVCFile.Reason = .continued) -> Bool {
        guard tokens.count >= options.minTokens, snapshot.nKeys == tokens.count else { return false }
        let url = directory.appendingPathComponent(entryName(tokens: tokens, modelName: modelName))
        guard !FileManager.default.fileExists(atPath: url.path) else { return false }

        var body = Data()
        appendU32(&body, UInt32(Data(modelName.utf8).count)); body.append(Data(modelName.utf8))
        appendU32(&body, UInt32(tokens.count))
        for t in tokens { appendU32(&body, UInt32(truncatingIfNeeded: t)) }
        let ph = DSV4PayloadHeader(
            savedContextSize: UInt32(contextSize), prefillChunk: 512,
            rawKVCapacity: UInt32(contextSize), rawSlidingWindow: 128,
            compressedKVCapacity: 0, checkpointTokenCount: UInt32(tokens.count),
            layerCount: UInt32(snapshot.layers.count), rawHeadKVDim: UInt32(snapshot.headDim),
            indexerHeadDim: 128, vocabSize: 0,
            liveRawRows: UInt32(snapshot.layers.first.map { snapshot.nKeys - $0.rawStart } ?? 0))
        body.append(contentsOf: ph.serialize())
        for layer in snapshot.layers {
            appendU32(&body, UInt32(layer.rawStart))
            appendU32(&body, UInt32(layer.raw.count)); appendFloats(&body, layer.raw)
            appendComp(&body, layer.comp)
            appendComp(&body, layer.idx)     // NSA indexer compressor (ratio-4 layers)
        }

        let now = UInt64(Date().timeIntervalSince1970)
        let header = KVCFile.fillHeader(KVCFile.Header(
            quantBits: quantBits, reason: reason.rawValue, extFlags: 0, modelId: 0,
            tokens: UInt32(tokens.count), hits: 0, ctxSize: UInt32(contextSize),
            createdAt: now, lastUsed: now, payloadBytes: UInt64(body.count)))
        var file = Data(header); file.append(body)
        guard UInt64(file.count) <= budgetBytes else { return false }   // can never fit
        evictToBudget(incomingBytes: UInt64(file.count),
                      incomingTokens: tokens, incomingModel: modelName)
        do { try file.write(to: url, options: .atomic) } catch { return false }
        return true
    }

    /// Evict lowest-score entries until the directory fits `budget − incomingBytes`.
    /// Score = ported `KVCFile.evictionScore` ×(0.05 + 0.45·h/(h+1)) when the entry
    /// is a CONTINUED strict token-prefix of the incoming checkpoint (the C
    /// supersede-continued rule: the longer checkpoint of the same conversation
    /// replaces the shorter one under pressure). Ties evict the older lastUsed.
    func evictToBudget(incomingBytes: UInt64, incomingTokens: [Int], incomingModel: String) {
        struct Victim { let url: URL; let size: UInt64; let lastUsed: UInt64; let score: Double }
        var victims: [Victim] = []
        var total: UInt64 = 0
        let now = UInt64(Date().timeIntervalSince1970)
        for url in entryURLs() {
            guard let h = readHeader(url) else { continue }
            let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
            let sz = (attrs?[.size] as? NSNumber)?.uint64Value ?? 0
            total += sz
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
            victims.append(Victim(url: url, size: sz, lastUsed: h.lastUsed, score: score))
        }
        guard incomingBytes <= budgetBytes else { return }
        let target = budgetBytes - incomingBytes
        guard total > target else { return }
        let order = victims.sorted {
            $0.score != $1.score ? $0.score < $1.score : $0.lastUsed < $1.lastUsed
        }
        for v in order where total > target {
            try? FileManager.default.removeItem(at: v.url)
            total -= min(total, v.size)
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

    /// Full parse of one entry's tensor body into a KVSnapshot.
    func loadSnapshot(_ url: URL) -> KVSnapshot? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        let b = [UInt8](data)
        var o = KVCFile.fixedHeader
        func u32() -> Int? {
            guard o + 4 <= b.count else { return nil }
            defer { o += 4 }
            return Int(KVCFile.leGet32(b, o))
        }
        func floats(_ n: Int) -> [Float]? {
            guard n >= 0, o + n * 4 <= b.count else { return nil }
            var out = [Float](repeating: 0, count: n)
            b.withUnsafeBytes { raw in
                _ = out.withUnsafeMutableBytes { dst in
                    memcpy(dst.baseAddress!, raw.baseAddress!.advanced(by: o), n * 4)
                }
            }
            o += n * 4
            return out
        }
        guard let nameLen = u32() else { return nil }
        o += nameLen
        guard let count = u32() else { return nil }
        o += count * 4
        guard o + DSV4PayloadHeader.u32Fields * 4 <= b.count,
              let ph = DSV4PayloadHeader(Array(b[o..<(o + DSV4PayloadHeader.u32Fields * 4)])) else { return nil }
        o += DSV4PayloadHeader.u32Fields * 4
        func readComp() -> CompSnapshot?? {       // nil = parse error; .some(nil) = absent
            guard o < b.count else { return nil }
            let has = b[o]; o += 1
            if has != 1 { return .some(nil) }
            guard let cCount = u32(), let stateLen = u32(),
                  let kv = floats(stateLen), let score = floats(stateLen),
                  let cacheLen = u32(), let cache = floats(cacheLen) else { return nil }
            return CompSnapshot(count: cCount, stateKv: kv, stateScore: score, cacheRows: cache)
        }
        var layers: [KVLayerSnapshot] = []
        for _ in 0..<Int(ph.layerCount) {
            guard let rawStart = u32(), let rawCount = u32(), let raw = floats(rawCount),
                  let comp = readComp(), let idx = readComp() else { return nil }
            layers.append(KVLayerSnapshot(rawStart: rawStart, raw: raw, comp: comp, idx: idx))
        }
        return KVSnapshot(nKeys: count, headDim: Int(ph.rawHeadKVDim), layers: layers)
    }

    // MARK: little-endian append helpers

    private func appendComp(_ d: inout Data, _ c: CompSnapshot?) {
        guard let c else { d.append(0); return }
        d.append(1)
        appendU32(&d, UInt32(c.count))
        appendU32(&d, UInt32(c.stateKv.count))
        appendFloats(&d, c.stateKv); appendFloats(&d, c.stateScore)
        appendU32(&d, UInt32(c.cacheRows.count)); appendFloats(&d, c.cacheRows)
    }

    private func appendU32(_ d: inout Data, _ v: UInt32) {
        d.append(contentsOf: [UInt8(v & 0xff), UInt8((v >> 8) & 0xff),
                              UInt8((v >> 16) & 0xff), UInt8((v >> 24) & 0xff)])
    }
    private func appendFloats(_ d: inout Data, _ a: [Float]) {
        a.withUnsafeBufferPointer { d.append(Data(buffer: $0)) }   // little-endian host (arm64)
    }
}
