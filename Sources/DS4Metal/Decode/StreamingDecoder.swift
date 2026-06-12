import Foundation
import Metal
import DS4Core

// Stage D: per-layer SSD streaming decode. Instead of holding all N layers'
// weights resident (164GB), each layer is loaded on demand (copied from the GGUF
// mmap into GPUTensors), computed, then evicted. Because a layer's GPU buffers
// must stay alive until the GPU finishes reading them, each layer runs in its OWN
// command buffer (commit + wait, then evict) — the "split command buffer"
// streaming pattern from the C engine. Working set = one layer's weights
// (~few GB) instead of the whole model, so the real model runs on 16GB.
//
// Resident across layers (small): output-head weights, embedding table, per-layer
// KV caches, HC ping-pong buffers, scratch. layerProvider(i) supplies layer i's
// weights (real path: GGUFWeights.layer(rt, model, i)); the returned LayerWeights
// is dropped after the layer commits, freeing its Metal buffers (eviction).

/// Per-phase wall-clock accumulator for the decode forward pass. Each phase is
/// timed around a committed (and waited) command buffer / a CPU gather, so the
/// numbers reflect real elapsed time and answer "I/O vs compute". Times are
/// totals over all forward() calls; `report()` averages per token.
public struct DecodeProfile: Sendable {
    public var embedS = 0.0       // token embedding
    public var routeS = 0.0       // attention + router (compute)
    public var gatherS = 0.0      // gather the 6 selected experts from the mmap (EXPERT I/O)
    public var expertsS = 0.0     // shared FFN + routed MoE matvec (compute)
    public var layerOtherS = 0.0  // non-split decode path (resident experts)
    public var headS = 0.0        // output head
    public var forwards = 0       // number of forward() calls (= tokens)
    public var layers = 0         // total per-layer iterations
    public var expertHits = 0     // expert slot-cache hits (persistent experts)
    public var expertMisses = 0   // expert slot-cache misses (changed experts)

    public init() {}

    public func report() -> String {
        guard forwards > 0 else { return "Profilo decode: nessun forward registrato." }
        let f = Double(forwards)
        let total = embedS + routeS + gatherS + expertsS + layerOtherS + headS
        func ms(_ s: Double) -> String { String(format: "%6.1f", s / f * 1000) }
        func pct(_ s: Double) -> String { String(format: "%2.0f%%", total > 0 ? s / total * 100 : 0) }
        let tps = total > 0 ? f / total : 0
        var cacheLine = ""
        if expertHits + expertMisses > 0 {
            let rate = Double(expertHits) / Double(expertHits + expertMisses) * 100
            cacheLine = "\n  cache expert \(expertHits) hit / \(expertMisses) miss  (\(String(format: "%.0f", rate))% hit)"
        }
        return """
        Profilo decode — \(forwards) token, \(layers) iterazioni-layer
          embed        \(ms(embedS)) ms/token  (\(pct(embedS)))
          route/attn   \(ms(routeS)) ms/token  (\(pct(routeS)))   compute
          gather IO    \(ms(gatherS)) ms/token  (\(pct(gatherS)))   <- streaming esperti (SSD/page cache)
          experts      \(ms(expertsS)) ms/token  (\(pct(expertsS)))   compute
          layer (alt)  \(ms(layerOtherS)) ms/token  (\(pct(layerOtherS)))
          output head  \(ms(headS)) ms/token  (\(pct(headS)))\(cacheLine)
          ----------------------------------------
          totale       \(ms(total)) ms/token  (~\(String(format: "%.2f", tps)) tok/s)
        """
    }
}

public final class StreamingDecoder {
    let rt: MetalRuntime
    let d: DSV4Dims
    let rope: RopeParams
    let nLayers: Int
    let layerProvider: (Int) throws -> LayerWeights
    let embedTable: GPUTensor
    let out: OutputHeadWeights
    let rmsEps: Float, hcEps: Float

    /// Per-phase decode timing (opt-in: read after a run, reset between runs).
    public var profile = DecodeProfile()
    public func resetProfile() { profile = DecodeProfile() }

    /// Expert-cache hook: given (layer index, the 6 selected ids), gather and pack
    /// ONLY those experts' gate/up/down. When set, forward() splits each layer at
    /// the router and loads 6/256 experts on demand instead of the full set.
    let expertGather: ((Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor))?
    /// Optional LRU slot-cache ("persistent + changing experts"): when set, the
    /// DECODE path serves hits from resident GPU pools (zero copies) and gathers
    /// only the misses; the matvec runs on the pool with slot-index ids.
    public let slotCache: ExpertSlotCache?
    /// Routing-frequency statistics (the "usage imatrix"): fed by every route,
    /// persisted by the service, and used to pre-warm the slot cache.
    public let usage: ExpertUsageStats?

    let scratch: DecodeScratch
    let rawCaches: [GPUTensor]
    let compStates: [CompressorState?]   // NSA compressor state per compressed layer (nil on layers 0,1)
    /// NSA indexer compressor state (DSA): ratio-4 layers only. Beyond
    /// `d.indexerTopK` compressed rows, attention is restricted to the top-K
    /// most relevant for the current query (C: indexer_allowed_decode_one).
    let indexStates: [CompressorState?]
    /// Halves of s.mask dirtied by the last indexer selection (0 = clean).
    private var maskDirtyCount = 0
    /// Layers with real KV allocation (full model: 0..<nLayers; distributed slice: its range).
    let kvRange: Range<Int>
    /// KV capacity in tokens (raw rows per layer).
    let maxKeys: Int
    let hcA, hcB, embd: GPUTensor
    let flat, pre, owts, otmp, oembd, onormed, logits: GPUTensor
    let idsPacked: GPUTensor   // [0,1,...,k-1] for the packed-experts matvec

    public init(rt: MetalRuntime, dims: DSV4Dims, rope: RopeParams, nLayers: Int,
                layerProvider: @escaping (Int) throws -> LayerWeights,
                embedTable: GPUTensor, out: OutputHeadWeights, maxKeys: Int,
                rmsEps: Float = 1e-5, hcEps: Float = 1e-3,
                expertGather: ((Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor))? = nil,
                slotCache: ExpertSlotCache? = nil,
                usage: ExpertUsageStats? = nil,
                kvLayers: Range<Int>? = nil) throws {
        self.rt = rt; self.d = dims; self.rope = rope; self.nLayers = nLayers
        self.layerProvider = layerProvider; self.embedTable = embedTable; self.out = out
        self.rmsEps = rmsEps; self.hcEps = hcEps; self.expertGather = expertGather
        self.slotCache = slotCache
        self.usage = usage
        let hcDim = dims.nHC * dims.nEmbd
        // Distributed slice: allocate KV/compressor state ONLY for `kvLayers`
        // (a worker never runs the other layers — dummy 1-float buffers there).
        // nil = full model (the single-machine default).
        let kvRange = kvLayers ?? 0..<nLayers
        self.kvRange = kvRange
        self.maxKeys = maxKeys
        // NSA compressor state per compressed layer (ratio!=0); comp rows accumulate
        // ~1 per `ratio` tokens, so the attention KV scratch must hold maxKeys raw rows
        // + up to maxKeys/4 compressed rows (ratio-4 is the densest).
        let maxComp = maxKeys / 4 + 8
        compStates = try (0..<nLayers).map { il -> CompressorState? in
            guard kvRange.contains(il) else { return nil }
            let ratio = DSV4Shape.compressRatio(layer: il)
            guard ratio != 0 else { return nil }
            return try CompressorState(rt, ratio: ratio, headDim: dims.headDim, maxComp: maxKeys / ratio + 8)
        }
        // NSA indexer compressor (DSA): ratio-4 layers only (head_dim 128).
        indexStates = try (0..<nLayers).map { il -> CompressorState? in
            guard kvRange.contains(il), DSV4Shape.compressRatio(layer: il) == 4 else { return nil }
            return try CompressorState(rt, ratio: 4, headDim: dims.nIndexerHeadDim, maxComp: maxKeys / 4 + 8)
        }
        scratch = try DecodeScratch(rt, dims, maxKeys: maxKeys + maxComp)
        idsPacked = try GPUTensor.bytes(rt, Array(0..<Int32(dims.k)).withUnsafeBytes { Array($0) }, elementCount: dims.k)
        rawCaches = try (0..<nLayers).map { il in
            kvRange.contains(il) ? try GPUTensor.zeros(rt, floatCount: maxKeys * dims.headDim)
                                 : try GPUTensor.zeros(rt, floatCount: 1)
        }
        hcA = try .zeros(rt, floatCount: hcDim); hcB = try .zeros(rt, floatCount: hcDim)
        embd = try .zeros(rt, floatCount: dims.nEmbd)
        flat = try .zeros(rt, floatCount: hcDim); pre = try .zeros(rt, floatCount: dims.nHC)
        owts = try .zeros(rt, floatCount: dims.nHC); otmp = try .zeros(rt, floatCount: dims.nHC)
        oembd = try .zeros(rt, floatCount: dims.nEmbd); onormed = try .zeros(rt, floatCount: dims.nEmbd)
        logits = try .zeros(rt, floatCount: dims.vocab)
    }

    public func forward(token: Int, pos: Int, nKeys: Int) throws -> [Float] {
        // Fresh sequence: reset the recurrent compressor state (score=-inf, count=0).
        if pos == 0 { for c in compStates { try c?.reset(rt) }; for c in indexStates { try c?.reset(rt) } }
        try embedToken(token, into: hcA)
        var cur = hcA, other = hcB
        for i in 0..<nLayers {
            let w = try layerProvider(i)        // LOAD layer i (dense; experts on demand if cached)
            try runLayer(i, w: w, layerRope: DSV4Shape.ropeParams(layer: i),
                         cur: cur, other: other, pos: pos, nKeys: nKeys)
            swap(&cur, &other)
            // w (and any gathered experts) drop here -> Metal buffers freed (EVICT)
        }
        profile.forwards += 1
        return try outputHead(cur)
    }

    // MARK: - Distributed slice execution (pipeline parallelism)
    //
    // These let a node run only PART of the model: the coordinator owns the
    // embedding + output head, each worker owns a contiguous layer range and runs
    // it over an incoming HC state. The HC state (nHC*nEmbd floats) is what crosses
    // the wire between nodes. They reuse embedToken/runLayer/outputHead, so a slice
    // [start,end] is numerically identical to the same layers inside forward().

    /// HC state width that crosses the wire (nHC * nEmbd floats).
    public var hcStateCount: Int { d.nHC * d.nEmbd }

    /// Coordinator: embed `token` into the HC state (the start of the pipeline).
    public func embed(token: Int, pos: Int) throws -> [Float] {
        try embedToken(token, into: hcA)
        return readHC(hcA)
    }

    /// Worker: run layers `start...end` over an incoming HC state at absolute `pos`,
    /// returning the produced HC state to forward to the next slice. Resets only this
    /// slice's recurrent compressor state on a fresh sequence (pos == 0).
    public func forwardSlice(hc hcIn: [Float], pos: Int, nKeys: Int, start: Int, end: Int) throws -> [Float] {
        precondition(start >= 0 && end < nLayers && start <= end, "invalid layer slice \(start)...\(end)")
        if pos == 0 { for i in start...end { try compStates[i]?.reset(rt); try indexStates[i]?.reset(rt) } }
        writeFloats(hcIn, into: hcA)
        var cur = hcA, other = hcB
        for i in start...end {
            let w = try layerProvider(i)
            try runLayer(i, w: w, layerRope: DSV4Shape.ropeParams(layer: i),
                         cur: cur, other: other, pos: pos, nKeys: nKeys)
            swap(&cur, &other)
        }
        profile.forwards += 1
        return readHC(cur)
    }

    /// Worker, chunked prefill: run layers `start...end` over `hcs.count` consecutive
    /// tokens' HC states starting at absolute `posBase`. Token-outer (numerically
    /// identical to consecutive forwardSlice calls); amortizes the NETWORK round
    /// trip over the chunk — one WORK/RESULT per chunk instead of per token.
    public func forwardSliceBatch(hcs: [[Float]], posBase: Int, start: Int, end: Int) throws -> [[Float]] {
        var out: [[Float]] = []
        out.reserveCapacity(hcs.count)
        for (i, hc) in hcs.enumerated() {
            let pos = posBase + i
            out.append(try forwardSlice(hc: hc, pos: pos, nKeys: pos + 1, start: start, end: end))
        }
        return out
    }

    /// Coordinator/last node: run the output head over the final HC state → logits.
    public func head(hc hcIn: [Float]) throws -> [Float] {
        writeFloats(hcIn, into: hcA)
        return try outputHead(hcA)
    }

    /// Read the HC state (nHC*nEmbd floats) out of a GPU buffer.
    private func readHC(_ t: GPUTensor) -> [Float] {
        let n = d.nHC * d.nEmbd
        let p = t.buffer.contents().advanced(by: t.byteOffset).bindMemory(to: Float.self, capacity: n)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

    /// once** (per chunk) instead of once per token, so the dominant weight I/O is
    /// amortized over all the chunk's tokens. Numerically **identical** to calling
    /// `forward()` for tokens 0..N-1 in order — same ops, same per-token order,
    /// same KV-cache and NSA-compressor evolution — just reordered (layer outer,
    /// token inner) so the mmap'd weights stay hot across tokens. The prompt is
    /// split into chunks of `chunk` tokens to bound activation memory (≈ 2·chunk
    /// HC buffers); KV cache and the recurrent compressor carry across chunks.
    /// Populates the KV cache for positions startPos..startPos+N-1 and returns the
    /// LAST token's logits. With `startPos > 0` the call is **incremental**: it does
    /// NOT reset the recurrent compressor and continues the KV cache from the given
    /// position (the caller guarantees positions 0..startPos-1 are already valid) —
    /// this is what enables KV reuse across turns (prefill only the new suffix).
    public func prefill(tokens: [Int], startPos: Int = 0, chunk: Int = 512) throws -> [Float] {
        precondition(!tokens.isEmpty)
        if startPos == 0 { for c in compStates { try c?.reset(rt) }; for c in indexStates { try c?.reset(rt) } }   // fresh sequence
        var lastHC: GPUTensor?
        var start = 0
        let step = max(1, chunk)
        while start < tokens.count {
            let end = min(start + step, tokens.count)
            lastHC = try prefillRange(tokens, start: start, end: end, posBase: startPos)
            start = end
        }
        profile.forwards += tokens.count
        return try outputHead(lastHC!)
    }

    /// Process one prompt chunk [start, end) layer-major at absolute positions
    /// posBase+start … . Weights for each layer are loaded once and applied to all
    /// the chunk's tokens (in order). On the expert-gather path the routed-FFN
    /// phase is BATCHED: each unique expert is gathered once per group instead of
    /// 6 per token. Returns the chunk's last token's final HC state.
    private func prefillRange(_ tokens: [Int], start: Int, end: Int, posBase: Int) throws -> GPUTensor {
        let n = end - start
        let hcDim = d.nHC * d.nEmbd
        var cur: [GPUTensor] = try (0..<n).map { _ in try .zeros(rt, floatCount: hcDim) }
        var other: [GPUTensor] = try (0..<n).map { _ in try .zeros(rt, floatCount: hcDim) }
        for j in 0..<n { try embedToken(tokens[start + j], into: cur[j]) }
        for i in 0..<nLayers {
            try Task.checkCancellation()
            let w = try layerProvider(i)            // LOAD layer i ONCE for all chunk tokens
            let layerRope = DSV4Shape.ropeParams(layer: i)
            if let gather = expertGather, n > 1 {
                try batchedExpertLayer(i, w: w, layerRope: layerRope, cur: cur, other: other,
                                       n: n, posBase: posBase + start, gather: gather)
            } else {
                for j in 0..<n {
                    let pos = posBase + start + j     // attends KV[0..pos] (incl. earlier chunks/turns)
                    try runLayer(i, w: w, layerRope: layerRope, cur: cur[j], other: other[j],
                                 pos: pos, nKeys: pos + 1)
                }
            }
            swap(&cur, &other)                       // w drops here -> EVICT
        }
        return cur[n - 1]
    }

    /// Max experts gathered per group in the batched prefill (bounds the packed
    /// union tensors' transient memory: ~7 MB/expert on the 2-bit model). Env
    /// override: DS4_PREFILL_UNION. Never below d.k.
    private var maxUnionExperts: Int {
        let v = ProcessInfo.processInfo.environment["DS4_PREFILL_UNION"].flatMap(Int.init) ?? 64
        return max(d.k, v)
    }

    /// One prefill layer over all chunk tokens with BATCHED expert I/O.
    /// Phase A — routes run sequentially per token (attention is causal: token j
    /// attends KV written by tokens 0..j in this same layer), saving each token's
    /// FFN inputs (attn-normed cur, residual, HC split) and its expert selection.
    /// Phase B — tokens are grouped; each group's UNION of selected experts is
    /// gathered ONCE and every token's FFN runs over it with remapped ids.
    /// Numerically identical to the per-token path (a token's FFN does not feed
    /// other tokens within the layer); only the expert I/O is deduplicated:
    /// ≤ min(6·tokens, 256) expert reads per layer instead of 6·tokens.
    private func batchedExpertLayer(_ i: Int, w: LayerWeights, layerRope: RopeParams,
                                    cur: [GPUTensor], other: [GPUTensor], n: Int, posBase: Int,
                                    gather: (Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor)) throws {
        // Phase A: sequential routes; save per-token FFN inputs + selection.
        var curT: [GPUTensor] = [], attnT: [GPUTensor] = [], splitT: [GPUTensor] = []
        var idsT: [[Int32]] = [], rwT: [[Float]] = []
        curT.reserveCapacity(n); attnT.reserveCapacity(n); splitT.reserveCapacity(n)
        for j in 0..<n {
            try Task.checkCancellation()
            let pos = posBase + j
            var t = Date()
            try encodeRoute(i, w: w, layerRope: layerRope, curHc: cur[j], pos: pos, nKeys: pos + 1)
            profile.routeS += Date().timeIntervalSince(t)
            let (ids, rw) = readRouteSelection(layer: i)
            idsT.append(ids); rwT.append(rw)
            let cT = try GPUTensor.zeros(rt, floatCount: d.nEmbd)
            let aT = try GPUTensor.zeros(rt, floatCount: d.nHC * d.nEmbd)
            let sT = try GPUTensor.zeros(rt, floatCount: 24)
            copyFloats(from: scratch.cur, to: cT, count: d.nEmbd)
            copyFloats(from: scratch.afterAttn, to: aT, count: d.nHC * d.nEmbd)
            copyFloats(from: scratch.split, to: sT, count: 24)
            curT.append(cT); attnT.append(aT); splitT.append(sT)
            profile.layers += 1
        }

        // Phase B: group consecutive tokens while the union stays under the cap,
        // gather each group's union once, run every token's FFN with remapped ids.
        let cap = maxUnionExperts
        var j0 = 0
        while j0 < n {
            var union: [Int32] = []
            var seen = Set<Int32>()
            var j1 = j0
            while j1 < n {
                let fresh = idsT[j1].filter { !seen.contains($0) }
                if !union.isEmpty && union.count + fresh.count > cap { break }
                for id in fresh { union.append(id); seen.insert(id) }
                j1 += 1
            }
            var t = Date()
            let (g, u, dn) = try gather(i, union)        // each unique expert ONCE
            profile.gatherS += Date().timeIntervalSince(t)
            var posOf: [Int32: Int32] = [:]
            for (p, id) in union.enumerated() { posOf[id] = Int32(p) }
            for j in j0..<j1 {
                try Task.checkCancellation()
                let K = idsT[j].count
                let remapped = idsT[j].map { posOf[$0]! }
                let idsBuf = try GPUTensor.bytes(rt, remapped.withUnsafeBytes { Array($0) },
                                                 elementCount: K)
                writeFloats(rwT[j], into: scratch.rw)
                zeroDown6(from: K)
                t = Date()
                let c2 = GraphContext(rt); try c2.begin()
                try c2.decodeExperts(w: w, s: scratch, d: d, gateExp: g, upExp: u, downExp: dn,
                                     ids: idsBuf, outHc: other[j], activeK: K,
                                     cur: curT[j], afterAttn: attnT[j], split: splitT[j])
                c2.commit()
                profile.expertsS += Date().timeIntervalSince(t)
            }
            j0 = j1
            // g/u/dn drop here -> the group's packed union tensors are freed
        }
    }

    /// Embed one token into the HC state buffer `hc` (own command buffer).
    private func embedToken(_ token: Int, into hc: GPUTensor) throws {
        let t = Date()
        let ec = GraphContext(rt)
        try ec.begin()
        try ec.embedTokenHC(table: embedTable, token: token, embd: embd, hc: hc,
                            nEmbd: d.nEmbd, nVocab: d.vocab, nHC: d.nHC)
        ec.commit()
        profile.embedS += Date().timeIntervalSince(t)
    }

    /// Read back the router's selection after a committed decodeRoute, applying
    /// the activeExperts top-K reduction (route weights renormalized to the
    /// original total). Returns the final (ids, weights), both of count K ≤ d.k.
    /// Also feeds the usage statistics ("usage imatrix") for `layer`.
    private func readRouteSelection(layer: Int) -> (ids: [Int32], rw: [Float]) {
        let selPtr = scratch.selected.buffer.contents().bindMemory(to: Int32.self, capacity: d.k)
        var ids = Array(UnsafeBufferPointer(start: selPtr, count: d.k))
        let wptr = scratch.rw.buffer.contents().bindMemory(to: Float.self, capacity: d.k)
        var rw = Array(UnsafeBufferPointer(start: wptr, count: d.k))
        let K = max(1, min(d.activeExperts, d.k))
        if K < d.k {
            let keep = (0..<d.k).sorted { rw[$0] > rw[$1] }.prefix(K)
            let origSum = rw.reduce(0, +)
            let keptSum = keep.reduce(Float(0)) { $0 + rw[$1] }
            let scale = keptSum > 0 ? origSum / keptSum : 1
            ids = keep.map { ids[$0] }
            rw = keep.map { rw[$0] * scale }
        }
        usage?.record(layer: layer, ids: ids)
        return (ids, rw)
    }

    /// CPU-write `a` into the head of a shared GPU buffer (safe between commits).
    private func writeFloats(_ a: [Float], into t: GPUTensor) {
        a.withUnsafeBytes {
            memcpy(t.buffer.contents().advanced(by: t.byteOffset), $0.baseAddress!, $0.count)
        }
    }

    /// CPU-copy `count` floats between shared GPU buffers (after a commit).
    private func copyFloats(from src: GPUTensor, to dst: GPUTensor, count: Int) {
        memcpy(dst.buffer.contents().advanced(by: dst.byteOffset),
               src.buffer.contents().advanced(by: src.byteOffset), count * 4)
    }

    /// Zero s.down6 rows K..d.k-1 so the fixed moeSum6 adds zeros for unused slots.
    private func zeroDown6(from K: Int) {
        guard K < d.k else { return }
        let dptr = scratch.down6.buffer.contents().bindMemory(to: Float.self, capacity: d.k * d.nEmbd)
        for r in K..<d.k { for c in 0..<d.nEmbd { dptr[r * d.nEmbd + c] = 0 } }
    }

    /// One decode layer for one token: `cur` (HC in) -> `other` (HC out). Writes
    /// KV[i][pos], updates compStates[i]. Shared by `forward` (decode) and the
    /// layer-major `prefill` — identical numerics either way.
    private func runLayer(_ i: Int, w: LayerWeights, layerRope: RopeParams,
                          cur: GPUTensor, other: GPUTensor, pos: Int, nKeys: Int) throws {
        if let gather = expertGather {
            // Phase 1: route (own cb) -> read the selected ids (top-K reduced).
            var t = Date()
            try encodeRoute(i, w: w, layerRope: layerRope, curHc: cur, pos: pos, nKeys: nKeys)
            profile.routeS += Date().timeIntervalSince(t)
            let (ids, rw) = readRouteSelection(layer: i)
            let K = ids.count
            if K < d.k {
                writeFloats(rw, into: scratch.rw)
                zeroDown6(from: K)
            }
            if let cache = slotCache {
                // Persistent + changing experts: hits are already resident in the
                // layer's GPU pool (zero copies); only misses are filled from the
                // mmap. The matvec indexes the pool with slot ids.
                t = Date()
                let (pool, slots) = try cache.acquire(layer: i, ids: ids)
                profile.gatherS += Date().timeIntervalSince(t)
                profile.expertHits = cache.hits
                profile.expertMisses = cache.misses
                let slotsBuf = try GPUTensor.bytes(rt, slots.withUnsafeBytes { Array($0) },
                                                   elementCount: K)
                t = Date()
                let c2 = GraphContext(rt); try c2.begin()
                try c2.decodeExperts(w: w, s: scratch, d: d, gateExp: pool.gate,
                                     upExp: pool.up, downExp: pool.down,
                                     ids: slotsBuf, outHc: other, activeK: K)
                c2.commit()
                profile.expertsS += Date().timeIntervalSince(t)
            } else {
                // Gather ONLY the selected experts (EXPERT I/O from the mmap), then phase 2.
                t = Date()
                let (g, u, dn) = try gather(i, ids)
                profile.gatherS += Date().timeIntervalSince(t)
                t = Date()
                let c2 = GraphContext(rt); try c2.begin()
                try c2.decodeExperts(w: w, s: scratch, d: d, gateExp: g, upExp: u, downExp: dn,
                                     ids: idsPacked, outHc: other, activeK: K)
                c2.commit()
                profile.expertsS += Date().timeIntervalSince(t)
            }
        } else {
            let t = Date()
            try encodeRoute(i, w: w, layerRope: layerRope, curHc: cur, pos: pos, nKeys: nKeys)
            let lc = GraphContext(rt); try lc.begin()
            try lc.decodeExperts(w: w, s: scratch, d: d, gateExp: w.expGate, upExp: w.expUp,
                                 downExp: w.expDown, ids: scratch.selected, outHc: other)
            lc.commit()                      // COMPUTE (GPU finishes before w is dropped)
            profile.layerOtherS += Date().timeIntervalSince(t)
        }
        profile.layers += 1
    }

    /// Encode (and COMMIT) the route for one token on layer `i`. When the NSA
    /// indexer is active (ratio-4 layer with more compressed rows than the top-K),
    /// the command buffer is split at the indexer scores: commit phase 1a, run the
    /// CPU top-K to write the compressed-row mask, then encode the attention —
    /// the C "dense top-k mask" path (indexer_allowed_decode_one). Otherwise a
    /// single command buffer, numerically identical to the pre-indexer code.
    private func encodeRoute(_ i: Int, w: LayerWeights, layerRope: RopeParams,
                             curHc: GPUTensor, pos: Int, nKeys: Int) throws {
        let idx = indexStates[i]
        let hasIdxWeights = w.idxKv != nil && w.idxQB != nil && w.idxProj != nil
        let active = hasIdxWeights && indexerActive(i, pos: pos)
        if active, let idx {
            let c1 = GraphContext(rt); try c1.begin()
            let nComp = try c1.decodeRoutePre(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                              rawCache: rawCaches[i], pos: pos, rmsEps: rmsEps,
                                              comp: compStates[i], idx: hasIdxWeights ? idx : nil,
                                              indexerScoring: true)
            c1.commit()
            applyIndexerMask(nKeys: nKeys, nComp: nComp, nIdxComp: idx.count)
            let c2 = GraphContext(rt); try c2.begin()
            try c2.decodeRouteAttn(w: w, s: scratch, d: d, rope: layerRope, rawCache: rawCaches[i],
                                   nKeys: nKeys, pos: pos, rmsEps: rmsEps, hcEps: hcEps,
                                   nComp: nComp, comp: compStates[i])
            c2.commit()
        } else {
            clearMaskIfDirty()
            let c1 = GraphContext(rt); try c1.begin()
            let nComp = try c1.decodeRoutePre(curHc: curHc, w: w, s: scratch, d: d, rope: layerRope,
                                              rawCache: rawCaches[i], pos: pos, rmsEps: rmsEps,
                                              comp: compStates[i], idx: hasIdxWeights ? idx : nil,
                                              indexerScoring: false)
            try c1.decodeRouteAttn(w: w, s: scratch, d: d, rope: layerRope, rawCache: rawCaches[i],
                                   nKeys: nKeys, pos: pos, rmsEps: rmsEps, hcEps: hcEps,
                                   nComp: nComp, comp: compStates[i])
            c1.commit()
        }
    }

    /// Will the indexer restrict this token's compressed rows on layer `i`?
    /// (prospective count: the compressor may emit one more row for this token.)
    private func indexerActive(_ i: Int, pos: Int) -> Bool {
        guard let idx = indexStates[i] else { return false }
        let prospective = idx.count + (((pos + 1) % idx.ratio) == 0 ? 1 : 0)
        return prospective > d.indexerTopK
    }

    /// CPU top-K over the indexer scores (s.idxScores[0..nIdxComp)) → f16 mask:
    /// raw window rows stay 0; compressed row c gets 0 if selected, -inf if not.
    /// Ties keep the LOWEST row index (the C argmax scan picks the first best).
    private func applyIndexerMask(nKeys: Int, nComp: Int, nIdxComp: Int) {
        let nRaw = nKeys - max(0, nKeys - d.nSWA)
        let scores = scratch.idxScores.buffer.contents()
            .advanced(by: scratch.idxScores.byteOffset).bindMemory(to: Float.self, capacity: nIdxComp)
        var order = Array(0..<nIdxComp)
        order.sort { scores[$0] != scores[$1] ? scores[$0] > scores[$1] : $0 < $1 }
        var allowed = [Bool](repeating: false, count: nIdxComp)
        for k in 0..<min(d.indexerTopK, nIdxComp) { allowed[order[k]] = true }

        let total = nRaw + nComp
        let mask = scratch.mask.buffer.contents().bindMemory(to: UInt16.self, capacity: total)
        let negInf = Half.bits(-Float.infinity)
        for j in 0..<nRaw { mask[j] = 0 }
        for c in 0..<nComp {
            let ok = c < nIdxComp ? allowed[c] : true
            mask[nRaw + c] = ok ? 0 : negInf
        }
        maskDirtyCount = max(maskDirtyCount, total)
    }

    /// Zero the mask region a previous indexer selection dirtied (offsets shift
    /// every token, so a stale -inf would mask the wrong key).
    private func clearMaskIfDirty() {
        guard maskDirtyCount > 0 else { return }
        memset(scratch.mask.buffer.contents(), 0, maskDirtyCount * 2)
        maskDirtyCount = 0
    }

    /// Output head for one token's final HC state -> logits[vocab].
    private func outputHead(_ cur: GPUTensor) throws -> [Float] {
        let hcDim = d.nHC * d.nEmbd
        let t = Date()
        let oc = GraphContext(rt)
        try oc.begin()
        try oc.rmsNorm(cur, weight: nil, out: flat, rows: 1, n: hcDim, eps: rmsEps)
        try oc.matmulF16(weight: out.hcFn, x: flat, out: pre, inDim: hcDim, outDim: d.nHC)
        try oc.outputHCWeights(pre: pre, scaleScalar: out.hcScaleScalar, base: out.hcBase,
                               weights: owts, tmp: otmp, nHC: d.nHC, eps: hcEps)
        try oc.hcWeightedSum(x: cur, weights: owts, out: oembd, nEmbd: d.nEmbd, nHC: d.nHC, nTokens: 1)
        try oc.rmsNorm(oembd, weight: out.norm, out: onormed, rows: 1, n: d.nEmbd, eps: rmsEps)
        try oc.matmulQ8_0(weight: out.head, x: onormed, out: logits, inDim: d.nEmbd, outDim: d.vocab)
        oc.commit()
        profile.headS += Date().timeIntervalSince(t)
        return logits.floatArray(d.vocab)
    }

    /// Convenience: streaming generate (same loop as DSV4Decoder.generate).
    public func generate(prompt: [Int], maxNew: Int, sampling: DSV4Decoder.Sampling = .init(), eos: Int? = nil) throws -> [Int] {
        precondition(!prompt.isEmpty)
        var rng = sampling.seed
        var pos = 0
        var last: [Float] = []
        for tok in prompt { last = try forward(token: tok, pos: pos, nKeys: pos + 1); pos += 1 }
        var gen: [Int] = []
        for _ in 0..<maxNew {
            let next = Sampler.sample(last, temperature: sampling.temperature, topK: sampling.topK,
                                      topP: sampling.topP, minP: sampling.minP, rng: &rng)
            if let e = eos, next == e { break }
            gen.append(next)
            last = try forward(token: next, pos: pos, nKeys: pos + 1); pos += 1
        }
        return gen
    }

    /// Build a streaming decoder backed by a real GGUF model (the real Stage D
    /// path): each layer is loaded from the mmap on demand.
    public static func fromGGUF(rt: MetalRuntime, model: GGUFModel, dims: DSV4Dims, rope: RopeParams,
                                nLayers: Int, maxKeys: Int, rmsEps: Float = 1e-5, hcEps: Float = 1e-3) throws -> StreamingDecoder {
        let (embed, head) = try GGUFWeights.outputHead(rt, model)
        return try StreamingDecoder(rt: rt, dims: dims, rope: rope, nLayers: nLayers,
                                    layerProvider: { try GGUFWeights.layer(rt, model, $0) },
                                    embedTable: embed, out: head, maxKeys: maxKeys, rmsEps: rmsEps, hcEps: hcEps)
    }

    /// Expert-cache streaming decoder: per layer, only the dense weights are
    /// loaded up front; after routing, ONLY the 6 selected experts are gathered
    /// from the mmap (6/256 ~= 40x less expert IO/RAM). Numerically identical to
    /// the resident path (validated by ExpertCacheLayerTests).
    public static func fromGGUFExpertCached(rt: MetalRuntime, model: GGUFModel, dims: DSV4Dims, rope: RopeParams,
                                            nLayers: Int, maxKeys: Int, rmsEps: Float = 1e-5, hcEps: Float = 1e-3) throws -> StreamingDecoder {
        let (embed, head) = try GGUFWeights.outputHead(rt, model)
        let gather: (Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor) = { il, ids in
            let g = try GGUFWeights.gatherExperts(rt, model, "blk.\(il).ffn_gate_exps.weight", ids: ids, inDim: dims.nEmbd, outRows: dims.expertFfn)
            let u = try GGUFWeights.gatherExperts(rt, model, "blk.\(il).ffn_up_exps.weight", ids: ids, inDim: dims.nEmbd, outRows: dims.expertFfn)
            let dn = try GGUFWeights.gatherExperts(rt, model, "blk.\(il).ffn_down_exps.weight", ids: ids, inDim: dims.expertFfn, outRows: dims.nEmbd)
            return (g, u, dn)
        }
        // Memoize the non-routed (dense + NSA compressor) weights: loaded once,
        // resident across tokens (the C --ssd-streaming model). Only the 6 selected
        // experts are gathered per token (gatherExperts memcpy's just those rows from
        // the mmap = ~6/256 of expert IO). This is the fast path: per token ~= a few
        // expert slabs from SSD + GPU compute, instead of re-streaming the whole model.
        let cache = CachedLayerProvider { try GGUFWeights.layer(rt, model, $0, loadExperts: false) }
        return try StreamingDecoder(rt: rt, dims: dims, rope: rope, nLayers: nLayers,
                                    layerProvider: { try cache.get($0) },
                                    embedTable: embed, out: head, maxKeys: maxKeys, rmsEps: rmsEps, hcEps: hcEps,
                                    expertGather: gather)
    }

    /// Fastest 16GB path (the C `--ssd-streaming` model): non-routed weights are
    /// NO-COPY mmap views (resident via the OS page cache, single copy, evictable —
    /// no per-token re-copy, no 8GB of dirty buffers that OOM), and only the 6 selected
    /// experts are gathered per token. No memoization needed: the page cache serves
    /// repeated weight reads across tokens. Requires model opened metalMapping:true.
    public static func fromGGUFExpertCachedMapped(rt: MetalRuntime, model: GGUFModel, dims: DSV4Dims, rope: RopeParams,
                                                  nLayers: Int, maxKeys: Int, rmsEps: Float = 1e-5, hcEps: Float = 1e-3,
                                                  cacheSlots: Int? = nil, kvLayers: Range<Int>? = nil) throws -> StreamingDecoder {
        let (embed, head) = try GGUFWeights.outputHeadMapped(rt, model)
        let gather: (Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor) = { il, ids in
            let g = try GGUFWeights.gatherExperts(rt, model, "blk.\(il).ffn_gate_exps.weight", ids: ids, inDim: dims.nEmbd, outRows: dims.expertFfn)
            let u = try GGUFWeights.gatherExperts(rt, model, "blk.\(il).ffn_up_exps.weight", ids: ids, inDim: dims.nEmbd, outRows: dims.expertFfn)
            let dn = try GGUFWeights.gatherExperts(rt, model, "blk.\(il).ffn_down_exps.weight", ids: ids, inDim: dims.expertFfn, outRows: dims.nEmbd)
            return (g, u, dn)
        }
        // Routing-frequency stats ("usage imatrix"): always collected (cheap);
        // the service persists them across sessions and they pre-warm the cache.
        let usage = ExpertUsageStats(nLayers: nLayers)
        // Persistent + changing experts (cacheSlots param, else env
        // DS4_EXPERT_CACHE_SLOTS; default off): per layer, an N-slot LRU pool
        // keeps hot experts resident in GPU buffers; only misses are memcpy'd
        // from the mmap. The pool is WIRED memory (~6.9 MB/slot on the 2-bit
        // model × nLayers): on tight-RAM machines start small (8) and watch the
        // hit rate in the decode profile / Tuning tab.
        let envSlots = ProcessInfo.processInfo.environment["DS4_EXPERT_CACHE_SLOTS"].flatMap(Int.init)
        let nSlots = cacheSlots ?? envSlots ?? 0
        var cache: ExpertSlotCache? = nil
        if nSlots > 0 {
            let S = max(8, nSlots)
            let gateBytes = (dims.nEmbd / 256) * dims.gateQuant.blockBytes * dims.expertFfn
            let upBytes = (dims.nEmbd / 256) * dims.upQuant.blockBytes * dims.expertFfn
            let downBytes = (dims.expertFfn / 256) * dims.downQuant.blockBytes * dims.nEmbd
            cache = ExpertSlotCache(slotsPerLayer: S, makePool: {
                (gate: try GPUTensor.zerosBytes(rt, byteLength: S * gateBytes),
                 up: try GPUTensor.zerosBytes(rt, byteLength: S * upBytes),
                 down: try GPUTensor.zerosBytes(rt, byteLength: S * downBytes))
            }, fill: { il, id, pool, slot in
                try GGUFWeights.copyExpert(model, "blk.\(il).ffn_gate_exps.weight", id: id,
                                           expertBytes: gateBytes, into: pool.gate, slot: slot)
                try GGUFWeights.copyExpert(model, "blk.\(il).ffn_up_exps.weight", id: id,
                                           expertBytes: upBytes, into: pool.up, slot: slot)
                try GGUFWeights.copyExpert(model, "blk.\(il).ffn_down_exps.weight", id: id,
                                           expertBytes: downBytes, into: pool.down, slot: slot)
            }, warm: { il in usage.top(layer: il, n: S) })
        }
        return try StreamingDecoder(rt: rt, dims: dims, rope: rope, nLayers: nLayers,
                                    layerProvider: { try GGUFWeights.layerMappedDense(rt, model, $0) },
                                    embedTable: embed, out: head, maxKeys: maxKeys, rmsEps: rmsEps, hcEps: hcEps,
                                    expertGather: gather, slotCache: cache, usage: usage, kvLayers: kvLayers)
    }

    /// Mapped-experts streaming decoder: per layer the dense weights are copied,
    /// but the routed experts are NO-COPY mmap views over the FULL expert tensors
    /// (all 256). The single-cb decode path runs mul_mv_id with the real selected
    /// ids; the OS page cache caches touched experts across tokens — no per-token
    /// re-gather. Requires model opened with metalMapping:true.
    public static func fromGGUFMappedExperts(rt: MetalRuntime, model: GGUFModel, dims: DSV4Dims, rope: RopeParams,
                                             nLayers: Int, maxKeys: Int, rmsEps: Float = 1e-5, hcEps: Float = 1e-3) throws -> StreamingDecoder {
        let (embed, head) = try GGUFWeights.outputHead(rt, model)
        // Memoize per-layer weights: dense (incl. NSA compressor) are COPIED resident
        // and reused across tokens; experts are no-copy mmap. Without this the ~8GB of
        // non-routed weights were re-copied from the mmap EVERY token (minutes/token on
        // 16GB). This is the C `--ssd-streaming` model: non-routed resident, experts paged.
        let cache = CachedLayerProvider { try GGUFWeights.layerMappedExperts(rt, model, $0) }
        return try StreamingDecoder(rt: rt, dims: dims, rope: rope, nLayers: nLayers,
                                    layerProvider: { try cache.get($0) },
                                    embedTable: embed, out: head, maxKeys: maxKeys, rmsEps: rmsEps, hcEps: hcEps,
                                    expertGather: nil)   // single-cb decodeLayer with real ids
    }
}

/// Loads each layer's weights once and reuses them across tokens (weights are
/// read-only during decode). Keeps non-routed weights resident instead of
/// re-streaming them from the mmap every token.
final class CachedLayerProvider {
    private let make: (Int) throws -> LayerWeights
    private var cache: [Int: LayerWeights] = [:]
    init(_ make: @escaping (Int) throws -> LayerWeights) { self.make = make }
    func get(_ il: Int) throws -> LayerWeights {
        if let w = cache[il] { return w }
        let w = try make(il); cache[il] = w; return w
    }
}
