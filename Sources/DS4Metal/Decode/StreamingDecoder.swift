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

    public init() {}

    public func report() -> String {
        guard forwards > 0 else { return "Profilo decode: nessun forward registrato." }
        let f = Double(forwards)
        let total = embedS + routeS + gatherS + expertsS + layerOtherS + headS
        func ms(_ s: Double) -> String { String(format: "%6.1f", s / f * 1000) }
        func pct(_ s: Double) -> String { String(format: "%2.0f%%", total > 0 ? s / total * 100 : 0) }
        let tps = total > 0 ? f / total : 0
        return """
        Profilo decode — \(forwards) token, \(layers) iterazioni-layer
          embed        \(ms(embedS)) ms/token  (\(pct(embedS)))
          route/attn   \(ms(routeS)) ms/token  (\(pct(routeS)))   compute
          gather IO    \(ms(gatherS)) ms/token  (\(pct(gatherS)))   <- streaming esperti (SSD/page cache)
          experts      \(ms(expertsS)) ms/token  (\(pct(expertsS)))   compute
          layer (alt)  \(ms(layerOtherS)) ms/token  (\(pct(layerOtherS)))
          output head  \(ms(headS)) ms/token  (\(pct(headS)))
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

    let scratch: DecodeScratch
    let rawCaches: [GPUTensor]
    let compStates: [CompressorState?]   // NSA compressor state per compressed layer (nil on layers 0,1)
    let hcA, hcB, embd: GPUTensor
    let flat, pre, owts, otmp, oembd, onormed, logits: GPUTensor
    let idsPacked: GPUTensor   // [0,1,...,k-1] for the packed-experts matvec

    public init(rt: MetalRuntime, dims: DSV4Dims, rope: RopeParams, nLayers: Int,
                layerProvider: @escaping (Int) throws -> LayerWeights,
                embedTable: GPUTensor, out: OutputHeadWeights, maxKeys: Int,
                rmsEps: Float = 1e-5, hcEps: Float = 1e-3,
                expertGather: ((Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor))? = nil) throws {
        self.rt = rt; self.d = dims; self.rope = rope; self.nLayers = nLayers
        self.layerProvider = layerProvider; self.embedTable = embedTable; self.out = out
        self.rmsEps = rmsEps; self.hcEps = hcEps; self.expertGather = expertGather
        let hcDim = dims.nHC * dims.nEmbd
        // NSA compressor state per compressed layer (ratio!=0); comp rows accumulate
        // ~1 per `ratio` tokens, so the attention KV scratch must hold maxKeys raw rows
        // + up to maxKeys/4 compressed rows (ratio-4 is the densest).
        let maxComp = maxKeys / 4 + 8
        compStates = try (0..<nLayers).map { il -> CompressorState? in
            let ratio = DSV4Shape.compressRatio(layer: il)
            guard ratio != 0 else { return nil }
            return try CompressorState(rt, ratio: ratio, headDim: dims.headDim, maxComp: maxKeys / ratio + 8)
        }
        scratch = try DecodeScratch(rt, dims, maxKeys: maxKeys + maxComp)
        idsPacked = try GPUTensor.bytes(rt, Array(0..<Int32(dims.k)).withUnsafeBytes { Array($0) }, elementCount: dims.k)
        rawCaches = try (0..<nLayers).map { _ in try GPUTensor.zeros(rt, floatCount: maxKeys * dims.headDim) }
        hcA = try .zeros(rt, floatCount: hcDim); hcB = try .zeros(rt, floatCount: hcDim)
        embd = try .zeros(rt, floatCount: dims.nEmbd)
        flat = try .zeros(rt, floatCount: hcDim); pre = try .zeros(rt, floatCount: dims.nHC)
        owts = try .zeros(rt, floatCount: dims.nHC); otmp = try .zeros(rt, floatCount: dims.nHC)
        oembd = try .zeros(rt, floatCount: dims.nEmbd); onormed = try .zeros(rt, floatCount: dims.nEmbd)
        logits = try .zeros(rt, floatCount: dims.vocab)
    }

    public func forward(token: Int, pos: Int, nKeys: Int) throws -> [Float] {
        let hcDim = d.nHC * d.nEmbd
        // Fresh sequence: reset the recurrent compressor state (score=-inf, count=0).
        if pos == 0 { for c in compStates { try c?.reset(rt) } }
        // embedding (own command buffer)
        var t = Date()
        let ec = GraphContext(rt)
        try ec.begin()
        try ec.embedTokenHC(table: embedTable, token: token, embd: embd, hc: hcA,
                            nEmbd: d.nEmbd, nVocab: d.vocab, nHC: d.nHC)
        ec.commit()
        profile.embedS += Date().timeIntervalSince(t)

        var cur = hcA, other = hcB
        for i in 0..<nLayers {
            let w = try layerProvider(i)        // LOAD layer i (dense weights; experts on demand if cached)
            let layerRope = DSV4Shape.ropeParams(layer: i)
            if let gather = expertGather {
                // Phase 1: route (own cb) -> read the 6 selected ids.
                t = Date()
                let c1 = GraphContext(rt); try c1.begin()
                try c1.decodeRoute(curHc: cur, w: w, s: scratch, d: d, rope: layerRope, rawCache: rawCaches[i],
                                   nKeys: nKeys, pos: pos, rmsEps: rmsEps, hcEps: hcEps, comp: compStates[i])
                c1.commit()
                profile.routeS += Date().timeIntervalSince(t)
                // The router picked the top-d.k of 256. If activeExperts<d.k, keep the
                // top-K of those by route weight (renormalized so they still sum to the
                // original total) and gather ONLY those -> less expert I/O. Top-K of 256.
                let K = max(1, min(d.activeExperts, d.k))
                let selPtr = scratch.selected.buffer.contents().bindMemory(to: Int32.self, capacity: d.k)
                var ids = Array(UnsafeBufferPointer(start: selPtr, count: d.k))
                if K < d.k {
                    let wptr = scratch.rw.buffer.contents().bindMemory(to: Float.self, capacity: d.k)
                    let rw = Array(UnsafeBufferPointer(start: wptr, count: d.k))
                    let keep = (0..<d.k).sorted { rw[$0] > rw[$1] }.prefix(K)
                    let origSum = rw.reduce(0, +)
                    let keptSum = keep.reduce(Float(0)) { $0 + rw[$1] }
                    let scale = keptSum > 0 ? origSum / keptSum : 1
                    var kept: [Int32] = []
                    for (j, idx) in keep.enumerated() { wptr[j] = rw[idx] * scale; kept.append(ids[idx]) }
                    ids = kept
                    // Zero down6 rows K..d.k-1 so the fixed sum6 adds zeros for unused slots.
                    let dptr = scratch.down6.buffer.contents().bindMemory(to: Float.self, capacity: d.k * d.nEmbd)
                    for r in K..<d.k { for c in 0..<d.nEmbd { dptr[r * d.nEmbd + c] = 0 } }
                }
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
            } else {
                t = Date()
                let lc = GraphContext(rt); try lc.begin()
                try lc.decodeLayer(curHc: cur, w: w, s: scratch, d: d, rope: layerRope, rawCache: rawCaches[i],
                                   nKeys: nKeys, pos: pos, outHc: other, rmsEps: rmsEps, hcEps: hcEps, comp: compStates[i])
                lc.commit()                      // COMPUTE (GPU finishes before w is dropped)
                profile.layerOtherS += Date().timeIntervalSince(t)
            }
            profile.layers += 1
            swap(&cur, &other)
            // w (and any gathered experts) drop here -> Metal buffers freed (EVICT)
        }

        // output head (own command buffer)
        t = Date()
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
        profile.forwards += 1
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
                                                  nLayers: Int, maxKeys: Int, rmsEps: Float = 1e-5, hcEps: Float = 1e-3) throws -> StreamingDecoder {
        let (embed, head) = try GGUFWeights.outputHeadMapped(rt, model)
        let gather: (Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor) = { il, ids in
            let g = try GGUFWeights.gatherExperts(rt, model, "blk.\(il).ffn_gate_exps.weight", ids: ids, inDim: dims.nEmbd, outRows: dims.expertFfn)
            let u = try GGUFWeights.gatherExperts(rt, model, "blk.\(il).ffn_up_exps.weight", ids: ids, inDim: dims.nEmbd, outRows: dims.expertFfn)
            let dn = try GGUFWeights.gatherExperts(rt, model, "blk.\(il).ffn_down_exps.weight", ids: ids, inDim: dims.expertFfn, outRows: dims.nEmbd)
            return (g, u, dn)
        }
        return try StreamingDecoder(rt: rt, dims: dims, rope: rope, nLayers: nLayers,
                                    layerProvider: { try GGUFWeights.layerMappedDense(rt, model, $0) },
                                    embedTable: embed, out: head, maxKeys: maxKeys, rmsEps: rmsEps, hcEps: hcEps,
                                    expertGather: gather)
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
