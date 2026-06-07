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

public final class StreamingDecoder {
    let rt: MetalRuntime
    let d: DSV4Dims
    let rope: RopeParams
    let nLayers: Int
    let layerProvider: (Int) throws -> LayerWeights
    let embedTable: GPUTensor
    let out: OutputHeadWeights
    let rmsEps: Float, hcEps: Float

    /// Expert-cache hook: given (layer index, the 6 selected ids), gather and pack
    /// ONLY those experts' gate/up/down. When set, forward() splits each layer at
    /// the router and loads 6/256 experts on demand instead of the full set.
    let expertGather: ((Int, [Int32]) throws -> (GPUTensor, GPUTensor, GPUTensor))?

    let scratch: DecodeScratch
    let rawCaches: [GPUTensor]
    /// Per-layer persistent KV-compression + indexer state (nil on ratio==0 layers).
    /// Carries the recurrent compressor state and emitted compressed caches across
    /// tokens. See docs/COMPRESSED-ATTENTION-PORT.md.
    let compStates: [CompressedLayerState?]
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
        scratch = try DecodeScratch(rt, dims, maxKeys: maxKeys)
        idsPacked = try GPUTensor.bytes(rt, Array(0..<Int32(dims.k)).withUnsafeBytes { Array($0) }, elementCount: dims.k)
        rawCaches = try (0..<nLayers).map { _ in try GPUTensor.zeros(rt, floatCount: maxKeys * dims.headDim) }
        compStates = try CompressedLayerState.perLayer(rt, nLayers: nLayers, ctxSize: maxKeys)
        hcA = try .zeros(rt, floatCount: hcDim); hcB = try .zeros(rt, floatCount: hcDim)
        embd = try .zeros(rt, floatCount: dims.nEmbd)
        flat = try .zeros(rt, floatCount: hcDim); pre = try .zeros(rt, floatCount: dims.nHC)
        owts = try .zeros(rt, floatCount: dims.nHC); otmp = try .zeros(rt, floatCount: dims.nHC)
        oembd = try .zeros(rt, floatCount: dims.nEmbd); onormed = try .zeros(rt, floatCount: dims.nEmbd)
        logits = try .zeros(rt, floatCount: dims.vocab)
    }

    public func forward(token: Int, pos: Int, nKeys: Int) throws -> [Float] {
        let hcDim = d.nHC * d.nEmbd
        // embedding (own command buffer)
        let ec = GraphContext(rt)
        try ec.begin()
        try ec.embedTokenHC(table: embedTable, token: token, embd: embd, hc: hcA,
                            nEmbd: d.nEmbd, nVocab: d.vocab, nHC: d.nHC)
        ec.commit()

        var cur = hcA, other = hcB
        for i in 0..<nLayers {
            let w = try layerProvider(i)        // LOAD layer i (dense weights; experts on demand if cached)
            let layerRope = DSV4Shape.ropeParams(layer: i)
            if let gather = expertGather {
                // Phase 1: route (own cb) -> read the 6 selected ids.
                let c1 = GraphContext(rt); try c1.begin()
                try c1.decodeRoute(curHc: cur, w: w, s: scratch, d: d, rope: layerRope, rawCache: rawCaches[i],
                                   nKeys: nKeys, pos: pos, rmsEps: rmsEps, hcEps: hcEps, compState: compStates[i])
                c1.commit()
                let selPtr = scratch.selected.buffer.contents().bindMemory(to: Int32.self, capacity: d.k)
                let ids = Array(UnsafeBufferPointer(start: selPtr, count: d.k))
                // Gather ONLY the 6 selected experts, then phase 2 (own cb).
                let (g, u, dn) = try gather(i, ids)
                let c2 = GraphContext(rt); try c2.begin()
                try c2.decodeExperts(w: w, s: scratch, d: d, gateExp: g, upExp: u, downExp: dn,
                                     ids: idsPacked, outHc: other)
                c2.commit()
            } else {
                let lc = GraphContext(rt); try lc.begin()
                try lc.decodeLayer(curHc: cur, w: w, s: scratch, d: d, rope: layerRope, rawCache: rawCaches[i],
                                   nKeys: nKeys, pos: pos, outHc: other, rmsEps: rmsEps, hcEps: hcEps,
                                   compState: compStates[i])
                lc.commit()                      // COMPUTE (GPU finishes before w is dropped)
            }
            swap(&cur, &other)
            // w (and any gathered experts) drop here -> Metal buffers freed (EVICT)
        }

        // output head (own command buffer)
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
        return try StreamingDecoder(rt: rt, dims: dims, rope: rope, nLayers: nLayers,
                                    layerProvider: { try GGUFWeights.layer(rt, model, $0, loadExperts: false) },
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
        return try StreamingDecoder(rt: rt, dims: dims, rope: rope, nLayers: nLayers,
                                    layerProvider: { try GGUFWeights.layerMappedExperts(rt, model, $0) },
                                    embedTable: embed, out: head, maxKeys: maxKeys, rmsEps: rmsEps, hcEps: hcEps,
                                    expertGather: nil)   // single-cb decodeLayer with real ids
    }
}
