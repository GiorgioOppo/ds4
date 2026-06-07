import Foundation
import Metal
import DS4Core

// Stage C: the full decode forward pass — embed token -> N decode layers (HC
// ping-pong, per-layer KV cache) -> output head (HC collapse -> norm -> vocab
// matmul) -> logits. Faithful composition of metal_graph_eval_token + the layer
// loop + metal_graph_encode_output_head. Structural; numerical fidelity gated on
// the real model (>=64GB). Requires nKeys % 32 == 0 (no-pad flash path).

public struct OutputHeadWeights {
    public var hcFn: GPUTensor          // F16 [nHC x hcDim]
    public var hcScaleScalar: Float     // single float from model
    public var hcBase: GPUTensor        // F32 [nHC]
    public var norm: GPUTensor          // F32 [nEmbd]
    public var head: GPUTensor          // Q8 [vocab x nEmbd]
    public init(hcFn: GPUTensor, hcScaleScalar: Float, hcBase: GPUTensor, norm: GPUTensor, head: GPUTensor) {
        self.hcFn = hcFn; self.hcScaleScalar = hcScaleScalar; self.hcBase = hcBase; self.norm = norm; self.head = head
    }
}

public final class DSV4Decoder {
    let rt: MetalRuntime
    let d: DSV4Dims
    let rope: RopeParams
    let layers: [LayerWeights]
    let embedTable: GPUTensor      // F16 bytes [vocab x nEmbd]
    let out: OutputHeadWeights
    let rmsEps: Float, hcEps: Float

    // resident state
    let scratch: DecodeScratch
    let rawCaches: [GPUTensor]     // one per layer: maxKeys x headDim
    let hcA, hcB, embd: GPUTensor
    // output-head scratch
    let flat, pre, owts, otmp, oembd, onormed, logits: GPUTensor

    public init(rt: MetalRuntime, dims: DSV4Dims, rope: RopeParams, layers: [LayerWeights],
                embedTable: GPUTensor, out: OutputHeadWeights, maxKeys: Int,
                rmsEps: Float = 1e-5, hcEps: Float = 1e-3) throws {
        self.rt = rt; self.d = dims; self.rope = rope; self.layers = layers
        self.embedTable = embedTable; self.out = out; self.rmsEps = rmsEps; self.hcEps = hcEps
        let hcDim = dims.nHC * dims.nEmbd
        scratch = try DecodeScratch(rt, dims, maxKeys: maxKeys)
        rawCaches = try (0..<layers.count).map { _ in try GPUTensor.zeros(rt, floatCount: maxKeys * dims.headDim) }
        hcA = try .zeros(rt, floatCount: hcDim); hcB = try .zeros(rt, floatCount: hcDim)
        embd = try .zeros(rt, floatCount: dims.nEmbd)
        flat = try .zeros(rt, floatCount: hcDim); pre = try .zeros(rt, floatCount: dims.nHC)
        owts = try .zeros(rt, floatCount: dims.nHC); otmp = try .zeros(rt, floatCount: dims.nHC)
        oembd = try .zeros(rt, floatCount: dims.nEmbd); onormed = try .zeros(rt, floatCount: dims.nEmbd)
        logits = try .zeros(rt, floatCount: dims.vocab)
    }

    /// Forward one token at position `pos`; `nKeys` = number of valid KV rows
    /// (pos+1), must be a multiple of 32 for the no-pad flash path.
    public func forward(token: Int, pos: Int, nKeys: Int) throws -> [Float] {
        let hcDim = d.nHC * d.nEmbd
        let ctx = GraphContext(rt)
        try ctx.begin()
        try ctx.embedTokenHC(table: embedTable, token: token, embd: embd, hc: hcA,
                             nEmbd: d.nEmbd, nVocab: d.vocab, nHC: d.nHC)
        var cur = hcA, other = hcB
        for i in 0..<layers.count {
            try ctx.decodeLayer(curHc: cur, w: layers[i], s: scratch, d: d, rope: DSV4Shape.ropeParams(layer: i),
                                rawCache: rawCaches[i], nKeys: nKeys, pos: pos, outHc: other,
                                rmsEps: rmsEps, hcEps: hcEps)
            swap(&cur, &other)
        }
        // output head
        try ctx.rmsNorm(cur, weight: nil, out: flat, rows: 1, n: hcDim, eps: rmsEps)
        try ctx.matmulF16(weight: out.hcFn, x: flat, out: pre, inDim: hcDim, outDim: d.nHC)
        try ctx.outputHCWeights(pre: pre, scaleScalar: out.hcScaleScalar, base: out.hcBase,
                                weights: owts, tmp: otmp, nHC: d.nHC, eps: hcEps)
        try ctx.hcWeightedSum(x: cur, weights: owts, out: oembd, nEmbd: d.nEmbd, nHC: d.nHC, nTokens: 1)
        try ctx.rmsNorm(oembd, weight: out.norm, out: onormed, rows: 1, n: d.nEmbd, eps: rmsEps)
        try ctx.matmulQ8_0(weight: out.head, x: onormed, out: logits, inDim: d.nEmbd, outDim: d.vocab)
        ctx.commit()
        return logits.floatArray(d.vocab)
    }

    public struct Sampling {
        public var temperature: Float, topK: Int, topP: Float, minP: Float, seed: UInt64
        public init(temperature: Float = 0.0, topK: Int = 0, topP: Float = 1, minP: Float = 0, seed: UInt64 = 0x1234) {
            self.temperature = temperature; self.topK = topK; self.topP = topP; self.minP = minP; self.seed = seed
        }
    }

    /// Generate up to `maxNew` tokens after `prompt`. Decode-style prefill (each
    /// prompt token populates the KV cache one position at a time), then
    /// autoregressive decode. Stops early on `eos`. Returns the generated tokens.
    public func generate(prompt: [Int], maxNew: Int, sampling: Sampling = Sampling(), eos: Int? = nil) throws -> [Int] {
        precondition(!prompt.isEmpty)
        var rng = sampling.seed
        var pos = 0
        var lastLogits: [Float] = []
        for tok in prompt {
            lastLogits = try forward(token: tok, pos: pos, nKeys: pos + 1)
            pos += 1
        }
        var generated: [Int] = []
        for _ in 0..<maxNew {
            let next = Sampler.sample(lastLogits, temperature: sampling.temperature, topK: sampling.topK,
                                      topP: sampling.topP, minP: sampling.minP, rng: &rng)
            if let e = eos, next == e { break }
            generated.append(next)
            lastLogits = try forward(token: next, pos: pos, nKeys: pos + 1)
            pos += 1
        }
        return generated
    }
}
