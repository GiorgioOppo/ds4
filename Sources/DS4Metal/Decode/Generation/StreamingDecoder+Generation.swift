import Foundation
import Metal
import DS4Core

extension StreamingDecoder {
    /// Output head for one token's final HC state -> logits[vocab].
    func outputHead(_ cur: GPUTensor) throws -> [Float] {
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
}
