import Foundation
import Metal
import DS4Core

/// Sampling controls shared by the production streaming decoder and the
/// internal resident reference used by synthetic tests. This public value type
/// keeps the production API independent from the test-only `DSV4Decoder`.
public struct DSV4Sampling: Sendable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var minP: Float
    public var seed: UInt64

    public init(temperature: Float = 0.0, topK: Int = 0, topP: Float = 1,
                minP: Float = 0, seed: UInt64 = 0x1234) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.minP = minP
        self.seed = seed
    }
}

extension StreamingDecoder {
    /// Run the output head for one token's final HC state, leaving the
    /// `vocab` logits in the decoder's shared `logits` tensor. Keeping the
    /// compute/readback split here lets teacher-forced evaluation inspect only
    /// a few top ids without allocating and copying a 129k-element Swift array
    /// for every scored position.
    private func computeOutputHead(_ cur: GPUTensor) throws {
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
    }

    /// Output head for one token's final HC state -> logits[vocab].
    func outputHead(_ cur: GPUTensor) throws -> [Float] {
        try computeOutputHead(cur)
        return logits.floatArray(d.vocab)
    }

    /// Highest-scoring token ids for one final HC state, read directly from the
    /// shared Metal buffer. Candidates are unique and ordered by descending
    /// logit; equal logits retain the lower token id because the vocabulary is
    /// scanned in ascending order and insertion uses a strict comparison.
    func outputHeadTopK(_ cur: GPUTensor, k: Int) throws -> [Int] {
        precondition(k > 0, "outputHeadTopK: k deve essere positivo")
        try computeOutputHead(cur)
        let values = (logits.buffer.contents() + logits.byteOffset)
            .bindMemory(to: Float.self, capacity: d.vocab)
        let limit = min(k, d.vocab)
        var bestIds: [Int] = []
        var bestValues: [Float] = []
        bestIds.reserveCapacity(limit)
        bestValues.reserveCapacity(limit)

        for tokenId in 0..<d.vocab {
            let value = values[tokenId]
            var rank = 0
            while rank < bestValues.count {
                let rankedValue = bestValues[rank]
                let rankedId = bestIds[rank]
                let precedes: Bool
                if value.isNaN {
                    precedes = false
                } else if rankedValue.isNaN {
                    precedes = true
                } else {
                    precedes = value > rankedValue
                        || (value == rankedValue && tokenId < rankedId)
                }
                if precedes { break }
                rank += 1
            }
            guard rank < limit else { continue }
            bestIds.insert(tokenId, at: rank)
            bestValues.insert(value, at: rank)
            if bestIds.count > limit {
                bestIds.removeLast()
                bestValues.removeLast()
            }
        }
        return bestIds
    }

    /// Greedy token id compatibility wrapper around the generic top-k scan.
    func outputHeadTop1(_ cur: GPUTensor) throws -> Int {
        try outputHeadTopK(cur, k: 1)[0]
    }

    /// Convenience generation loop over the streaming decoder.
    public func generate(prompt: [Int], maxNew: Int,
                         sampling: DSV4Sampling = .init(), eos: Int? = nil) throws -> [Int] {
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
