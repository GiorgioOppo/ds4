import Foundation

// Faithful Swift port of the ds4.c sampler: argmax, the xorshift* RNG, and the
// temperature / top-k / top-p / min-p sampler (ds4_sample_logits ->
// sample_top_p_min_p -> sample_full_vocab). Float math uses libm expf so results
// match the C bit-for-bit on the same platform. Phase 6 of the C->Swift port.
//
// Validation: cross-checked against the public C ds4_sample_logits over many
// (temperature, top_k, top_p, min_p, seed) configurations with identical logits.

public enum Sampler {
    static let negInf: Float = -1.0e30

    /// Port of sample_argmax.
    public static func argmax(_ logits: [Float]) -> Int {
        var best = 0
        var bestV = negInf
        for i in 0..<logits.count where logits[i] > bestV {
            bestV = logits[i]; best = i
        }
        return best
    }

    /// Port of sample_rng_next (xorshift64* with the C's nonzero reseed).
    public static func rngNext(_ state: inout UInt64) -> UInt64 {
        var x = state
        if x == 0 { x = 0x9e37_79b9_7f4a_7c15 }
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        state = x
        return x &* 0x2545_f491_4f6c_dd1d
    }

    /// Port of sample_rng_f32: 24-bit uniform in [0, 1).
    public static func rngF32(_ state: inout UInt64) -> Float {
        let x = rngNext(&state)
        return Float((x >> 40) & 0xff_ffff) / 16_777_216.0
    }

    private struct Candidate { var id: Int; var logit: Float; var prob: Float }

    /// Port of sample_full_vocab (top_k <= 0 path).
    private static func fullVocab(_ logits: [Float], _ nVocab: Int,
                                  _ temperature: Float, _ topP: Float, _ minP: Float,
                                  _ rng: inout UInt64) -> Int {
        var maxLogit = negInf
        var best = 0
        var finite = 0
        for i in 0..<nVocab where logits[i].isFinite {
            finite += 1
            if logits[i] > maxLogit { maxLogit = logits[i]; best = i }
        }
        if finite == 0 { return argmax(logits) }

        if topP >= 1.0 {
            let minRel: Float = minP > 0.0 ? minP : 0.0
            var sum: Float = 0.0
            for i in 0..<nVocab {
                let v = logits[i]
                if !v.isFinite { continue }
                let p = expf((v - maxLogit) / temperature)
                if p < minRel { continue }
                sum += p
            }
            if sum <= 0.0 || !sum.isFinite { return best }
            var r = rngF32(&rng) * sum
            for i in 0..<nVocab {
                let v = logits[i]
                if !v.isFinite { continue }
                let p = expf((v - maxLogit) / temperature)
                if p < minRel { continue }
                r -= p
                if r <= 0.0 { return i }
            }
            return best
        }

        var cand: [Candidate] = []
        cand.reserveCapacity(finite)
        var sum: Float = 0.0
        for i in 0..<nVocab {
            let v = logits[i]
            if !v.isFinite { continue }
            let p = expf((v - maxLogit) / temperature)
            cand.append(Candidate(id: i, logit: v, prob: p))
            sum += p
        }
        if sum <= 0.0 || !sum.isFinite { return best }

        // Descending by logit. With distinct logits this matches qsort exactly.
        cand.sort { $0.logit > $1.logit }
        let minProb = (cand[0].prob / sum) * (minP > 0.0 ? minP : 0.0)
        var filteredSum: Float = 0.0
        var filtered = 0
        for i in 0..<cand.count {
            let p = cand[i].prob / sum
            if i > 0 && p < minProb { break }
            filteredSum += cand[i].prob
            filtered += 1
            if filteredSum / sum >= topP { break }
        }
        if filtered == 0 { return best }

        var r = rngF32(&rng) * filteredSum
        for i in 0..<filtered {
            r -= cand[i].prob
            if r <= 0.0 { return cand[i].id }
        }
        return cand[filtered - 1].id
    }

    /// Port of sample_top_p_min_p / ds4_sample_logits.
    public static func sample(_ logits: [Float], temperature: Float,
                              topK: Int, topP: Float, minP: Float,
                              rng: inout UInt64) -> Int {
        let nVocab = logits.count
        if temperature <= 0.0 { return argmax(logits) }
        var topP = topP
        var minP = minP
        var topK = topK
        if topP <= 0.0 || topP > 1.0 { topP = 1.0 }
        if minP < 0.0 { minP = 0.0 }
        if topK <= 0 { return fullVocab(logits, nVocab, temperature, topP, minP, &rng) }
        if topK > 1024 { topK = 1024 }
        if topK > nVocab { topK = nVocab }

        // Partial insertion sort keeping the top_k largest finite logits.
        var ids = [Int](repeating: 0, count: topK)
        var vals = [Float](repeating: 0, count: topK)
        var n = 0
        for i in 0..<nVocab {
            let v = logits[i]
            if !v.isFinite { continue }
            if n == topK && v <= vals[n - 1] { continue }
            var j: Int
            if n < topK { j = n; n += 1 } else { j = n - 1 }
            while j > 0 && vals[j - 1] < v {
                vals[j] = vals[j - 1]; ids[j] = ids[j - 1]; j -= 1
            }
            vals[j] = v; ids[j] = i
        }
        if n == 0 { return argmax(logits) }

        var probs = [Float](repeating: 0, count: n)
        let maxLogit = vals[0]
        var sum: Float = 0.0
        for i in 0..<n {
            probs[i] = expf((vals[i] - maxLogit) / temperature)
            sum += probs[i]
        }
        if sum <= 0.0 || !sum.isFinite { return ids[0] }

        let minProb = (probs[0] / sum) * minP
        var filteredSum: Float = 0.0
        var filtered = 0
        for i in 0..<n {
            let p = probs[i] / sum
            if i > 0 && p < minProb { break }
            filteredSum += probs[i]
            filtered += 1
            if filteredSum / sum >= topP { break }
        }
        if filtered <= 0 { return ids[0] }

        var r = rngF32(&rng) * filteredSum
        for i in 0..<filtered {
            r -= probs[i]
            if r <= 0.0 { return ids[i] }
        }
        return ids[filtered - 1]
    }
}
