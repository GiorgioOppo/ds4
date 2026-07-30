import Foundation

// Faithful Swift port of the ds4.c sampler: argmax, the xorshift* RNG, and the
// temperature / top-k / top-p / min-p sampler (ds4_sample_logits ->
// sample_top_p_min_p -> sample_full_vocab). Float math uses libm expf so results
// match the C bit-for-bit on the same platform. Phase 6 of the C->Swift port.
//
// Validation: cross-checked against the public C ds4_sample_logits over many
// (temperature, top_k, top_p, min_p, seed) configurations with identical logits.
//
// DS4_FAST_SAMPLER (default on) short-circuits the full-vocabulary candidate
// sort when min_p > 0 by collecting only the reachable candidates — same
// selection and same RNG stream; `=0` restores the historical full build for
// parity diagnostics.

public enum Sampler {
    static let negInf: Float = -1.0e30

    /// DS4_FAST_SAMPLER (default on; `=0` disables): threshold-collected fast
    /// path for the full-vocabulary sampler (`topK <= 0`, `topP < 1`,
    /// `minP > 0` — the server/subagent default config). See `fullVocab`.
    /// Cached once: ProcessInfo.environment materializes a fresh dictionary
    /// per call — far too costly per token.
    static let fastFullVocab: Bool =
        ProcessInfo.processInfo.environment["DS4_FAST_SAMPLER"] != "0"

    private static func argmaxValues<C: RandomAccessCollection>(
        _ logits: C
    ) -> Int where C.Element == Float, C.Index == Int {
        var best = 0
        var bestV = negInf
        for i in 0..<logits.count where logits[i] > bestV {
            bestV = logits[i]; best = i
        }
        return best
    }

    /// Port of sample_argmax.
    public static func argmax(_ logits: [Float]) -> Int {
        argmaxValues(logits)
    }

    /// Zero-copy form for logits held in a shared Metal buffer.
    public static func argmax(
        _ logits: UnsafeBufferPointer<Float>
    ) -> Int {
        argmaxValues(logits)
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

    /// Port of sample_full_vocab (top_k <= 0 path). With `minP > 0` and
    /// DS4_FAST_SAMPLER (default) the candidate build collects only the tokens
    /// that the min-p walk can actually reach, skipping the full-vocabulary
    /// sort — same selection, same RNG consumption (see the fast-path comment).
    /// `fast` is injectable (internal) so the parity test can compare both
    /// builds without touching the process environment.
    private static func fullVocabValues<C: RandomAccessCollection>(
        _ logits: C, _ nVocab: Int,
        _ temperature: Float, _ topP: Float, _ minP: Float,
        _ rng: inout UInt64, fast: Bool
    ) -> Int where C.Element == Float, C.Index == Int {
        var maxLogit = negInf
        var best = 0
        var finite = 0
        for i in 0..<nVocab where logits[i].isFinite {
            finite += 1
            if logits[i] > maxLogit { maxLogit = logits[i]; best = i }
        }
        if finite == 0 { return argmaxValues(logits) }

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
        var sum: Float = 0.0
        if fast && minP > 0.0 {
            // FAST PATH (DS4_FAST_SAMPLER): the sorted walk below never RETAINS
            // a candidate whose p = prob/sum falls under minProb = (1/sum)*minP
            // — probs are monotone in the sort order, so the first such
            // candidate breaks the loop and everything after it is unreachable.
            // Collecting only candidates with prob >= minP/4 therefore yields
            // the SAME filtered prefix: the 4x margin is orders of magnitude
            // beyond the few-ulp wobble of the exact per-candidate comparison
            // (both sides divided by the same sum), and the max — prob =
            // expf(0) = 1 — always survives min(pre, 1.0). `sum`, every
            // retained expression and the RNG draw are evaluated verbatim in
            // the same order, so the selection can only differ from the
            // historical path in the ordering of EXACT logit ties (which the
            // full sort itself leaves unspecified — see the sort note below).
            // Net effect: the descending sort runs on the few candidates above
            // threshold instead of on all ~129k (which cost ms per token on
            // the topK=0 server/subagent path).
            cand.reserveCapacity(1024)
            let pre = min(minP * 0.25, 1.0)
            for i in 0..<nVocab {
                let v = logits[i]
                if !v.isFinite { continue }
                let p = expf((v - maxLogit) / temperature)
                sum += p
                if p >= pre { cand.append(Candidate(id: i, logit: v, prob: p)) }
            }
        } else {
            cand.reserveCapacity(finite)
            for i in 0..<nVocab {
                let v = logits[i]
                if !v.isFinite { continue }
                let p = expf((v - maxLogit) / temperature)
                cand.append(Candidate(id: i, logit: v, prob: p))
                sum += p
            }
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

    static func fullVocab(
        _ logits: [Float], _ nVocab: Int,
        _ temperature: Float, _ topP: Float, _ minP: Float,
        _ rng: inout UInt64, fast: Bool = Sampler.fastFullVocab
    ) -> Int {
        fullVocabValues(
            logits, nVocab, temperature, topP, minP, &rng, fast: fast)
    }

    /// Apply a repetition penalty to a copy of the logits for the recently produced
    /// tokens (llama.cpp `penalty_repeat`): logit /= penalty if positive, else
    /// logit *= penalty. Returns the logits unchanged when disabled.
    static func applyRepetitionPenalty(_ logits: [Float], recent: ArraySlice<Int>, penalty: Float) -> [Float] {
        guard penalty > 1.0, !recent.isEmpty else { return logits }
        var out = logits
        let n = out.count
        for id in Set(recent) where id >= 0 && id < n {
            out[id] = out[id] > 0 ? out[id] / penalty : out[id] * penalty
        }
        return out
    }

    /// Port of sample_top_p_min_p / ds4_sample_logits, plus an optional repetition
    /// penalty over the recently produced tokens (llama.cpp style): without it a
    /// quantized model that emits one low-confidence token can lock into a repeat
    /// loop ("è è è ( ( (") and never recover. `recent` is the tail of the token
    /// stream to penalize; `repetitionPenalty` > 1 divides those tokens' logits.
    private static func sampleValues<C: RandomAccessCollection>(
        _ logits: C, temperature: Float,
        topK: Int, topP: Float, minP: Float,
        repetitionPenalty: Float,
        recent: ArraySlice<Int>,
        rng: inout UInt64
    ) -> Int where C.Element == Float, C.Index == Int {
        if repetitionPenalty > 1.0, !recent.isEmpty {
            let penalized = applyRepetitionPenalty(
                Array(logits), recent: recent, penalty: repetitionPenalty)
            return sampleValues(
                penalized, temperature: temperature,
                topK: topK, topP: topP, minP: minP,
                repetitionPenalty: 1, recent: [], rng: &rng)
        }
        let nVocab = logits.count
        if temperature <= 0.0 { return argmaxValues(logits) }
        var topP = topP
        var minP = minP
        var topK = topK
        if topP <= 0.0 || topP > 1.0 { topP = 1.0 }
        if minP < 0.0 { minP = 0.0 }
        if topK <= 0 {
            return fullVocabValues(
                logits, nVocab, temperature, topP, minP, &rng,
                fast: fastFullVocab)
        }
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
        if n == 0 { return argmaxValues(logits) }

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

    public static func sample(
        _ logits: [Float], temperature: Float,
        topK: Int, topP: Float, minP: Float,
        repetitionPenalty: Float = 1.0,
        recent: ArraySlice<Int> = ArraySlice<Int>(),
        rng: inout UInt64
    ) -> Int {
        sampleValues(
            logits, temperature: temperature,
            topK: topK, topP: topP, minP: minP,
            repetitionPenalty: repetitionPenalty, recent: recent, rng: &rng)
    }

    /// Zero-copy sampling for logits exposed directly from shared GPU memory.
    /// A repetition penalty still materializes a private copy because only
    /// the penalized token positions may be mutated.
    public static func sample(
        _ logits: UnsafeBufferPointer<Float>, temperature: Float,
        topK: Int, topP: Float, minP: Float,
        repetitionPenalty: Float = 1.0,
        recent: ArraySlice<Int> = ArraySlice<Int>(),
        rng: inout UInt64
    ) -> Int {
        sampleValues(
            logits, temperature: temperature,
            topK: topK, topP: topP, minP: minP,
            repetitionPenalty: repetitionPenalty, recent: recent, rng: &rng)
    }
}
