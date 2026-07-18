import Foundation

/// Sampling for the GLM chat path, mirroring the DeepSeek service's knobs:
/// temperature, top-K and repetition penalty over a recent-token window.
/// Pure and injectable-RNG so every branch is unit-tested without a device.
///
/// Conventions (llama.cpp-compatible, like the DeepSeek sampler):
/// - `temperature <= 0` or `topK == 1` collapses to greedy argmax;
/// - repetition penalty divides positive logits and multiplies negative ones
///   for every token inside the recent window (`penaltyWindow`, default 64);
/// - top-K keeps the K highest logits, then softmax at `temperature` and one
///   categorical draw.
public enum GLM52Sampler {
    public static let penaltyWindow = 64

    public static func sample<R: RandomNumberGenerator>(
        logits: [Float],
        temperature: Float,
        topK: Int,
        repetitionPenalty: Float,
        recentTokens: ArraySlice<Int32>,
        using generator: inout R) -> Int32? {
        guard !logits.isEmpty else { return nil }
        var adjusted = logits
        if repetitionPenalty > 0, repetitionPenalty != 1 {
            for token in recentTokens.suffix(penaltyWindow) {
                let index = Int(token)
                guard index >= 0, index < adjusted.count else { continue }
                adjusted[index] = adjusted[index] > 0
                    ? adjusted[index] / repetitionPenalty
                    : adjusted[index] * repetitionPenalty
            }
        }
        guard temperature > 0, topK != 1 else {
            return GLM52GreedyDecoding.argmax(adjusted)
        }

        // Top-K candidate set (K <= 0 means the full vocabulary).
        let limit = topK > 0 ? min(topK, adjusted.count) : adjusted.count
        var candidates = Array(adjusted.enumerated())
        candidates.sort { lhs, rhs in
            lhs.element != rhs.element
                ? lhs.element > rhs.element
                : lhs.offset < rhs.offset
        }
        candidates.removeSubrange(limit...)

        // Stable softmax at temperature, then one categorical draw.
        let peak = candidates[0].element
        var weights = [Double](repeating: 0, count: candidates.count)
        var total = 0.0
        for (rank, candidate) in candidates.enumerated() {
            let weight = exp(Double((candidate.element - peak) / temperature))
            weights[rank] = weight
            total += weight
        }
        guard total > 0, total.isFinite else {
            return Int32(candidates[0].offset)
        }
        var threshold = Double.random(in: 0..<1, using: &generator) * total
        for (rank, candidate) in candidates.enumerated() {
            threshold -= weights[rank]
            if threshold <= 0 { return Int32(candidate.offset) }
        }
        return Int32(candidates[candidates.count - 1].offset)
    }

    /// System-RNG convenience used by the chat service.
    public static func sample(logits: [Float],
                              temperature: Float,
                              topK: Int,
                              repetitionPenalty: Float,
                              recentTokens: ArraySlice<Int32>) -> Int32? {
        var generator = SystemRandomNumberGenerator()
        return sample(logits: logits, temperature: temperature, topK: topK,
                      repetitionPenalty: repetitionPenalty,
                      recentTokens: recentTokens, using: &generator)
    }
}
