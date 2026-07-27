import Foundation
import DS4Core

/// GLM chat sampling — a thin, faithful wrapper over the shared DS4Core
/// `Sampler` (the bit-for-bit port of the C `sample_top_p_min_p`). Using the
/// same code path as DeepSeek closes the previous GLM divergence: upstream runs
/// GLM through the SAME sampler (defaults temp 1.0, top-p 0.95, min-p 0), so GLM
/// must honor top-p AND min-p, draw from a seedable C-compatible xorshift RNG,
/// and use `expf`-float softmax — not the old top-K-only, `Double`-softmax,
/// system-RNG path.
///
/// Conventions (llama.cpp / C-compatible):
/// - `temperature <= 0` collapses to argmax;
/// - `topK <= 0` samples the full vocabulary (top-p/min-p only);
/// - repetition penalty (>1) divides positive logits / multiplies negative ones
///   for every token in the recent window (`penaltyWindow`, default 64).
public enum GLM52Sampler {
    public static let penaltyWindow = 64

    /// Sample one token. `rng` is the caller-owned xorshift state (seed it from
    /// `SamplingParams.seed` for reproducibility, exactly like the DeepSeek path).
    public static func sample(
        logits: [Float],
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repetitionPenalty: Float,
        recentTokens: ArraySlice<Int32>,
        rng: inout UInt64) -> Int32? {
        guard !logits.isEmpty else { return nil }
        let recent = recentTokens.suffix(penaltyWindow).map(Int.init)
        let id = Sampler.sample(logits, temperature: temperature,
                                topK: topK, topP: topP, minP: minP,
                                repetitionPenalty: repetitionPenalty,
                                recent: recent[...], rng: &rng)
        return Int32(id)
    }
}
