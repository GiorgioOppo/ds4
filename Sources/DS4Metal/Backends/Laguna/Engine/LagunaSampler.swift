import Foundation
import DS4Core

/// Laguna chat sampling — the same thin, faithful wrapper over the shared
/// DS4Core `Sampler` (bit-for-bit port of the C `sample_top_p_min_p`) that
/// `GLM52Sampler` is: seedable C-compatible xorshift RNG, `expf`-float
/// softmax, llama.cpp conventions (`temperature <= 0` collapses to argmax,
/// `topK <= 0` samples the full vocabulary). The reference Laguna defaults
/// live in `LagunaConversationProtocol.SamplingDefaults` (temp 0.7, top-k 20,
/// top-p 0.95, min-p 0.05); explicit caller parameters always win.
public enum LagunaSampler {
    public static let penaltyWindow = 64

    /// Sample one token. `rng` is the caller-owned xorshift state (seed it
    /// from `SamplingParams.seed` for reproducibility).
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

    /// Zero-copy form for the engine's shared logits buffer.
    public static func sample(
        logits: UnsafeBufferPointer<Float>,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repetitionPenalty: Float,
        recentTokens: ArraySlice<Int32>,
        rng: inout UInt64
    ) -> Int32? {
        guard !logits.isEmpty else { return nil }
        let recent = recentTokens.suffix(penaltyWindow).map(Int.init)
        let id = Sampler.sample(
            logits, temperature: temperature,
            topK: topK, topP: topP, minP: minP,
            repetitionPenalty: repetitionPenalty,
            recent: recent[...], rng: &rng)
        return Int32(id)
    }
}
