import CDS4

/// Token sampling configuration. `.greedy` is deterministic argmax; the default
/// matches the engine's documented API defaults (temperature 1, top_p 1,
/// min_p 0.05 — relative-probability filtering rather than nucleus mass).
public struct SamplingParams: Sendable, Equatable {
    public var temperature: Float
    public var topK: Int32
    public var topP: Float
    public var minP: Float

    public init(temperature: Float = DS4_DEFAULT_TEMPERATURE,
                topK: Int32 = 0,
                topP: Float = DS4_DEFAULT_TOP_P,
                minP: Float = DS4_DEFAULT_MIN_P) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.minP = minP
    }

    /// Deterministic greedy decoding (temperature 0 -> argmax in the engine).
    public static let greedy = SamplingParams(temperature: 0, topK: 0, topP: 1, minP: 0)

    public var isGreedy: Bool { temperature <= 0 }
}
