/// Geometry published in the pinned antirez Kimi K3 GGUF header.
///
/// These constants establish reviewable scaffolding only; future tensor-map
/// validation must still prove every dimension before the runtime gate opens.
public enum KimiK3Shape {
    public static let blockCount = 93
    public static let embeddingLength = 7_168
    public static let contextLength = 1_048_576
    public static let vocabularySize = 163_840
    public static let expertCount = 896
    public static let activeExpertCount = 16
    public static let routedHiddenLength = 3_584
    public static let expertFeedForwardLength = 3_072
}
