import DS4Core

/// Registration record for the existing, fully implemented DeepSeek-V4 path.
/// The numerical implementation remains concrete (`StreamingDecoder`) so the
/// decode loop gains no existential or protocol-dispatch overhead.
public enum DeepSeekV4BackendDefinition {
    /// Single declaration of the DeepSeek-V4 profiles that this build can run
    /// locally. Runtime selection and the model catalog both derive their gate
    /// from this list, so enabling a profile cannot leave the GUI and backend in
    /// disagreement.
    public static let locallyRunnableVariants: [DeepSeekV4Variant] = [.flash, .pro]

    public static var supportedLayerCounts: Set<Int> {
        Set(locallyRunnableVariants.map { variant in
            switch variant {
            case .flash: Int(DeepSeekV4Shape.flash.nLayer)
            case .pro: Int(DeepSeekV4Shape.pro.nLayer)
            }
        })
    }

    public static func supportsLocalRuntime(_ variant: DeepSeekV4Variant) -> Bool {
        locallyRunnableVariants.contains { $0.rawValue == variant.rawValue }
    }

    public static let modelCapabilities: ModelCapabilities = [
        .chat, .tools, .reasoning, .mixtureOfExperts, .compressedAttention,
    ]

    public static let runtimeCapabilities: BackendCapabilities = [
        .generation, .reasoning, .tools, .diskKV,
        .expertRouting, .expertBundle, .deepSeekPerformanceTuning,
        .distributedPipeline, .expertParallel,
    ]
}
