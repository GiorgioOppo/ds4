import DS4Core
import DS4Metal

/// Registration record for the Laguna S 2.1 (`laguna`) family.
public enum LagunaBackendDefinition {
    public static let supportedArchitecture: ModelArchitectureID = .laguna
    public static let expectedBlockCount = 48

    /// The complete frontend exists (native chat, interleaved reasoning,
    /// tagged tool calls), so the portable capabilities can be advertised
    /// even while the decoder is missing.
    public static let modelCapabilities: ModelCapabilities = [
        .chat, .tools, .reasoning, .mixtureOfExperts,
    ]

    /// What the Laguna runtime SERVES today, behind the same opt-in gate as
    /// the selector routing: chat generation with interleaved reasoning and
    /// native tool calls over the resident/streaming engine
    /// (`LagunaChatService`). Outside stay the GLM/DeepSeek-specific
    /// surfaces (auto-tune, distributed).
    public static var runtimeCapabilities: BackendCapabilities {
        guard runtimeEnabled else { return [] }
        return [
            .generation, .reasoning, .tools, .diskKV, .expertRouting,
        ]
    }

    /// The enablement switch, forwarded from the target-visible gate in
    /// DS4Metal (`LagunaRuntimeGate.enabled`): selector routing, catalog
    /// availability and demo dispatch all key off that single constant.
    public static var runtimeEnabled: Bool { LagunaRuntimeGate.enabled }
}
