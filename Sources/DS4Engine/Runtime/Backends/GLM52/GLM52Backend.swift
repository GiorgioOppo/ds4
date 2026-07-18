import DS4Core
import DS4Metal

/// Registration record for the GLM 5.2 (`glm-dsa`) family.
public enum GLM52BackendDefinition {
    public static let supportedArchitecture: ModelArchitectureID = .glmDSA
    public static let expectedBlockCount = 79
    public static let executableLayerCount = 78

    public static let modelCapabilities: ModelCapabilities = [
        .chat, .tools, .reasoning, .mixtureOfExperts, .compressedAttention,
    ]

    /// What the GLM runtime SERVES today — the GUI mostra i controlli
    /// (think, tool, disk KV, knob esperti, bottone sidecar) in base a
    /// questo set. Fuori restano solo le capacità DeepSeek-specifiche
    /// (tuning misurato, distribuito).
    public static let runtimeCapabilities: BackendCapabilities = [
        .generation, .reasoning, .tools, .diskKV, .expertRouting,
        .expertBundle,
    ]

    /// The enablement switch, forwarded from the target-visible gate in
    /// DS4Metal (`GLM52RuntimeGate.enabled`): selector routing, catalog
    /// availability and demo dispatch all key off that single constant.
    public static var runtimeEnabled: Bool { GLM52RuntimeGate.enabled }
}
