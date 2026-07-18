import DS4Core
import DS4Metal

/// Registration record for the GLM 5.2 (`glm-dsa`) family.
///
/// The portable model contract, tokenizer and conversation grammar can be
/// validated before numerical inference exists. Runtime capabilities remain
/// empty until the resident Metal graph passes the GLM correctness fixtures;
/// this prevents the GUI or demo from dispatching a GLM file into DeepSeek V4.
public enum GLM52BackendDefinition {
    public static let supportedArchitecture: ModelArchitectureID = .glmDSA
    public static let expectedBlockCount = 79
    public static let executableLayerCount = 78

    public static let modelCapabilities: ModelCapabilities = [
        .chat, .tools, .reasoning, .mixtureOfExperts, .compressedAttention,
    ]

    public static let runtimeCapabilities: BackendCapabilities = []

    /// The enablement switch, forwarded from the target-visible gate in
    /// DS4Metal (`GLM52RuntimeGate.enabled`): selector routing, catalog
    /// availability and demo dispatch all key off that single constant.
    public static var runtimeEnabled: Bool { GLM52RuntimeGate.enabled }
}
