import DS4Core
import DS4Metal

/// Registration record for the Kimi K3 (`kimi-k3`) family.
///
/// This first slice intentionally advertises only portable facts read from
/// the published GGUF. Runtime capabilities stay empty until the tokenizer,
/// split-file reader, tensor map and decoder have independent validation.
public enum KimiK3BackendDefinition {
    public static let supportedArchitecture: ModelArchitectureID = .kimiK3
    public static let expectedBlockCount = KimiK3Shape.blockCount

    public static let modelCapabilities: ModelCapabilities = [
        .chat, .reasoning, .mixtureOfExperts, .compressedAttention,
    ]

    public static let runtimeCapabilities: BackendCapabilities = []
    public static var runtimeEnabled: Bool { KimiK3RuntimeGate.enabled }
}
