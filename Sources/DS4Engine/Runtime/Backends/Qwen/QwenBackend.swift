import DS4Core

/// Registration record for recognized Qwen GGUF families.
///
/// Prompt and decoder implementations intentionally do not exist yet. Inspection
/// can still describe the model and hide DeepSeek-only UI controls, while backend
/// construction fails explicitly before any DeepSeek metadata is parsed.
public enum QwenBackendDefinition {
    public static func modelCapabilities(for architecture: ModelArchitectureID)
        -> ModelCapabilities {
        // Family recognition alone cannot certify a variant's tool grammar,
        // reasoning markers or MoE tensor schema. Those flags will be enabled
        // only after inspecting and testing the corresponding metadata/template.
        _ = architecture
        return [.chat]
    }

    /// No runtime generation capability is advertised until the Qwen decoder and
    /// conversation format have been implemented and validated together.
    public static let runtimeCapabilities: BackendCapabilities = []
}
