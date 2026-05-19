import Foundation

/// Which transformer family a checkpoint belongs to (TODO §10.2 /
/// T2). Selected at load time from either explicit user choice (CLI
/// flag) or GGUF `general.architecture` metadata. The loader then
/// dispatches to the matching `LlamaModel` / `Transformer` factory.
///
/// The two paths share most of the engine — Linear, RMSNorm, RoPE,
/// SwiGLU, kernels — but differ in their attention block (MLA vs.
/// standard GQA) and their FFN (MoE vs. dense). Adding a new
/// architecture (e.g. Mamba) would mean a new case here plus a
/// distinct forward pass; the engine's primitives stay reusable.
public enum ModelArchitecture: String, Sendable, CaseIterable {
    /// DeepSeek-V4 (MLA + MoE + HyperConnections). The default and
    /// the only path that has full feature coverage today.
    case deepseekV4

    /// Llama / Mistral / Qwen / CodeLlama (standard GQA + dense
    /// SwiGLU). New as of T2; the GGUF loader picks this when the
    /// file's `general.architecture` metadata is one of "llama",
    /// "mistral", "qwen", "qwen2" (all share the layer shape).
    case llama

    /// Resolve a GGUF `general.architecture` string to a known
    /// architecture. Returns nil for unrecognized values so the
    /// caller can decide whether to fall back or error out.
    public static func fromGGUFArchString(_ s: String) -> ModelArchitecture? {
        switch s.lowercased() {
        case "deepseek-v4", "deepseek_v4", "deepseekv4":
            return .deepseekV4
        case "llama", "llama2", "llama3",
             "mistral", "qwen", "qwen2", "qwen3",
             "codellama", "code-llama":
            return .llama
        default:
            return nil
        }
    }
}
