import Foundation
import Metal

/// One transformer block for the Llama architecture (TODO §10.2 /
/// T2). Pre-norm residual layout:
///
///   h = x + attn(rmsnorm_attn(x), startPos)
///   y = h + ffn(rmsnorm_ffn(h))
///
/// All four sub-layers (`attnNorm`, `attn`, `ffnNorm`, `ffn`) are
/// passed in by the loader so the architecture dispatcher can swap
/// MoE for dense FFN or change the attention variant without
/// touching this file. Today the Llama path always uses
/// `StandardMHA` + `SwiGLU`; Mistral / Qwen / Code-Llama would
/// reuse the same shape.
///
/// Compared to `DecoderLayer` (DeepSeek): no hyper-connections, no
/// MoE gating, no per-layer `compressRatio` switch. Straight pre-norm.
public final class LlamaDecoderLayer {
    public let layerId: Int
    public let attnNorm: RMSNorm
    public let attn: StandardMHA
    public let ffnNorm: RMSNorm
    public let ffn: SwiGLU

    public init(layerId: Int,
                attnNorm: RMSNorm, attn: StandardMHA,
                ffnNorm: RMSNorm, ffn: SwiGLU)
    {
        self.layerId = layerId
        self.attnNorm = attnNorm
        self.attn = attn
        self.ffnNorm = ffnNorm
        self.ffn = ffn
    }

    /// Forward pass on the residual stream. `x` is `[B, S, hidden]`
    /// F32; output is the same shape. The KV cache update happens
    /// inside `StandardMHA.callAsFunction`.
    public func callAsFunction(_ x: Tensor,
                                startPos: Int,
                                in cmd: inout MTLCommandBuffer) -> Tensor
    {
        precondition(x.dtype == .f32 && x.shape.count == 3,
                      "LlamaDecoderLayer: expected f32 [B, S, hidden]")
        // h = x + attn(attnNorm(x))
        let normedA = attnNorm(x, in: cmd)
        let attnOut = attn(normedA, startPos: startPos, in: &cmd)
        // attnOut += x  → attnOut now holds the residual sum h.
        Elementwise.addInPlace(attnOut, x, in: cmd)

        // y = h + ffn(ffnNorm(h))
        let normedF = ffnNorm(attnOut, in: cmd)
        let ffnOut = ffn(normedF, in: cmd)
        Elementwise.addInPlace(ffnOut, attnOut, in: cmd)
        return ffnOut
    }

    /// Drop the layer's KV cache between unrelated prompts so the
    /// underlying pages can return to the system.
    public func releaseCache() {
        attn.releaseCache()
    }
}
