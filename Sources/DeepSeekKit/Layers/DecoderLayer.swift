import Foundation
import Metal

/// One transformer block with HC mixing wrapping attention and FFN.
/// Mirrors `Block` in `Reference/inference/model.py` lines 647–700.
///
/// forward(x):
///   residual = x
///   x, post, comb = hc_pre(x, hc_attn_fn, hc_attn_scale, hc_attn_base)
///   x = attn_norm(x); x = attn(x, start_pos)
///   x = hc_post(x, residual, post, comb)
///
///   residual = x
///   x, post, comb = hc_pre(x, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
///   x = ffn_norm(x); x = ffn(x, input_ids)
///   x = hc_post(x, residual, post, comb)
///   return x
public final class Block {
    public let layerId: Int
    public let attn: MLA
    public let ffn: MoEFFN
    public let attnNorm: RMSNorm
    public let ffnNorm: RMSNorm
    public let hc: HyperConnections

    // Per-sublayer HC parameters (attn + ffn each get their own triple)
    public let hcAttnFn: Tensor          // [(2+hc)*hc, hc*dim] f32
    public let hcAttnBase: Tensor        // [(2+hc)*hc] f32
    public let hcAttnScale: Tensor       // [3] f32
    public let hcFfnFn: Tensor
    public let hcFfnBase: Tensor
    public let hcFfnScale: Tensor

    public init(layerId: Int, config: ModelConfig,
                attn: MLA, ffn: MoEFFN,
                attnNorm: RMSNorm, ffnNorm: RMSNorm,
                hcAttnFn: Tensor, hcAttnBase: Tensor, hcAttnScale: Tensor,
                hcFfnFn: Tensor, hcFfnBase: Tensor, hcFfnScale: Tensor) {
        self.layerId = layerId
        self.attn = attn; self.ffn = ffn
        self.attnNorm = attnNorm; self.ffnNorm = ffnNorm
        self.hc = HyperConnections(config: config, dim: config.dim)
        self.hcAttnFn = hcAttnFn; self.hcAttnBase = hcAttnBase; self.hcAttnScale = hcAttnScale
        self.hcFfnFn = hcFfnFn; self.hcFfnBase = hcFfnBase; self.hcFfnScale = hcFfnScale
    }

    /// Forward — input is [b, s, hc, d], output same.
    /// NOT IMPLEMENTED until HyperConnections.pre/post and MLA.forward
    /// and MoEFFN.forward are.
    public func callAsFunction(_ x: Tensor, startPos: Int, inputIds: [Int],
                               in cmd: MTLCommandBuffer) -> Tensor {
        fatalError("Block.forward depends on MLA / MoE / HC implementations")
    }
}
