import Foundation
import Metal

/// One transformer block with HC mixing wrapping attention and FFN.
/// Mirrors `Block` in `Reference/inference/model.py` lines 647–700.
public final class Block {
    public let layerId: Int
    public let attn: MLA
    public let ffn: MoEFFN
    public let attnNorm: RMSNorm
    public let ffnNorm: RMSNorm
    public let hc: HyperConnections
    public let dim: Int
    public let hcMult: Int

    public let hcAttnFn: Tensor
    public let hcAttnBase: Tensor
    public let hcAttnScale: Tensor
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
        self.dim = config.dim
        self.hcMult = config.hcMult
        self.hcAttnFn = hcAttnFn; self.hcAttnBase = hcAttnBase; self.hcAttnScale = hcAttnScale
        self.hcFfnFn = hcFfnFn; self.hcFfnBase = hcFfnBase; self.hcFfnScale = hcFfnScale
    }

    /// `x`: [B, S, hc, D] f32. Returns same shape.
    ///
    /// `cmd` is `inout`: both `attn` (when the indexer is enabled) and
    /// `ffn` need to commit-and-wait mid-flight to read GPU output back to
    /// host. They replace `cmd` with a fresh buffer on swap; the rest of
    /// this method continues encoding into the swapped value.
    public func callAsFunction(_ x: Tensor, startPos: Int, inputIds: [Int32],
                                in cmd: inout MTLCommandBuffer) -> Tensor {
        precondition(x.dtype == .f32 && x.shape.count == 4)
        let B = x.shape[0], S = x.shape[1]
        let N = B * S

        // ---- Attention sublayer ----
        let xFlat = x.reshape([N, hcMult, dim])

        let attnPre = hc.pre(x: xFlat, hcFn: hcAttnFn,
                             hcScale: hcAttnScale, hcBase: hcAttnBase, in: cmd)
        // attnPre.y: [N, dim]
        let yNorm = attnNorm(attnPre.y, in: cmd).reshape([B, S, dim])
        let attnOut = attn(yNorm, startPos: startPos, in: &cmd)       // [B, S, dim]

        let xMid = hc.post(x: attnOut.reshape([N, dim]),
                           residual: xFlat,
                           post: attnPre.post, comb: attnPre.comb, in: cmd)
        // xMid: [N, hc, dim]

        // ---- FFN sublayer ----
        let ffnPre = hc.pre(x: xMid, hcFn: hcFfnFn,
                            hcScale: hcFfnScale, hcBase: hcFfnBase, in: cmd)
        let yNorm2 = ffnNorm(ffnPre.y, in: cmd).reshape([B, S, dim])
        let ffnOut = ffn(yNorm2, inputIds: inputIds, in: &cmd)        // [B, S, dim]
        let xOut = hc.post(x: ffnOut.reshape([N, dim]),
                           residual: xMid,
                           post: ffnPre.post, comb: ffnPre.comb, in: cmd)
        return xOut.reshape([B, S, hcMult, dim])
    }
}
