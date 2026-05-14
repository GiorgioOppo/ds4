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

        @inline(__always) func traceHere(_ name: String, _ t: Tensor) {
            guard TraceFlags.normTrace && (layerId == 0 || layerId == 5 || layerId == 6) else { return }
            cmd.commit(); cmd.waitUntilCompleted()
            traceTensorStats("block[\(layerId)] \(name)", t)
            cmd = Device.shared.queue.makeCommandBuffer()!
        }

        // ---- Attention sublayer ----
        let xFlat = x.reshape([N, hcMult, dim])

        let attnPre = hc.pre(x: xFlat, hcFn: hcAttnFn,
                             hcScale: hcAttnScale, hcBase: hcAttnBase, in: cmd)
        traceHere("after hc.pre(attn).y", attnPre.y)
        traceHere("after hc.pre(attn).post", attnPre.post)
        traceHere("after hc.pre(attn).comb", attnPre.comb)
        // attnPre.y: [N, dim]
        let yNorm = attnNorm(attnPre.y, in: cmd).reshape([B, S, dim])
        traceHere("after attnNorm", yNorm)
        let attnOut = attn(yNorm, startPos: startPos, in: &cmd)       // [B, S, dim]
        traceHere("after attn (MLA returned)", attnOut)

        let xMid = hc.post(x: attnOut.reshape([N, dim]),
                           residual: xFlat,
                           post: attnPre.post, comb: attnPre.comb, in: cmd)
        // BF16 round-trip on the residual stream — mirrors the reference's
        traceHere("after hc.post(attn)", xMid)
        // xMid: [N, hc, dim]

        // ---- FFN sublayer ----
        let ffnPre = hc.pre(x: xMid, hcFn: hcFfnFn,
                            hcScale: hcFfnScale, hcBase: hcFfnBase, in: cmd)
        traceHere("after hc.pre(ffn).y", ffnPre.y)
        let yNorm2 = ffnNorm(ffnPre.y, in: cmd).reshape([B, S, dim])
        traceHere("after ffnNorm", yNorm2)
        let ffnOut = ffn(yNorm2, inputIds: inputIds, in: &cmd)        // [B, S, dim]
        traceHere("after ffn", ffnOut)
        let xOut = hc.post(x: ffnOut.reshape([N, dim]),
                           residual: xMid,
                           post: ffnPre.post, comb: ffnPre.comb, in: cmd)
        traceHere("after hc.post(ffn)", xOut)
        return xOut.reshape([B, S, hcMult, dim])
    }
}
