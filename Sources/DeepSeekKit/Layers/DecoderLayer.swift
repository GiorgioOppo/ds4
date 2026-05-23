import Foundation
import MLX
import MLXNN

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

    public func callAsFunction(_ x: Tensor, startPos: Int, inputIds: [Int32]) -> Tensor {
        let B = x.shape[0]
        let S = x.shape[1]
        let N = B * S

        let xFlat = Tensor(array: x.array.reshaped([N, hcMult, dim]), dtype: x.dtype)

        let attnPre = hc.pre(x: xFlat, hcFn: hcAttnFn, hcScale: hcAttnScale, hcBase: hcAttnBase)
        
        let yNorm = attnNorm(attnPre.y).array.reshaped([B, S, dim])
        
        let attnOut = attn(Tensor(array: yNorm, dtype: .f32), startPos: startPos)
        
        let xMid = hc.post(x: Tensor(array: attnOut.array.reshaped([N, dim]), dtype: .f32),
                           residual: xFlat,
                           post: attnPre.post, comb: attnPre.comb)

        let ffnPre = hc.pre(x: xMid, hcFn: hcFfnFn, hcScale: hcFfnScale, hcBase: hcFfnBase)
        
        let yNorm2 = ffnNorm(ffnPre.y).array.reshaped([B, S, dim])
        
        let ffnOut = ffn(Tensor(array: yNorm2, dtype: .f32), inputIds: inputIds)
        
        let xOut = hc.post(x: Tensor(array: ffnOut.array.reshaped([N, dim]), dtype: .f32),
                           residual: xMid,
                           post: ffnPre.post, comb: ffnPre.comb)
                           
        return Tensor(array: xOut.array.reshaped([B, S, hcMult, dim]), dtype: x.dtype)
    }
}
