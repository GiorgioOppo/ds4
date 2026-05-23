import Foundation
import MLX
import MLXNN

public final class MTPBlock {
    public let block: Block
    public let eProj: Linear
    public let hProj: Linear
    public let eNorm: RMSNorm
    public let hNorm: RMSNorm
    public let norm: RMSNorm
    public let hcHeadFn: Tensor
    public let hcHeadBase: Tensor
    public let hcHeadScale: Tensor

    public weak var embed: ParallelEmbedding?
    public weak var head: ParallelHead?

    public init(block: Block,
                eProj: Linear, hProj: Linear,
                eNorm: RMSNorm, hNorm: RMSNorm, norm: RMSNorm,
                hcHeadFn: Tensor, hcHeadBase: Tensor, hcHeadScale: Tensor) {
        self.block = block
        self.eProj = eProj; self.hProj = hProj
        self.eNorm = eNorm; self.hNorm = hNorm; self.norm = norm
        self.hcHeadFn = hcHeadFn; self.hcHeadBase = hcHeadBase; self.hcHeadScale = hcHeadScale
    }

    public func callAsFunction(_ x: Tensor, startPos: Int, inputIds: [Int32]) -> Tensor {
        guard let embed = embed, let head = head else {
            fatalError("MTPBlock.embed and .head must be wired by the parent Transformer")
        }
        let B = x.shape[0], S = x.shape[1], HC = x.shape[2], D = x.shape[3]
        let N = B * S

        let e = embed.lookup(inputIds)
        let eN = eNorm(e)

        let xN = hNorm(Tensor(array: x.array.reshaped([N, HC * D]), dtype: x.dtype)).array.reshaped([N, HC, D])

        let eProjOut = eProj(eN).array.reshaped([N, 1, D])
        let hProjOut = hProj(Tensor(array: xN.reshaped([N, HC * D]), dtype: x.dtype)).array.reshaped([N, HC, D])

        let combined = hProjOut + eProjOut
        
        let combined4 = Tensor(array: combined.reshaped([B, S, HC, D]), dtype: .f32)
        let after = block(combined4, startPos: startPos, inputIds: inputIds)

        return head(after, hcFn: hcHeadFn, hcScale: hcHeadScale,
                    hcBase: hcHeadBase, norm: norm)
    }
}
