import Foundation
import MLX
import MLXNN

public final class HCSinkhorn {
    public let hcMult: Int
    public let sinkhornIters: Int
    public let hcEps: Float

    public init(hcMult: Int, sinkhornIters: Int, hcEps: Float) {
        self.hcMult = hcMult
        self.sinkhornIters = sinkhornIters
        self.hcEps = hcEps
    }

    public struct Output {
        public let pre: Tensor
        public let post: Tensor
        public let comb: Tensor
    }

    public func split(mixes: Tensor, hcScale: Tensor, hcBase: Tensor) -> Output {
        let n = mixes.shape[0]
        let hc = hcMult
        
        let mArr = mixes.array
        let sArr = hcScale.array
        let bArr = hcBase.array
        
        let p_mix = mArr[0..., 0..<hc]
        let q_mix = mArr[0..., hc..<(2*hc)]
        let c_mix = mArr[0..., (2*hc)...]
        
        let p_base = bArr[0..<hc]
        let q_base = bArr[hc..<(2*hc)]
        let c_base = bArr[(2*hc)...]
        
        let preArr = sigmoid(p_mix * sArr[0] + p_base) + hcEps
        let postArr = 2 * sigmoid(q_mix * sArr[1] + q_base)
        
        var combArr = c_mix * sArr[2] + c_base
        combArr = combArr.reshaped([n, hc, hc])
        
        combArr = softmax(combArr, axis: 2) + hcEps
        combArr = combArr / (combArr.sum(axes: [1], keepDims: true) + hcEps)
        
        for _ in 1..<sinkhornIters {
            combArr = combArr / (combArr.sum(axes: [2], keepDims: true) + hcEps)
            combArr = combArr / (combArr.sum(axes: [1], keepDims: true) + hcEps)
        }
        
        return Output(pre: Tensor(array: preArr, dtype: .f32),
                      post: Tensor(array: postArr, dtype: .f32),
                      comb: Tensor(array: combArr, dtype: .f32))
    }
}
