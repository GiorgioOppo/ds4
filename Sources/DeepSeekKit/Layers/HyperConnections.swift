import Foundation
import MLX
import MLXNN

public final class HyperConnections {
    public let hcMult: Int
    public let normEps: Float
    public let hcEps: Float
    public let dim: Int
    public let sinkhornIters: Int

    private let sinkhorn: HCSinkhorn

    public init(config: ModelConfig, dim: Int) {
        self.hcMult = config.hcMult
        self.normEps = config.normEps
        self.hcEps = config.hcEps
        self.dim = dim
        self.sinkhornIters = config.hcSinkhornIters
        self.sinkhorn = HCSinkhorn(hcMult: config.hcMult,
                                   sinkhornIters: config.hcSinkhornIters,
                                   hcEps: config.hcEps)
    }

    public struct PreOutput {
        public let y: Tensor
        public let post: Tensor
        public let comb: Tensor
    }

    public func pre(x: Tensor, hcFn: Tensor, hcScale: Tensor, hcBase: Tensor) -> PreOutput {
        let xArr = x.array
        let n = xArr.shape[0]
        let hc = xArr.shape[1]
        let d = xArr.shape[2]
        
        let xFlat = xArr.reshaped([n, hc * d])
        
        let rsqrtVal = rsqrt(mean(square(xFlat), axes: [-1], keepDims: true) + normEps)
        
        let mixes = matmul(xFlat, hcFn.array.transposed()) * rsqrtVal
        
        let split = sinkhorn.split(mixes: Tensor(array: mixes, dtype: .f32), hcScale: hcScale, hcBase: hcBase)
        
        let preArr = split.pre.array.expandedDimensions(axes: [2])
        let yArr = (preArr * xArr).sum(axes: [1])
        
        return PreOutput(y: Tensor(array: yArr, dtype: .f32), post: split.post, comb: split.comb)
    }

    public func post(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor {
        let xArr = x.array
        let resArr = residual.array
        let postArr = post.array
        let combArr = comb.array
        
        let term1 = postArr.expandedDimensions(axes: [2]) * xArr.expandedDimensions(axes: [1])
        let term2 = matmul(combArr, resArr)
        
        let yArr = term1 + term2
        return Tensor(array: yArr, dtype: .f32)
    }
}
