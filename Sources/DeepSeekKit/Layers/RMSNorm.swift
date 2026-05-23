import Foundation
import MLX
import MLXNN

public final class RMSNorm {
    public let weight: Tensor
    public let eps: Float

    public init(weight: Tensor, eps: Float) {
        self.weight = weight
        self.eps = eps
    }

    public func callAsFunction(_ x: Tensor) -> Tensor {
        let xArr = x.array
        let wArr = weight.array
        
        let varX = mean(square(xArr), axes: [-1], keepDims: true)
        let outArr = xArr * rsqrt(varX + eps) * wArr
        
        return Tensor(array: outArr, dtype: x.dtype)
    }
}
