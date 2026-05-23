import Foundation
import MLX
import MLXNN

public final class Linear {
    public let inFeatures: Int
    public let outFeatures: Int
    public let weight: Tensor
    public let scale: Tensor?

    public let castOutputToBF16: Bool
    public let useW8A8Activations: Bool
    public var inverseChannelScale: Tensor? = nil

    public init(inFeatures: Int, outFeatures: Int, weight: Tensor, scale: Tensor?,
                castOutputToBF16: Bool = false,
                useW8A8Activations: Bool = false,
                inverseChannelScale: Tensor? = nil) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.weight = weight
        self.scale = scale
        self.castOutputToBF16 = castOutputToBF16
        self.useW8A8Activations = useW8A8Activations
        self.inverseChannelScale = inverseChannelScale
    }

    public func callAsFunction(_ xIn: Tensor) -> Tensor {
        var xArr = xIn.array
        
        if let invScale = inverseChannelScale {
            xArr = xArr * invScale.array
        }
        
        // Typical PyTorch / safetensors linear weight shape is [outFeatures, inFeatures]
        // so we transpose the weight for x * W^T
        let wArr = weight.array
        // If weight shape is already [inFeatures, outFeatures] (e.g. converted GGUF),
        // we might not need to transpose. We'll assume transposed() for standard [out, in].
        let transposeNeeded = wArr.shape.last == inFeatures
        
        let yArr: MLXArray
        if transposeNeeded {
            yArr = matmul(xArr, wArr.transposed())
        } else {
            yArr = matmul(xArr, wArr)
        }
        
        var outArr = yArr
        if let s = scale {
             outArr = outArr * s.array
        }
        
        if castOutputToBF16 {
            outArr = outArr.asType(.bfloat16)
        }
        
        return Tensor(array: outArr, dtype: castOutputToBF16 ? .bf16 : .f32)
    }
}
