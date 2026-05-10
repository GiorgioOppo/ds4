import Foundation
import Metal

public final class RMSNorm {
    public let weight: Tensor
    public let eps: Float
    private let pipeline: MTLComputePipelineState

    public init(weight: Tensor, eps: Float) {
        self.weight = weight
        self.eps = eps
        self.pipeline = Device.shared.makePipeline("rmsnorm_f32")
    }

    public func callAsFunction(_ x: Tensor, in cmd: MTLCommandBuffer) -> Tensor {
        precondition(x.dtype == .f32)
        let dim = x.shape.last!
        let rows = x.count / dim
        let y = Tensor.empty(shape: x.shape, dtype: .f32)

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(y.buffer, offset: 0, index: 2)
        var d = UInt32(dim); var e = eps
        enc.setBytes(&d, length: 4, index: 3)
        enc.setBytes(&e, length: 4, index: 4)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: tg)
        enc.endEncoding()
        return y
    }
}
