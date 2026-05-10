import Foundation
import Metal

public final class Linear {
    public let outFeatures: Int
    public let inFeatures: Int
    public let weight: Tensor               // either f32/f16 dense, or i4 packed
    public let bias: Tensor?

    private let pipelineGEMM: MTLComputePipelineState
    private let pipelineQ4: MTLComputePipelineState

    public init(weight: Tensor, bias: Tensor? = nil, outFeatures: Int, inFeatures: Int) {
        self.weight = weight
        self.bias = bias
        self.outFeatures = outFeatures
        self.inFeatures = inFeatures
        self.pipelineGEMM = Device.shared.makePipeline("matmul_f32")
        self.pipelineQ4 = Device.shared.makePipeline("matvec_q4")
    }

    /// y = x @ weight^T  (+ bias)
    /// `x`: [M, inFeatures] f32. Output: [M, outFeatures] f32.
    public func callAsFunction(_ x: Tensor, in cmd: MTLCommandBuffer) -> Tensor {
        precondition(x.dtype == .f32, "Linear input must be f32 (cast upstream)")
        let M = x.shape.dropLast().reduce(1, *)
        let y = Tensor.empty(shape: Array(x.shape.dropLast()) + [outFeatures], dtype: .f32)

        let enc = cmd.makeComputeCommandEncoder()!
        if weight.dtype == .i4 {
            precondition(M == 1, "Q4 path is matvec only; use prefill GEMM for M>1")
            enc.setComputePipelineState(pipelineQ4)
            enc.setBuffer(x.buffer, offset: x.offset, index: 0)
            enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
            enc.setBuffer(y.buffer, offset: 0, index: 2)
            var dims = SIMD2<UInt32>(UInt32(outFeatures), UInt32(inFeatures))
            enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
            let tg = MTLSize(width: 64, height: 1, depth: 1)
            let gr = MTLSize(width: outFeatures, height: 1, depth: 1)
            enc.dispatchThreads(gr, threadsPerThreadgroup: tg)
        } else {
            enc.setComputePipelineState(pipelineGEMM)
            enc.setBuffer(x.buffer, offset: x.offset, index: 0)
            enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
            enc.setBuffer(y.buffer, offset: 0, index: 2)
            var dims = SIMD3<UInt32>(UInt32(M), UInt32(outFeatures), UInt32(inFeatures))
            enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
            let tg = MTLSize(width: 16, height: 16, depth: 1)
            let gx = (outFeatures + 15) / 16
            let gy = (M + 15) / 16
            enc.dispatchThreadgroups(MTLSize(width: gx, height: gy, depth: 1), threadsPerThreadgroup: tg)
        }
        enc.endEncoding()
        return y
    }
}
