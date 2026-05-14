import Foundation
import Metal

public final class RMSNorm {
    public let weight: Tensor
    public let eps: Float
    private let pipelineF32W: MTLComputePipelineState
    private let pipelineBF16W: MTLComputePipelineState

    public init(weight: Tensor, eps: Float) {
        self.weight = weight
        self.eps = eps
        self.pipelineF32W = Device.shared.makePipeline("rmsnorm_f32")
        self.pipelineBF16W = Device.shared.makePipeline("rmsnorm_bf16w_f32")
    }

    public func callAsFunction(_ x: Tensor, in cmd: MTLCommandBuffer) -> Tensor {
        precondition(x.dtype == .f32)
        let dim = x.shape.last!
        let rows = x.count / dim
        let y = Tensor.empty(shape: x.shape, dtype: .f32)

        // HF-native checkpoints (`torch_dtype: bfloat16`) ship every
        // RMSNorm gain as BF16. The earlier code path always
        // dispatched the f32-weight kernel, which read 4 bytes per
        // element from a 2-bytes-per-element buffer; in streaming-
        // pool mode the spill went into the next tensor's bytes and
        // produced 1e35 values that NaN'd the whole forward at the
        // first RMSNorm whose neighbour happened to be FP8 weight
        // codes. Dispatch by dtype.
        let pipeline: MTLComputePipelineState
        switch weight.dtype {
        case .f32:  pipeline = pipelineF32W
        case .bf16: pipeline = pipelineBF16W
        default:
            fatalError("RMSNorm: weight dtype \(weight.dtype) not supported (expected f32 or bf16)")
        }

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
