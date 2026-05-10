import Foundation
import Metal

/// Linear layer dispatching to BF16 / FP8 / FP4 GEMM based on weight dtype.
/// Mirrors `linear()` in `Reference/inference/model.py` lines 108–120.
///
/// FP8 path: act_quant(x) → fp8_gemm(x_q, x_s, w, w_s) → f32 out
/// FP4 path: act_quant(x) → fp4_gemm(x_q, x_s, w, w_s) → f32 out
/// Dense BF16/F32 path: direct GEMM, f32 out
public final class Linear {
    public let inFeatures: Int
    public let outFeatures: Int
    public let weight: Tensor
    public let scale: Tensor?

    private static let pBF16ToF32  = Device.shared.makePipeline("gemm_bf16_to_f32")
    private static let pF32BF16    = Device.shared.makePipeline("gemm_f32_bf16_to_f32")
    private static let pF32        = Device.shared.makePipeline("gemm_f32_to_f32")
    private static let pFP8        = Device.shared.makePipeline("gemm_fp8_to_f32")
    private static let pFP4        = Device.shared.makePipeline("gemm_fp8_fp4_to_f32")

    public init(inFeatures: Int, outFeatures: Int, weight: Tensor, scale: Tensor?) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.weight = weight
        self.scale = scale
    }

    /// `x`: [M, K] f32 or bf16. Output: [M, N] f32.
    public func callAsFunction(_ x: Tensor, in cmd: MTLCommandBuffer) -> Tensor {
        let M = x.shape.dropLast().reduce(1, *)
        let outShape = Array(x.shape.dropLast()) + [outFeatures]
        let y = Tensor.empty(shape: outShape, dtype: .f32)

        switch weight.dtype {
        case .fp4E2M1:
            guard let s = scale else { fatalError("FP4 Linear needs scale") }
            fp4Forward(x: x, y: y, M: M, scale: s, in: cmd)
        case .fp8E4M3:
            guard let s = scale else { fatalError("FP8 Linear needs scale") }
            fp8Forward(x: x, y: y, M: M, scale: s, in: cmd)
        case .bf16:
            denseForward(x: x, y: y, M: M, pipelineForFloatX: Self.pF32BF16,
                         pipelineForBFloatX: Self.pBF16ToF32, in: cmd)
        case .f32:
            precondition(x.dtype == .f32, "f32 weight requires f32 input")
            dispatchGEMM(pipeline: Self.pF32, x: x, y: y, M: M, in: cmd)
        default:
            fatalError("Linear: unsupported weight dtype \(weight.dtype)")
        }
        return y
    }

    private func denseForward(x: Tensor, y: Tensor, M: Int,
                              pipelineForFloatX: MTLComputePipelineState,
                              pipelineForBFloatX: MTLComputePipelineState,
                              in cmd: MTLCommandBuffer) {
        let pipeline: MTLComputePipelineState
        switch x.dtype {
        case .f32: pipeline = pipelineForFloatX
        case .bf16: pipeline = pipelineForBFloatX
        default: fatalError("Linear dense: input dtype \(x.dtype) not supported")
        }
        dispatchGEMM(pipeline: pipeline, x: x, y: y, M: M, in: cmd)
    }

    private func dispatchGEMM(pipeline: MTLComputePipelineState,
                              x: Tensor, y: Tensor, M: Int,
                              in cmd: MTLCommandBuffer) {
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(y.buffer, offset: 0, index: 2)
        var dims = SIMD3<UInt32>(UInt32(M), UInt32(outFeatures), UInt32(inFeatures))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)

        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let gx = (outFeatures + 15) / 16
        let gy = (M + 15) / 16
        enc.dispatchThreadgroups(MTLSize(width: gx, height: gy, depth: 1),
                                 threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    private func fp8Forward(x: Tensor, y: Tensor, M: Int, scale wScale: Tensor,
                            in cmd: MTLCommandBuffer) {
        precondition(x.dtype == .f32, "FP8 Linear: input must be f32 for now")
        // Quantize activations to FP8 with per-128 block scales (NOT inplace).
        let aq = ActQuant(format: .fp8)
        let act = aq.quant(x.reshape([M, inFeatures]), inplace: false, in: cmd)
        guard let qbytes = act.qbytes else { fatalError("ActQuant did not produce qbytes") }

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(Self.pFP8)
        enc.setBuffer(qbytes.buffer, offset: 0, index: 0)
        enc.setBuffer(act.scales.buffer, offset: 0, index: 1)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 2)
        enc.setBuffer(wScale.buffer, offset: wScale.offset, index: 3)
        enc.setBuffer(y.buffer, offset: 0, index: 4)
        var dims = SIMD3<UInt32>(UInt32(M), UInt32(outFeatures), UInt32(inFeatures))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 5)

        enc.dispatchThreads(MTLSize(width: outFeatures, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }

    private func fp4Forward(x: Tensor, y: Tensor, M: Int, scale wScale: Tensor,
                            in cmd: MTLCommandBuffer) {
        precondition(x.dtype == .f32, "FP4 Linear: input must be f32 for now")
        // FP4 weights pair with FP8 activations (per the reference); use FP8 act_quant.
        let aq = ActQuant(format: .fp8)
        let act = aq.quant(x.reshape([M, inFeatures]), inplace: false, in: cmd)
        guard let qbytes = act.qbytes else { fatalError("ActQuant did not produce qbytes") }

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(Self.pFP4)
        enc.setBuffer(qbytes.buffer, offset: 0, index: 0)
        enc.setBuffer(act.scales.buffer, offset: 0, index: 1)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 2)
        enc.setBuffer(wScale.buffer, offset: wScale.offset, index: 3)
        enc.setBuffer(y.buffer, offset: 0, index: 4)
        var dims = SIMD3<UInt32>(UInt32(M), UInt32(outFeatures), UInt32(inFeatures))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 5)

        enc.dispatchThreads(MTLSize(width: outFeatures, height: M, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
}
