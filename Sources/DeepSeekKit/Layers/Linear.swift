import Foundation
import Metal

/// Linear layer dispatching to BF16 / FP8 / FP4 GEMM based on weight dtype,
/// matching `linear()` in `Original/DeepSeek-V4-Pro/inference/model.py`.
///
/// FP8 path: act_quant(x) → fp8_gemm(x_q, x_s, w, w_s) → BF16 out
/// FP4 path: act_quant(x) → fp4_gemm(x_q, x_s, w, w_s) → BF16 out
///
/// The act_quant / fp8_gemm / fp4_gemm kernels are not yet implemented; the
/// public callable here will trap until they are. Bias is not supported in
/// the reference (assert bias is None).
public final class Linear {
    public let inFeatures: Int
    public let outFeatures: Int
    public let weight: Tensor
    public let scale: Tensor?    // [out, in/blockK] in E8M0 for fp8/fp4 weights

    public init(inFeatures: Int, outFeatures: Int, weight: Tensor, scale: Tensor?) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.weight = weight
        self.scale = scale
    }

    public func callAsFunction(_ x: Tensor, in cmd: MTLCommandBuffer) -> Tensor {
        switch weight.dtype {
        case .fp4E2M1:
            return fp4Forward(x, in: cmd)
        case .fp8E4M3:
            return fp8Forward(x, in: cmd)
        case .f32, .f16, .bf16:
            return denseForward(x, in: cmd)
        default:
            fatalError("Linear: unsupported weight dtype \(weight.dtype)")
        }
    }

    private func denseForward(_ x: Tensor, in cmd: MTLCommandBuffer) -> Tensor {
        // BF16/FP16/FP32 dense GEMM — needs a generic GEMM kernel that
        // covers x:[M, K] f32 @ w:[N, K] (same dtype as x) → y:[M, N] f32.
        // The previous int4 GEMM has been removed; the BF16 GEMM is one of
        // the next things to write. See README roadmap.
        fatalError("Dense Linear not implemented — needs gemm_bf16.metal")
    }

    private func fp8Forward(_ x: Tensor, in cmd: MTLCommandBuffer) -> Tensor {
        fatalError("FP8 Linear not implemented — needs act_quant.metal + fp8_gemm.metal")
    }

    private func fp4Forward(_ x: Tensor, in cmd: MTLCommandBuffer) -> Tensor {
        fatalError("FP4 Linear not implemented — needs act_quant.metal + fp4_gemm.metal")
    }
}
