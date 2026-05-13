import Foundation
import Metal

public enum Elementwise {
    private static let siluMulP = Device.shared.makePipeline("silu_mul_f32")
    private static let axpyP = Device.shared.makePipeline("axpy_f32")
    private static let scaleP = Device.shared.makePipeline("scale_f32")
    private static let addP = Device.shared.makePipeline("add_inplace_f32")

    /// SwiGLU body: `y = silu(gate) * up`. When `swigluLimit > 0`, applies
    /// the V4-Flash clipping (`gate.clamp(max=limit)`, `up.clamp(±limit)`)
    /// described in Reference/inference/model.py:600-603. Without this
    /// clipping, activations in deep V4 models compound to ~5000× across
    /// 43 layers and the final residual stream loses prompt-conditioned
    /// signal.
    public static func siluMul(_ g: Tensor, _ u: Tensor,
                                 swigluLimit: Float = 0.0,
                                 in cmd: MTLCommandBuffer) -> Tensor {
        precondition(g.shape == u.shape && g.dtype == .f32 && u.dtype == .f32)
        let y = Tensor.empty(shape: g.shape, dtype: .f32)
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(siluMulP)
        enc.setBuffer(g.buffer, offset: g.offset, index: 0)
        enc.setBuffer(u.buffer, offset: u.offset, index: 1)
        enc.setBuffer(y.buffer, offset: 0, index: 2)
        var n = UInt32(g.count)
        enc.setBytes(&n, length: 4, index: 3)
        var lim = swigluLimit
        enc.setBytes(&lim, length: 4, index: 4)
        dispatch1D(enc, count: g.count)
        enc.endEncoding()
        return y
    }

    public static func axpy(_ y: Tensor, _ x: Tensor, alpha: Float, in cmd: MTLCommandBuffer) {
        precondition(y.shape == x.shape && y.dtype == .f32 && x.dtype == .f32)
        var a = alpha
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(axpyP)
        enc.setBuffer(y.buffer, offset: y.offset, index: 0)
        enc.setBuffer(x.buffer, offset: x.offset, index: 1)
        enc.setBytes(&a, length: 4, index: 2)
        var n = UInt32(y.count)
        enc.setBytes(&n, length: 4, index: 3)
        dispatch1D(enc, count: y.count)
        enc.endEncoding()
    }

    public static func scale(_ x: Tensor, by alpha: Float, in cmd: MTLCommandBuffer) -> Tensor {
        let y = Tensor.empty(shape: x.shape, dtype: .f32)
        var a = alpha
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(scaleP)
        enc.setBuffer(y.buffer, offset: 0, index: 0)
        enc.setBuffer(x.buffer, offset: x.offset, index: 1)
        enc.setBytes(&a, length: 4, index: 2)
        var n = UInt32(x.count)
        enc.setBytes(&n, length: 4, index: 3)
        dispatch1D(enc, count: x.count)
        enc.endEncoding()
        return y
    }

    public static func addInPlace(_ y: Tensor, _ x: Tensor, in cmd: MTLCommandBuffer) {
        precondition(y.shape == x.shape)
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(addP)
        enc.setBuffer(y.buffer, offset: y.offset, index: 0)
        enc.setBuffer(x.buffer, offset: x.offset, index: 1)
        var n = UInt32(y.count)
        enc.setBytes(&n, length: 4, index: 2)
        dispatch1D(enc, count: y.count)
        enc.endEncoding()
    }

    private static func dispatch1D(_ enc: MTLComputeCommandEncoder, count: Int) {
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
    }
}
