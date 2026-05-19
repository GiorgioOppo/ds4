import Foundation
import Metal

public enum Elementwise {
    private static let siluMulP = Device.shared.makePipeline("silu_mul_f32")
    private static let axpyP = Device.shared.makePipeline("axpy_f32")
    private static let scaleP = Device.shared.makePipeline("scale_f32")
    private static let addP = Device.shared.makePipeline("add_inplace_f32")
    private static let bf16RTP = Device.shared.makePipeline("bf16_round_trip_f32_inplace")

    /// In-place F32 → BF16 → F32 round-trip on every element of `x`.
    ///
    /// Mirrors the reference's BF16 activation propagation: V4 is trained
    /// with every sub-layer output cast to BF16 (`torch_dtype: bfloat16`),
    /// so each block boundary loses ~16 bits of mantissa precision. Our
    /// forward keeps full F32 through every layer, which is technically
    /// more precise but takes the activations off the training
    /// distribution; calling this at the end of each sub-layer (i.e.
    /// after each `hc.post`) re-injects the missing noise without
    /// touching the GEMM kernels themselves.
    public static func bf16RoundTripInplace(_ x: Tensor,
                                              in cmd: MTLCommandBuffer) {
        precondition(x.dtype == .f32)
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(bf16RTP)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        var n = UInt32(x.count)
        enc.setBytes(&n, length: 4, index: 1)
        dispatch1D(enc, pipeline: bf16RTP, count: x.count)
        enc.endEncoding()
    }

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
        dispatch1D(enc, pipeline: siluMulP, count: g.count)
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
        dispatch1D(enc, pipeline: axpyP, count: y.count)
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
        dispatch1D(enc, pipeline: scaleP, count: x.count)
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
        dispatch1D(enc, pipeline: addP, count: y.count)
        enc.endEncoding()
    }

    /// `y[r, c] = x[r, c] * scale[c]` — per-channel multiply over
    /// the last axis. Used by `Linear`'s AWQ / SmoothQuant
    /// inverse-scale path (the smoothing migrated difficulty to
    /// the weight; the runtime applies the inverse on the
    /// activation here so the math cancels exactly).
    public static func channelScale(_ x: Tensor,
                                       scale: Tensor,
                                       in cmd: MTLCommandBuffer) -> Tensor
    {
        precondition(x.dtype == .f32 && scale.dtype == .f32)
        let cols = x.shape.last!
        precondition(scale.count == cols,
                      "channelScale: scale length \(scale.count) != cols \(cols)")
        let y = Tensor.empty(shape: x.shape, dtype: .f32)
        let pipeline = Device.shared.makePipeline("channel_scale_f32")
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(scale.buffer, offset: scale.offset, index: 1)
        enc.setBuffer(y.buffer, offset: 0, index: 2)
        var dims = SIMD2<UInt32>(UInt32(x.count), UInt32(cols))
        enc.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
        dispatch1D(enc, pipeline: pipeline, count: x.count)
        enc.endEncoding()
        return y
    }

    /// TG calcolato dinamicamente in funzione della pipeline
    /// (`threadExecutionWidth`, `maxTotalThreadsPerThreadgroup`). Vedi
    /// `Sources/DeepSeekKit/PipelineTuning.swift`.
    private static func dispatch1D(_ enc: MTLComputeCommandEncoder,
                                    pipeline: MTLComputePipelineState,
                                    count: Int) {
        let grid = MTLSize(width: count, height: 1, depth: 1)
        enc.dispatchThreads(grid,
                            threadsPerThreadgroup: pipeline.tunedThreadgroup(forGrid: grid))
    }
}
