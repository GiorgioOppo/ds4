import Foundation
import Metal

public final class RoPE {
    public let theta: Float
    private let pipeline: MTLComputePipelineState

    public init(theta: Float) {
        self.theta = theta
        self.pipeline = Device.shared.makePipeline("rope_f32")
    }

    /// Applies rotary embedding in place. `x`: [tokens, heads, headDim] f32.
    public func apply(_ x: Tensor, posOffset: Int, in cmd: MTLCommandBuffer) {
        precondition(x.dtype == .f32 && x.shape.count == 3)
        let T = x.shape[0], H = x.shape[1], D = x.shape[2]

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        var dims = SIMD4<UInt32>(UInt32(T), UInt32(H), UInt32(D), UInt32(posOffset))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 1)
        var th = theta
        enc.setBytes(&th, length: 4, index: 2)
        let grid = MTLSize(width: T, height: H, depth: D / 2)
        let tg = MTLSize(width: min(T, 8), height: min(H, 8), depth: min(D / 2, 8))
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}
