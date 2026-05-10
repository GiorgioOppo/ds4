import Foundation
import Metal

/// Compressor's overlap shuffle. See model.py:307-314.
public enum OverlapTransform {
    private static let pipeline = Device.shared.makePipeline("overlap_transform_f32")

    /// `x`: [B, S, R, 2D] f32. Returns [B, S, 2R, D] f32.
    public static func apply(_ x: Tensor, padValue: Float = 0,
                             in cmd: MTLCommandBuffer) -> Tensor {
        precondition(x.dtype == .f32 && x.shape.count == 4)
        let B = x.shape[0], S = x.shape[1], R = x.shape[2]
        let twoD = x.shape[3]
        precondition(twoD % 2 == 0, "last dim must be 2*D")
        let D = twoD / 2

        let out = Tensor.empty(shape: [B, S, 2 * R, D], dtype: .f32)

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        var dims = SIMD4<UInt32>(UInt32(B), UInt32(S), UInt32(R), UInt32(D))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 2)
        var pad = padValue
        enc.setBytes(&pad, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: D, height: 2 * R, depth: B * S),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 4, depth: 4))
        enc.endEncoding()
        return out
    }

    public static func referenceCPU(_ x: [Float], B: Int, S: Int, R: Int, D: Int,
                                     padValue: Float) -> [Float] {
        var out = [Float](repeating: 0, count: B * S * 2 * R * D)
        for b in 0..<B {
            for s in 0..<S {
                for j in 0..<(2 * R) {
                    for di in 0..<D {
                        let outIdx = ((b * S + s) * (2 * R) + j) * D + di
                        if j >= R {
                            let inIdx = ((b * S + s) * R + (j - R)) * (2 * D) + (D + di)
                            out[outIdx] = x[inIdx]
                        } else if s > 0 {
                            let inIdx = ((b * S + (s - 1)) * R + j) * (2 * D) + di
                            out[outIdx] = x[inIdx]
                        } else {
                            out[outIdx] = padValue
                        }
                    }
                }
            }
        }
        return out
    }
}
