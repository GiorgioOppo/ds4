import Foundation
import Metal

/// Walsh-Hadamard transform along the last axis. Mirrors `rotate_activation`
/// in `Reference/inference/model.py` lines 247–251.
public enum Hadamard {
    private static let pipeline = Device.shared.makePipeline("hadamard_f32")

    /// Applies the FWHT in place. `x` is treated as `[rows, dim]` where
    /// `rows = x.count / dim`. `dim` must be a power of 2.
    public static func apply(_ x: Tensor, in cmd: MTLCommandBuffer) {
        precondition(x.dtype == .f32, "Hadamard expects f32")
        let dim = x.shape.last!
        precondition(dim > 0 && (dim & (dim - 1)) == 0, "dim must be power of 2")
        let rows = x.count / dim

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        var d = UInt32(dim)
        enc.setBytes(&d, length: 4, index: 1)
        enc.setThreadgroupMemoryLength(dim * MemoryLayout<Float>.size, index: 0)

        let tg = MTLSize(width: dim / 2, height: 1, depth: 1)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// Pure-Swift FWHT used as the test reference.
    public static func referenceCPU(_ row: inout [Float]) {
        let n = row.count
        precondition(n > 0 && (n & (n - 1)) == 0)
        var stride = 1
        while stride < n {
            var i = 0
            while i < n {
                for j in 0..<stride {
                    let a = row[i + j]
                    let b = row[i + j + stride]
                    row[i + j] = a + b
                    row[i + j + stride] = a - b
                }
                i += stride * 2
            }
            stride *= 2
        }
        let s = 1.0 / sqrt(Float(n))
        for k in 0..<n { row[k] *= s }
    }
}
