import XCTest
import Metal
@testable import DeepSeekKit

final class HadamardTests: XCTestCase {

    /// Compare the Metal FWHT against the pure-Swift reference for a few
    /// dims that show up in the V4 model (128 = index head, 512 = head_dim).
    func testFWHTMatchesReferenceCPU() throws {
        for dim in [4, 16, 64, 128, 512] {
            let rows = 3
            var input = randomRow(rows: rows, dim: dim, seed: UInt64(dim))

            // CPU reference
            var cpu = input
            for r in 0..<rows {
                var slice = Array(cpu[r * dim ..< (r + 1) * dim])
                Hadamard.referenceCPU(&slice)
                for i in 0..<dim { cpu[r * dim + i] = slice[i] }
            }

            // GPU
            let t = input.withUnsafeBytes { raw in
                Tensor.from(bytes: raw, shape: [rows, dim], dtype: .f32)
            }
            let cmd = Device.shared.queue.makeCommandBuffer()!
            Hadamard.apply(t, in: cmd)
            cmd.commit(); cmd.waitUntilCompleted()
            let gpu = t.toFloatArray()

            for i in 0..<gpu.count {
                XCTAssertEqual(gpu[i], cpu[i], accuracy: 1e-4,
                               "dim=\(dim) idx=\(i)")
            }
            _ = input  // keep input alive
        }
    }

    /// FWHT is an involution up to scale (H @ H = N * I, normalized H is its
    /// own inverse). Apply twice → expect input back.
    func testFWHTInvolution() {
        let dim = 256
        var x = randomRow(rows: 1, dim: dim, seed: 7)
        let original = x

        let t = x.withUnsafeBytes { raw in
            Tensor.from(bytes: raw, shape: [1, dim], dtype: .f32)
        }
        let cmd = Device.shared.queue.makeCommandBuffer()!
        Hadamard.apply(t, in: cmd)
        Hadamard.apply(t, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let restored = t.toFloatArray()

        for i in 0..<dim {
            XCTAssertEqual(restored[i], original[i], accuracy: 1e-4)
        }
        _ = x
    }

    private func randomRow(rows: Int, dim: Int, seed: UInt64) -> [Float] {
        var state = seed | 1
        var out = [Float](repeating: 0, count: rows * dim)
        for i in 0..<out.count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let frac = Float(Double(state >> 11) / Double(1 << 53))
            out[i] = (frac - 0.5) * 4
        }
        return out
    }
}
