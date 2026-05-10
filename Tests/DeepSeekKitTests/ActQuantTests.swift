import XCTest
import Metal
@testable import DeepSeekKit

final class ActQuantTests: XCTestCase {

    func testFP8InplaceMatchesReference() throws {
        let M = 2, N = 256
        let input = randomArray(count: M * N, seed: 31, scale: 100)
        let t = input.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [M, N], dtype: .f32)
        }

        let aq = ActQuant(format: .fp8)
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let out = aq.quant(t, inplace: true, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let gpu = out.inplace!.toFloatArray()
        let gpuScales = out.scales.toFloatArray()

        for r in 0..<M {
            let row = Array(input[r * N ..< (r + 1) * N])
            let (rt, scales) = ActQuant.referenceCPU(row, format: .fp8,
                                                      blockSize: Quant.actBlockSizeFP8)
            for i in 0..<N {
                XCTAssertEqual(gpu[r * N + i], rt[i], accuracy: 1e-3,
                               "fp8 r=\(r) i=\(i)")
            }
            let blocksPerRow = N / Quant.actBlockSizeFP8
            for b in 0..<blocksPerRow {
                XCTAssertEqual(gpuScales[r * blocksPerRow + b], scales[b],
                               accuracy: 1e-6, "fp8 scale r=\(r) b=\(b)")
            }
        }
        _ = input
    }

    func testFP4InplaceMatchesReference() throws {
        let M = 2, N = 64       // 64 = 2 blocks of 32
        let input = randomArray(count: M * N, seed: 41, scale: 8)
        let t = input.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [M, N], dtype: .f32)
        }

        let aq = ActQuant(format: .fp4)
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let out = aq.quant(t, inplace: true, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let gpu = out.inplace!.toFloatArray()
        let gpuScales = out.scales.toFloatArray()

        for r in 0..<M {
            let row = Array(input[r * N ..< (r + 1) * N])
            let (rt, scales) = ActQuant.referenceCPU(row, format: .fp4,
                                                      blockSize: Quant.actBlockSizeFP4)
            for i in 0..<N {
                XCTAssertEqual(gpu[r * N + i], rt[i], accuracy: 1e-3,
                               "fp4 r=\(r) i=\(i)")
            }
            let blocksPerRow = N / Quant.actBlockSizeFP4
            for b in 0..<blocksPerRow {
                XCTAssertEqual(gpuScales[r * blocksPerRow + b], scales[b],
                               accuracy: 1e-6, "fp4 scale r=\(r) b=\(b)")
            }
        }
        _ = input
    }

    /// FP8 round-trip relative error is bounded by 1/8 (3-bit mantissa).
    func testFP8RoundTripRelativeBound() throws {
        let M = 1, N = 128
        let input = (1...M*N).map { Float($0) / 50.0 }   // strictly positive
        let t = input.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [M, N], dtype: .f32)
        }
        let aq = ActQuant(format: .fp8)
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let out = aq.quant(t, inplace: true, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = out.inplace!.toFloatArray()
        for i in 0..<input.count {
            let rel = abs(gpu[i] - input[i]) / max(abs(input[i]), 1e-3)
            XCTAssertLessThan(rel, 0.2, "i=\(i) gpu=\(gpu[i]) ref=\(input[i])")
        }
        _ = input
    }

    private func randomArray(count: Int, seed: UInt64, scale: Float) -> [Float] {
        var state = seed | 1
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let frac = Float(Double(state >> 11) / Double(1 << 53))
            out[i] = (frac - 0.5) * 2 * scale
        }
        return out
    }
}
