import XCTest
import Metal
@testable import DeepSeekKit

final class OverlapTransformTests: XCTestCase {

    func testMatchesCPU() throws {
        let B = 2, S = 5, R = 4, D = 6
        let pad: Float = -42.0
        let x = randomArray(B * S * R * 2 * D, seed: 1)
        let xT = x.withUnsafeBytes { Tensor.from(bytes: $0, shape: [B, S, R, 2 * D], dtype: .f32) }

        let cmd = Device.shared.queue.makeCommandBuffer()!
        let yT = OverlapTransform.apply(xT, padValue: pad, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = yT.toFloatArray()

        let cpu = OverlapTransform.referenceCPU(x, B: B, S: S, R: R, D: D, padValue: pad)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], cpu[i], accuracy: 0,
                           "i=\(i)")    // exact equality — pure shuffle
        }
        _ = x
    }

    /// At s == 0 the first R block is filled with the pad value.
    func testFirstRowIsPadded() throws {
        let B = 1, S = 2, R = 3, D = 2
        let x = (0..<B * S * R * 2 * D).map { Float($0) }
        let xT = x.withUnsafeBytes { Tensor.from(bytes: $0, shape: [B, S, R, 2 * D], dtype: .f32) }
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let yT = OverlapTransform.apply(xT, padValue: 99.0, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = yT.toFloatArray()

        // Output shape [1, 2, 6, 2]. For s=0, j<R (0..3) → pad.
        for j in 0..<R {
            for di in 0..<D {
                let idx = ((0 * S + 0) * (2 * R) + j) * D + di
                XCTAssertEqual(gpu[idx], 99.0, "expected pad at out[0,0,\(j),\(di)]")
            }
        }
        _ = x
    }

    private func randomArray(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed | 1
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let frac = Float(Double(state >> 11) / Double(1 << 53))
            out[i] = (frac - 0.5) * 4
        }
        return out
    }
}
