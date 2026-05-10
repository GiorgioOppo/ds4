import XCTest
import Metal
@testable import DeepSeekKit

final class SoftmaxAxisTests: XCTestCase {

    func testLastAxis() throws {
        try compare(shape: [3, 4, 5], axis: 2, seed: 1)
    }

    func testMiddleAxis() throws {
        try compare(shape: [3, 4, 5], axis: 1, seed: 2)
    }

    func testFirstAxis() throws {
        try compare(shape: [3, 4, 5], axis: 0, seed: 3)
    }

    /// Compressor's actual layout: [b, num_blocks, ratio, head_dim] softmax over axis=2.
    func testCompressorShape() throws {
        try compare(shape: [2, 6, 4, 8], axis: 2, seed: 4)
    }

    private func compare(shape: [Int], axis: Int, seed: UInt64) throws {
        let count = shape.reduce(1, *)
        let input = randomArray(count, seed: seed)
        let t = input.withUnsafeBytes { Tensor.from(bytes: $0, shape: shape, dtype: .f32) }

        let cmd = Device.shared.queue.makeCommandBuffer()!
        SoftmaxAxis.apply(t, axis: axis, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = t.toFloatArray()

        let cpu = SoftmaxAxis.referenceCPU(input, shape: shape, axis: axis)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], cpu[i], accuracy: 1e-5, "shape=\(shape) axis=\(axis) i=\(i)")
        }
        _ = input
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
