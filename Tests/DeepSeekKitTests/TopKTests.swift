import XCTest
import Metal
@testable import DeepSeekKit

final class TopKTests: XCTestCase {

    func testMatchesCPU() throws {
        let N = 4, V = 32, k = 5
        let scores = randomArray(N * V, seed: 1)
        let t = scores.withUnsafeBytes { Tensor.from(bytes: $0, shape: [N, V], dtype: .f32) }

        let cmd = Device.shared.queue.makeCommandBuffer()!
        let out = TopK.apply(t, k: k, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let gpuV = out.values.toFloatArray()
        let gpuIPtr = out.indices.buffer.contents().bindMemory(to: Int32.self, capacity: N * k)
        let gpuI = Array(UnsafeBufferPointer(start: gpuIPtr, count: N * k))

        let cpu = TopK.referenceCPU(scores, N: N, V: V, k: k)
        for i in 0..<gpuV.count {
            XCTAssertEqual(gpuV[i], cpu.values[i], accuracy: 1e-5, "values[\(i)]")
            XCTAssertEqual(gpuI[i], cpu.indices[i], "indices[\(i)]")
        }
        _ = scores
    }

    /// k > 32 routes to the bitonic kernel. Compares against the CPU
    /// reference on a deterministic seed; ties on float values can flip
    /// indices, so this picks a width with distinct draws.
    func testLargeKMatchesCPU() throws {
        let N = 3, V = 256, k = 128
        let scores = randomArray(N * V, seed: 7)
        let t = scores.withUnsafeBytes { Tensor.from(bytes: $0, shape: [N, V], dtype: .f32) }

        let cmd = Device.shared.queue.makeCommandBuffer()!
        let out = TopK.apply(t, k: k, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let gpuV = out.values.toFloatArray()
        let gpuIPtr = out.indices.buffer.contents().bindMemory(to: Int32.self, capacity: N * k)
        let gpuI = Array(UnsafeBufferPointer(start: gpuIPtr, count: N * k))

        let cpu = TopK.referenceCPU(scores, N: N, V: V, k: k)
        for i in 0..<gpuV.count {
            XCTAssertEqual(gpuV[i], cpu.values[i], accuracy: 1e-5, "values[\(i)]")
            XCTAssertEqual(gpuI[i], cpu.indices[i], "indices[\(i)]")
        }
        _ = scores
    }

    /// V not a power of two — bitonic kernel must pad internally with
    /// -INF so the padding never enters the top-K window.
    func testLargeKWithNonPow2V() throws {
        let N = 2, V = 300, k = 64
        let scores = randomArray(N * V, seed: 11)
        let t = scores.withUnsafeBytes { Tensor.from(bytes: $0, shape: [N, V], dtype: .f32) }

        let cmd = Device.shared.queue.makeCommandBuffer()!
        let out = TopK.apply(t, k: k, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let gpuV = out.values.toFloatArray()
        let gpuIPtr = out.indices.buffer.contents().bindMemory(to: Int32.self, capacity: N * k)
        let gpuI = Array(UnsafeBufferPointer(start: gpuIPtr, count: N * k))

        let cpu = TopK.referenceCPU(scores, N: N, V: V, k: k)
        for i in 0..<gpuV.count {
            XCTAssertEqual(gpuV[i], cpu.values[i], accuracy: 1e-5, "values[\(i)]")
            XCTAssertEqual(gpuI[i], cpu.indices[i], "indices[\(i)]")
        }
        _ = scores
    }

    /// k = 1 → just argmax.
    func testKEquals1() throws {
        let N = 3, V = 16
        let scores = randomArray(N * V, seed: 2)
        let t = scores.withUnsafeBytes { Tensor.from(bytes: $0, shape: [N, V], dtype: .f32) }

        let cmd = Device.shared.queue.makeCommandBuffer()!
        let out = TopK.apply(t, k: 1, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpuI = Array(UnsafeBufferPointer(
            start: out.indices.buffer.contents().bindMemory(to: Int32.self, capacity: N),
            count: N))

        for n in 0..<N {
            var maxV = -Float.infinity; var maxI = 0
            for v in 0..<V {
                if scores[n * V + v] > maxV { maxV = scores[n * V + v]; maxI = v }
            }
            XCTAssertEqual(Int(gpuI[n]), maxI, "row=\(n)")
        }
        _ = scores
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
