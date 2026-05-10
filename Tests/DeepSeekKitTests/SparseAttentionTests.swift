import XCTest
import Metal
@testable import DeepSeekKit

final class SparseAttentionTests: XCTestCase {

    func testMatchesCPUReference() throws {
        let B = 1, M = 2, N = 16, H = 4, D = 8, K = 5
        let q = randomArray(B*M*H*D, seed: 1, scale: 0.5)
        let kv = randomArray(B*N*D, seed: 2, scale: 0.5)
        let sink = randomArray(H, seed: 3, scale: 0.1)
        var tk = [Int32](repeating: 0, count: B*M*K)
        var rng = MiniRNG(seed: 42)
        for i in 0..<tk.count { tk[i] = Int32(rng.next() % UInt64(N)) }
        // Sprinkle some -1 padding to test the gather mask path.
        tk[1] = -1
        tk[K + 2] = -1

        let qT = q.withUnsafeBytes { Tensor.from(bytes: $0, shape: [B, M, H, D], dtype: .f32) }
        let kvT = kv.withUnsafeBytes { Tensor.from(bytes: $0, shape: [B, N, D], dtype: .f32) }
        let sinkT = sink.withUnsafeBytes { Tensor.from(bytes: $0, shape: [H], dtype: .f32) }
        let tkT = tk.withUnsafeBytes { Tensor.from(bytes: $0, shape: [B, M, K], dtype: .i32) }

        let scale: Float = pow(Float(D), -0.5)
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let o = SparseAttention.apply(q: qT, kv: kvT, sink: sinkT,
                                      topkIdxs: tkT, scale: scale, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = o.toFloatArray()

        let cpu = SparseAttention.referenceCPU(q: q, kv: kv, sink: sink, tk: tk,
                                                B: B, M: M, N: N, H: H, D: D, K: K,
                                                scale: scale)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], cpu[i], accuracy: 1e-4, "i=\(i)")
        }
        _ = q; _ = kv; _ = sink; _ = tk
    }

    /// All-padding indices: output should be 0 (sink-only denominator, no
    /// numerator contributions).
    func testAllPaddingProducesZero() throws {
        let B = 1, M = 1, N = 8, H = 2, D = 4, K = 3
        let q = randomArray(B*M*H*D, seed: 7)
        let kv = randomArray(B*N*D, seed: 8)
        let sink = [Float](repeating: 0, count: H)
        let tk = [Int32](repeating: -1, count: B*M*K)

        let qT = q.withUnsafeBytes { Tensor.from(bytes: $0, shape: [B, M, H, D], dtype: .f32) }
        let kvT = kv.withUnsafeBytes { Tensor.from(bytes: $0, shape: [B, N, D], dtype: .f32) }
        let sinkT = sink.withUnsafeBytes { Tensor.from(bytes: $0, shape: [H], dtype: .f32) }
        let tkT = tk.withUnsafeBytes { Tensor.from(bytes: $0, shape: [B, M, K], dtype: .i32) }

        let cmd = Device.shared.queue.makeCommandBuffer()!
        let o = SparseAttention.apply(q: qT, kv: kvT, sink: sinkT,
                                      topkIdxs: tkT, scale: 1.0, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = o.toFloatArray()
        for v in gpu { XCTAssertEqual(v, 0.0, accuracy: 1e-6) }
        _ = q; _ = kv; _ = sink; _ = tk
    }

    private func randomArray(_ count: Int, seed: UInt64, scale: Float = 1) -> [Float] {
        var rng = MiniRNG(seed: seed)
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let frac = Float(Double(rng.next() >> 11) / Double(1 << 53))
            out[i] = (frac - 0.5) * 2 * scale
        }
        return out
    }
}

private struct MiniRNG {
    var state: UInt64
    init(seed: UInt64) { self.state = seed | 1 }
    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}
