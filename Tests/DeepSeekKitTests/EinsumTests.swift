import XCTest
import Metal
@testable import DeepSeekKit

final class EinsumTests: XCTestCase {

    func testBshdBtd() throws {
        let B = 2, S = 3, H = 4, D = 8, T = 5
        let q = randomArray(B * S * H * D, seed: 1)
        let kv = randomArray(B * T * D, seed: 2)
        let qT = q.withUnsafeBytes { Tensor.from(bytes: $0, shape: [B, S, H, D], dtype: .f32) }
        let kvT = kv.withUnsafeBytes { Tensor.from(bytes: $0, shape: [B, T, D], dtype: .f32) }

        let cmd = Device.shared.queue.makeCommandBuffer()!
        let outT = Einsum.bshdBtd(q: qT, kv: kvT, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = outT.toFloatArray()

        let cpu = Einsum.referenceBshdBtdCPU(q: q, kv: kv, B: B, S: S, H: H, D: D, T: T)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], cpu[i], accuracy: 1e-4, "i=\(i)")
        }
        _ = q; _ = kv
    }

    func testBsgdGrd() throws {
        let B = 1, S = 4, G = 2, D = 6, R = 3
        let o = randomArray(B * S * G * D, seed: 11)
        let woA = randomArray(G * R * D, seed: 12)
        let oT = o.withUnsafeBytes { Tensor.from(bytes: $0, shape: [B, S, G, D], dtype: .f32) }
        let woAT = woA.withUnsafeBytes { Tensor.from(bytes: $0, shape: [G, R, D], dtype: .f32) }

        let cmd = Device.shared.queue.makeCommandBuffer()!
        let outT = Einsum.bsgdGrd(o: oT, woA: woAT, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = outT.toFloatArray()

        let cpu = Einsum.referenceBsgdGrdCPU(o: o, woA: woA, B: B, S: S, G: G, D: D, R: R)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], cpu[i], accuracy: 1e-4, "i=\(i)")
        }
        _ = o; _ = woA
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
