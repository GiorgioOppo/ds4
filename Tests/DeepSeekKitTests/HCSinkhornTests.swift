import XCTest
import Metal
@testable import DeepSeekKit

final class HCSinkhornTests: XCTestCase {

    func testSinkhornMatchesReferenceCPU() throws {
        let hc = 4
        let mixHc = (2 + hc) * hc        // 24
        let n = 5
        let iters = 20
        let eps: Float = 1e-6

        let mixesArr = randomArray(count: n * mixHc, seed: 11)
        let scaleArr: [Float] = [0.7, 0.9, 1.1]
        let baseArr = randomArray(count: mixHc, seed: 23)

        let mixes = mixesArr.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [n, mixHc], dtype: .f32)
        }
        let hcScale = scaleArr.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [3], dtype: .f32)
        }
        let hcBase = baseArr.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [mixHc], dtype: .f32)
        }

        let mod = HCSinkhorn(hcMult: hc, sinkhornIters: iters, hcEps: eps)
        let cmd = Device.shared.queue.makeCommandBuffer()!
        let out = mod.split(mixes: mixes, hcScale: hcScale, hcBase: hcBase, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpuPre = out.pre.toFloatArray()
        let gpuPost = out.post.toFloatArray()
        let gpuComb = out.comb.toFloatArray()

        for r in 0..<n {
            let mixesRow = Array(mixesArr[r * mixHc ..< (r + 1) * mixHc])
            let ref = mod.referenceCPU(mixes: mixesRow, hcScale: scaleArr, hcBase: baseArr)
            for j in 0..<hc {
                XCTAssertEqual(gpuPre[r * hc + j], ref.pre[j], accuracy: 1e-4)
                XCTAssertEqual(gpuPost[r * hc + j], ref.post[j], accuracy: 1e-4)
                for k in 0..<hc {
                    XCTAssertEqual(gpuComb[(r * hc + j) * hc + k], ref.comb[j][k],
                                   accuracy: 1e-4, "row=\(r) j=\(j) k=\(k)")
                }
            }
        }
        _ = mixesArr; _ = scaleArr; _ = baseArr
    }

    /// After 20 Sinkhorn iters, comb is approximately doubly stochastic
    /// (row sums ≈ 1, col sums ≈ 1) up to the additive eps drift.
    func testCombApproximatelyDoublyStochastic() throws {
        let hc = 4
        let mixHc = (2 + hc) * hc
        let n = 8
        let mod = HCSinkhorn(hcMult: hc, sinkhornIters: 20, hcEps: 1e-6)

        let mixesArr = randomArray(count: n * mixHc, seed: 99)
        let scaleArr: [Float] = [1, 1, 1]
        let baseArr = [Float](repeating: 0, count: mixHc)

        let mixes = mixesArr.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [n, mixHc], dtype: .f32)
        }
        let hcScale = scaleArr.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [3], dtype: .f32)
        }
        let hcBase = baseArr.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [mixHc], dtype: .f32)
        }

        let cmd = Device.shared.queue.makeCommandBuffer()!
        let out = mod.split(mixes: mixes, hcScale: hcScale, hcBase: hcBase, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let comb = out.comb.toFloatArray()

        for r in 0..<n {
            for j in 0..<hc {
                var rowSum: Float = 0
                for k in 0..<hc { rowSum += comb[(r * hc + j) * hc + k] }
                XCTAssertEqual(rowSum, 1.0, accuracy: 1e-3, "row sum r=\(r) j=\(j)")
            }
            for k in 0..<hc {
                var colSum: Float = 0
                for j in 0..<hc { colSum += comb[(r * hc + j) * hc + k] }
                XCTAssertEqual(colSum, 1.0, accuracy: 1e-3, "col sum r=\(r) k=\(k)")
            }
        }
        _ = mixesArr; _ = scaleArr; _ = baseArr
    }

    private func randomArray(count: Int, seed: UInt64) -> [Float] {
        var state = seed | 1
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let frac = Float(Double(state >> 11) / Double(1 << 53))
            out[i] = (frac - 0.5) * 2
        }
        return out
    }
}
