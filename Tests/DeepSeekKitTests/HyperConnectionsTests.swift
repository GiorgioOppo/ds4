import XCTest
import Metal
@testable import DeepSeekKit

final class HyperConnectionsTests: XCTestCase {

    func testPreMatchesCPU() throws {
        let N = 3, hc = 4, d = 8
        let mixHc = (2 + hc) * hc           // 24
        let hcD = hc * d                     // 32
        let normEps: Float = 1e-6
        let hcEps: Float = 1e-6
        let iters = 20

        let x = randomArray(N * hc * d, seed: 1, scale: 0.5)
        let hcFn = randomArray(mixHc * hcD, seed: 2, scale: 0.1)
        let hcScale: [Float] = [0.7, 0.9, 1.1]
        let hcBase = randomArray(mixHc, seed: 3, scale: 0.5)

        let xT = x.withUnsafeBytes { Tensor.from(bytes: $0, shape: [N, hc, d], dtype: .f32) }
        let hcFnT = hcFn.withUnsafeBytes { Tensor.from(bytes: $0, shape: [mixHc, hcD], dtype: .f32) }
        let hcScaleT = hcScale.withUnsafeBytes { Tensor.from(bytes: $0, shape: [3], dtype: .f32) }
        let hcBaseT = hcBase.withUnsafeBytes { Tensor.from(bytes: $0, shape: [mixHc], dtype: .f32) }

        var cfg = ModelConfig()
        cfg.hcMult = hc
        cfg.hcSinkhornIters = iters
        cfg.normEps = normEps
        cfg.hcEps = hcEps
        let hc_ = HyperConnections(config: cfg, dim: d)

        let cmd = Device.shared.queue.makeCommandBuffer()!
        let out = hc_.pre(x: xT, hcFn: hcFnT, hcScale: hcScaleT, hcBase: hcBaseT, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let gpuY = out.y.toFloatArray()
        let gpuPost = out.post.toFloatArray()
        let gpuComb = out.comb.toFloatArray()

        let cpu = HyperConnections.referencePreCPU(
            x: x, hcFn: hcFn, hcScale: hcScale, hcBase: hcBase,
            N: N, hc: hc, d: d,
            sinkhornIters: iters, normEps: normEps, hcEps: hcEps)

        for i in 0..<gpuY.count {
            XCTAssertEqual(gpuY[i], cpu.y[i], accuracy: 1e-3, "y[\(i)]")
        }
        for i in 0..<gpuPost.count {
            XCTAssertEqual(gpuPost[i], cpu.post[i], accuracy: 1e-4, "post[\(i)]")
        }
        for i in 0..<gpuComb.count {
            XCTAssertEqual(gpuComb[i], cpu.comb[i], accuracy: 1e-4, "comb[\(i)]")
        }
        _ = x; _ = hcFn; _ = hcScale; _ = hcBase
    }

    func testPostMatchesCPU() throws {
        let N = 3, hc = 4, d = 8
        let x = randomArray(N * d, seed: 11, scale: 0.5)
        let residual = randomArray(N * hc * d, seed: 12, scale: 0.5)
        // post / comb don't need to be Sinkhorn-valid for the contraction itself.
        let post = randomArray(N * hc, seed: 13, scale: 0.5)
        let comb = randomArray(N * hc * hc, seed: 14, scale: 0.2)

        let xT = x.withUnsafeBytes { Tensor.from(bytes: $0, shape: [N, d], dtype: .f32) }
        let residT = residual.withUnsafeBytes { Tensor.from(bytes: $0, shape: [N, hc, d], dtype: .f32) }
        let postT = post.withUnsafeBytes { Tensor.from(bytes: $0, shape: [N, hc], dtype: .f32) }
        let combT = comb.withUnsafeBytes { Tensor.from(bytes: $0, shape: [N, hc, hc], dtype: .f32) }

        var cfg = ModelConfig()
        cfg.hcMult = hc
        let hc_ = HyperConnections(config: cfg, dim: d)

        let cmd = Device.shared.queue.makeCommandBuffer()!
        let yT = hc_.post(x: xT, residual: residT, post: postT, comb: combT, in: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = yT.toFloatArray()

        let cpu = HyperConnections.referencePostCPU(
            x: x, residual: residual, post: post, comb: comb, N: N, hc: hc, d: d)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], cpu[i], accuracy: 1e-4, "y[\(i)]")
        }
        _ = x; _ = residual; _ = post; _ = comb
    }

    private func randomArray(_ count: Int, seed: UInt64, scale: Float = 1) -> [Float] {
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
