import XCTest
import Foundation
@testable import DS4Metal

/// Stage C: validates the HC-reduce block (collapse the 4 HC streams to one
/// embedding row): hcSplitSinkhorn produces the pre weights, hcWeightedSum
/// collapses residual with them — chained in one command buffer, vs CPU.
final class GraphHCReduceTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_hc.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testHCReduceChain() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xC0DE5
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let nEmbd = 512, nHC = 4, iters = 3
        let eps: Float = 1e-3
        let scale: [Float] = [0.8, 1.0, 0.9]
        var mix = [Float](repeating: 0, count: 24)
        var base = [Float](repeating: 0, count: 24)
        var residual = [Float](repeating: 0, count: nHC * nEmbd)
        for i in 0..<24 { mix[i] = rndF(); base[i] = rndF() }
        for i in 0..<residual.count { residual[i] = rndF() }

        let ctx = GraphContext(rt)
        let mixT = try GPUTensor.floats(rt, mix)
        let scT = try GPUTensor.floats(rt, scale)
        let baseT = try GPUTensor.floats(rt, base)
        let split = try GPUTensor.zeros(rt, floatCount: 24)
        let resT = try GPUTensor.floats(rt, residual)
        let outT = try GPUTensor.zeros(rt, floatCount: nEmbd)

        try ctx.begin()
        try ctx.hcSplitSinkhorn(mix: mixT, scale: scT, base: baseT, out: split, nRows: 1, sinkhornIters: iters, eps: eps)
        try ctx.hcWeightedSum(x: resT, weights: split, out: outT, nEmbd: nEmbd, nHC: nHC, nTokens: 1)
        ctx.commit()

        // CPU: pre weights = sigmoid(mix[h]*scale[0]+base[h]) + eps; out[d] = sum_h pre[h]*residual[h][d]
        var pre = [Float](repeating: 0, count: nHC)
        for h in 0..<nHC { let z = mix[h]*scale[0]+base[h]; pre[h] = 1/(1+expf(-z)) + eps }
        let got = outT.floatArray(nEmbd)
        var maxRel: Float = 0
        for d in 0..<nEmbd {
            var acc: Float = 0
            for h in 0..<nHC { acc += residual[h*nEmbd+d] * pre[h] }
            maxRel = max(maxRel, abs(got[d]-acc)/max(abs(acc),0.1))
        }
        XCTAssertLessThan(maxRel, 2e-3, "HC reduce chain max rel \(maxRel)")
    }
}
