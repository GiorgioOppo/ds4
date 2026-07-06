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

    /// The FUSED HC-reduce tail (kernel_dsv4_hc_split_weighted_sum_norm4, the
    /// C decode release path) must reproduce the unfused triple — split +
    /// weighted sum + weighted RMSNorm — on all three outputs. The split and
    /// collapsed row are the same math in the same order (exact); the norm
    /// differs only in reduction order (±ulp), hence the tight rel tolerance.
    func testFusedHCReduceMatchesUnfused() throws {
        let rt = try makeRuntime()
        // The test compiles the USER's vendored metal dir (not the embedded
        // source): skip, don't fail, if that copy predates the fused kernel.
        let hcSrc = (try? String(contentsOfFile: Self.metalDir + "/dsv4_hc.metal", encoding: .utf8)) ?? ""
        try XCTSkipUnless(hcSrc.contains("kernel_dsv4_hc_split_weighted_sum_norm4"),
                          "vendored dsv4_hc.metal predates the fused HC kernel")
        var seed: UInt64 = 0xFACC1A
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let nEmbd = 512, nHC = 4, iters = 20
        let hcEps: Float = 1e-3, rmsEps: Float = 1e-6
        let scale: [Float] = [0.8, 1.0, 0.9]
        var mix = [Float](repeating: 0, count: 24)
        var base = [Float](repeating: 0, count: 24)
        var residual = [Float](repeating: 0, count: nHC * nEmbd)
        var normW = [Float](repeating: 0, count: nEmbd)
        for i in 0..<24 { mix[i] = rndF(); base[i] = rndF() }
        for i in 0..<residual.count { residual[i] = rndF() }
        for i in 0..<nEmbd { normW[i] = 1 + 0.1 * rndF() }

        let mixT = try GPUTensor.floats(rt, mix)
        let scT = try GPUTensor.floats(rt, scale)
        let baseT = try GPUTensor.floats(rt, base)
        let resT = try GPUTensor.floats(rt, residual)
        let normWT = try GPUTensor.floats(rt, normW)

        // Unfused reference: split -> weighted sum -> weighted RMSNorm.
        let splitU = try GPUTensor.zeros(rt, floatCount: 24)
        let embdU = try GPUTensor.zeros(rt, floatCount: nEmbd)
        let curU = try GPUTensor.zeros(rt, floatCount: nEmbd)
        let c1 = GraphContext(rt)
        try c1.begin()
        try c1.hcSplitSinkhorn(mix: mixT, scale: scT, base: baseT, out: splitU, nRows: 1, sinkhornIters: iters, eps: hcEps)
        try c1.hcWeightedSum(x: resT, weights: splitU, out: embdU, nEmbd: nEmbd, nHC: nHC, nTokens: 1)
        try c1.rmsNorm(embdU, weight: normWT, out: curU, rows: 1, n: nEmbd, eps: rmsEps)
        c1.commit()

        // Fused: one dispatch, three outputs.
        let splitF = try GPUTensor.zeros(rt, floatCount: 24)
        let embdF = try GPUTensor.zeros(rt, floatCount: nEmbd)
        let curF = try GPUTensor.zeros(rt, floatCount: nEmbd)
        let c2 = GraphContext(rt)
        try c2.begin()
        try c2.hcSplitWeightedSumNorm4(mix: mixT, scale: scT, base: baseT, x: resT,
                                       split: splitF, embd: embdF, normWeight: normWT,
                                       normOut: curF, nEmbd: nEmbd, nRows: 1,
                                       sinkhornIters: iters, eps: hcEps, normEps: rmsEps)
        c2.commit()

        // Same math, but two different kernels: the compiler may contract FMAs
        // differently, so compare with a ulp-class relative tolerance rather
        // than bit equality (the MoE packed test compares the SAME kernel).
        func maxRel(_ a: [Float], _ b: [Float], floor: Float) -> Float {
            var m: Float = 0
            for i in 0..<a.count { m = max(m, abs(a[i]-b[i])/max(abs(b[i]), floor)) }
            return m
        }
        // Floors sized to the data (split in (0,1); embd/cur terms O(1-20)) so
        // FMA-contraction noise on a near-zero sum can't blow up the ratio: a
        // real bug (wrong stride/eps/field offset) shows as O(1) error, orders
        // above these thresholds.
        let relSplit = maxRel(splitF.floatArray(24), splitU.floatArray(24), floor: 1e-3)
        XCTAssertLessThan(relSplit, 1e-5, "split fused vs unfused max rel \(relSplit)")
        let relEmbd = maxRel(embdF.floatArray(nEmbd), embdU.floatArray(nEmbd), floor: 0.1)
        XCTAssertLessThan(relEmbd, 1e-4, "embd fused vs unfused max rel \(relEmbd)")
        let relCur = maxRel(curF.floatArray(nEmbd), curU.floatArray(nEmbd), floor: 0.1)
        XCTAssertLessThan(relCur, 1e-4, "normed row fused vs unfused max rel \(relCur)")
    }
}
