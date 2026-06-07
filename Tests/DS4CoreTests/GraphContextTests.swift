import XCTest
import Foundation
@testable import DS4Metal

/// Stage B: validates the GPUTensor + GraphContext tensor-op layer — chained
/// dispatches in ONE command buffer (rmsNorm -> matmulF32, and swiglu) — produce
/// the same result as a CPU reference, proving resident-buffer chaining works.
final class GraphContextTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/norm.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testChainedRMSNormMatmul() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xB001
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 29) }

        let inDim = 1024, outDim = 300
        let eps: Float = 1e-5
        var x = [Float](repeating: 0, count: inDim)
        var w = [Float](repeating: 0, count: inDim)        // rmsnorm weight
        var wm = [Float](repeating: 0, count: inDim * outDim) // matmul weight
        for i in 0..<inDim { x[i] = rndF(); w[i] = rndF() }
        for i in 0..<wm.count { wm[i] = rndF() }

        let ctx = GraphContext(rt)
        let xt = try GPUTensor.floats(rt, x)
        let wt = try GPUTensor.floats(rt, w)
        let normed = try GPUTensor.zeros(rt, floatCount: inDim)
        let wmT = try GPUTensor.floats(rt, wm)
        let outT = try GPUTensor.zeros(rt, floatCount: outDim)

        try ctx.begin()
        try ctx.rmsNorm(xt, weight: wt, out: normed, rows: 1, n: inDim, eps: eps)
        try ctx.matmulF32(weight: wmT, x: normed, out: outT, inDim: inDim, outDim: outDim)
        ctx.commit()

        // CPU reference
        var ss: Float = 0
        for i in 0..<inDim { ss += x[i] * x[i] }
        let scale = 1.0 / (ss / Float(inDim) + eps).squareRoot()
        var nrm = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim { nrm[i] = x[i] * scale * w[i] }
        var ref = [Float](repeating: 0, count: outDim)
        for r in 0..<outDim {
            var acc: Float = 0
            for i in 0..<inDim { acc += wm[r*inDim+i] * nrm[i] }
            ref[r] = acc
        }
        let got = outT.floatArray(outDim)
        var maxRel: Float = 0
        for r in 0..<outDim { maxRel = max(maxRel, abs(got[r]-ref[r]) / max(abs(ref[r]),1)) }
        XCTAssertLessThan(maxRel, 2e-3, "chained rmsnorm->matmul max rel \(maxRel)")
    }

    func testChainedSwiGLU() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xB002
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 29) }
        let n = 512
        var g = [Float](repeating: 0, count: n), u = [Float](repeating: 0, count: n)
        for i in 0..<n { g[i] = rndF(); u[i] = rndF() }

        let ctx = GraphContext(rt)
        let gt = try GPUTensor.floats(rt, g), ut = try GPUTensor.floats(rt, u)
        let ot = try GPUTensor.zeros(rt, floatCount: n)
        try ctx.begin()
        try ctx.swiglu(gate: gt, up: ut, out: ot, n: n)
        ctx.commit()

        let got = ot.floatArray(n)
        for i in 0..<n {
            let ref = (g[i] / (1 + expf(-g[i]))) * u[i]
            XCTAssertEqual(got[i], ref, accuracy: max(abs(ref),1)*1e-4, "swiglu \(i)")
        }
    }
}
