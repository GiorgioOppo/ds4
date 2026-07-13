import XCTest
import Foundation
@testable import DS4Metal

/// Stage B: validates the GPUTensor + GraphContext tensor-op layer — chained
/// dispatches in ONE command buffer (rmsNorm -> matmulF32, and swiglu) — produce
/// the same result as a CPU reference, proving resident-buffer chaining works.
final class GraphContextTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testTensorSubviewsShareStorageAndHonorOffsets() throws {
        let rt = try makeRuntime()
        let slab = try GPUTensor.zeros(rt, floatCount: 12)
        let middle = slab.subview(byteOffset: 4 * 4, byteLength: 4 * 4, count: 4)
        let values: [Float] = [11, 12, 13, 14]
        _ = values.withUnsafeBytes {
            memcpy(middle.buffer.contents() + middle.byteOffset, $0.baseAddress!, $0.count)
        }

        XCTAssertEqual(middle.floatArray(), values)
        XCTAssertEqual(slab.floatArray(), [0, 0, 0, 0, 11, 12, 13, 14, 0, 0, 0, 0])

        middle.zero()
        XCTAssertEqual(slab.floatArray(), [Float](repeating: 0, count: 12))
    }

    /// Regression for the contiguous prefill slabs: a logical row is a view
    /// into one MTLBuffer, so every compute binding must add GPUTensor.byteOffset.
    /// Binding at zero makes every prompt token read row 0 and destroys quality.
    func testMatmulReadsAndWritesSubviewOffsets() throws {
        let rt = try makeRuntime()
        let inDim = 1024, outDim = 2
        let inputSlab = try GPUTensor.zeros(rt, floatCount: inDim * 2)
        let input = inputSlab.subview(byteOffset: inDim * 4,
                                      byteLength: inDim * 4, count: inDim)
        var values = [Float](repeating: 0, count: inDim)
        values[0] = 3; values[1] = -5
        _ = values.withUnsafeBytes {
            memcpy(input.buffer.contents() + input.byteOffset, $0.baseAddress!, $0.count)
        }
        var weights = [Float](repeating: 0, count: inDim * outDim)
        weights[0] = 2
        weights[inDim + 1] = 4
        let weight = try GPUTensor.floats(rt, weights)
        let outputSlab = try GPUTensor.zeros(rt, floatCount: outDim * 2)
        let output = outputSlab.subview(byteOffset: outDim * 4,
                                        byteLength: outDim * 4, count: outDim)

        let ctx = GraphContext(rt)
        try ctx.begin()
        try ctx.matmulF32(weight: weight, x: input, out: output,
                          inDim: inDim, outDim: outDim)
        ctx.commit()

        XCTAssertEqual(outputSlab.floatArray(), [0, 0, 6, -20])
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
