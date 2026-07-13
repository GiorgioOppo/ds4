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

    func testFusedRouterProbabilitiesMatchTwoPass() throws {
        let rt = try makeRuntime()
        var logits = [Float](repeating: 0, count: 256)
        for i in logits.indices {
            // Covers the softplus branch boundary and normal router range.
            logits[i] = Float(i - 128) * 0.1875
        }
        let src = try GPUTensor.floats(rt, logits)
        let softplus = try GPUTensor.zeros(rt, floatCount: 256)
        let reference = try GPUTensor.zeros(rt, floatCount: 256)
        let fused = try GPUTensor.zeros(rt, floatCount: 256)
        let ctx = GraphContext(rt)
        try ctx.begin()
        try ctx.unary(src, op: .softplus, out: softplus, width: 256)
        try ctx.unary(softplus, op: .sqrt, out: reference, width: 256)
        try ctx.routerProbabilities(logits: src, probabilities: fused, width: 256)
        ctx.commit()

        let a = reference.floatArray(), b = fused.floatArray()
        XCTAssertEqual(a.count, b.count)
        for i in a.indices {
            XCTAssertEqual(a[i].bitPattern, b[i].bitPattern, "router probability \(i)")
        }
    }

    func testDenseQ4KernelMatchesSingleExpertWrapper() throws {
        let rt = try makeRuntime()
        let inDim = 512, outDim = 128, blocksPerRow = inDim / 256
        var seed: UInt64 = 0xD345_EA5E
        func nextByte() -> UInt8 {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return UInt8(truncatingIfNeeded: seed >> 40)
        }
        func nextFloat() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30)
        }

        var q4 = [UInt8](repeating: 0, count: outDim * blocksPerRow * 144)
        var off = 0
        for _ in 0..<(outDim * blocksPerRow) {
            let d = Float16(abs(nextFloat()) * 0.05)
            let dmin = Float16(abs(nextFloat()) * 0.02)
            withUnsafeBytes(of: d.bitPattern.littleEndian) {
                q4[off] = $0[0]; q4[off + 1] = $0[1]
            }
            withUnsafeBytes(of: dmin.bitPattern.littleEndian) {
                q4[off + 2] = $0[0]; q4[off + 3] = $0[1]
            }
            for i in 4..<144 { q4[off + i] = nextByte() }
            off += 144
        }
        var activation = [Float](repeating: 0, count: inDim)
        for i in activation.indices { activation[i] = nextFloat() }

        let weights = try GPUTensor.bytes(rt, q4, elementCount: inDim * outDim)
        let x = try GPUTensor.floats(rt, activation)
        let id0 = try GPUTensor.bytes(rt, [0, 0, 0, 0], elementCount: 1)
        let reference = try GPUTensor.zeros(rt, floatCount: outDim)
        let dense = try GPUTensor.zeros(rt, floatCount: outDim)
        let ctx = GraphContext(rt)
        try ctx.begin()
        try ctx.moeMatvecID(.q4_K, experts: weights, ids: id0,
                            activation: x, out: reference, k: 1,
                            inDim: inDim, outDim: outDim, perExpertAct: false)
        try ctx.matmulQ4_K(weight: weights, x: x, out: dense,
                           inDim: inDim, outDim: outDim)
        ctx.commit()

        let a = reference.floatArray(), b = dense.floatArray()
        XCTAssertEqual(a.count, b.count)
        for i in a.indices {
            XCTAssertEqual(a[i].bitPattern, b[i].bitPattern, "dense Q4 row \(i)")
        }
    }

    func testGroupedAttentionQ4KernelMatchesIdentityMoEWrapper() throws {
        let rt = try makeRuntime()
        let groups = 4, inDim = 512, outDim = 64
        let rowBytes = (inDim / 256) * 144
        var seed: UInt64 = 0xA770_04A4
        func byte() -> UInt8 {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return UInt8(truncatingIfNeeded: seed >> 40)
        }
        func value() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30)
        }
        var q4 = [UInt8](repeating: 0, count: groups * outDim * rowBytes)
        var off = 0
        for _ in 0..<(groups * outDim * inDim / 256) {
            let d = Float16(abs(value()) * 0.05), dm = Float16(abs(value()) * 0.02)
            withUnsafeBytes(of: d.bitPattern.littleEndian) { q4[off] = $0[0]; q4[off + 1] = $0[1] }
            withUnsafeBytes(of: dm.bitPattern.littleEndian) { q4[off + 2] = $0[0]; q4[off + 3] = $0[1] }
            for i in 4..<144 { q4[off + i] = byte() }
            off += 144
        }
        var x = [Float](repeating: 0, count: groups * inDim)
        for i in x.indices { x[i] = value() }
        let weights = try GPUTensor.bytes(rt, q4, elementCount: groups * outDim * inDim)
        let activation = try GPUTensor.floats(rt, x)
        let ids = try GPUTensor.bytes(rt, Array(0..<Int32(groups)).withUnsafeBytes { Array($0) },
                                      elementCount: groups)
        let reference = try GPUTensor.zeros(rt, floatCount: groups * outDim)
        let grouped = try GPUTensor.zeros(rt, floatCount: groups * outDim)
        let ctx = GraphContext(rt)
        try ctx.begin()
        try ctx.moeMatvecID(.q4_K, experts: weights, ids: ids, activation: activation,
                            out: reference, k: groups, inDim: inDim, outDim: outDim,
                            perExpertAct: true)
        try ctx.attnOutLowQ4(outputA: weights, heads: activation, low: grouped,
                             nGroups: groups, groupDim: inDim, rank: outDim)
        ctx.commit()
        let a = reference.floatArray(), b = grouped.floatArray()
        for i in a.indices {
            XCTAssertEqual(a[i].bitPattern, b[i].bitPattern, "grouped Q4 output \(i)")
        }
    }

    func testPairedF16MatvecMatchesTwoDispatchesBitExactly() throws {
        let rt = try makeRuntime()
        let inDim = 4096, outDim = 512
        var seed: UInt64 = 0xC04F_16A2
        func value() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 31)
        }
        var x = [Float](repeating: 0, count: inDim)
        var a = [UInt16](repeating: 0, count: inDim * outDim)
        var b = [UInt16](repeating: 0, count: inDim * outDim)
        for i in x.indices { x[i] = value() }
        for i in a.indices {
            a[i] = Float16(value()).bitPattern
            b[i] = Float16(value()).bitPattern
        }
        let xt = try GPUTensor.floats(rt, x)
        let wa = try GPUTensor.bytes(rt, a.withUnsafeBytes { Array($0) }, elementCount: a.count)
        let wb = try GPUTensor.bytes(rt, b.withUnsafeBytes { Array($0) }, elementCount: b.count)
        let refA = try GPUTensor.zeros(rt, floatCount: outDim)
        let refB = try GPUTensor.zeros(rt, floatCount: outDim)
        let pairA = try GPUTensor.zeros(rt, floatCount: outDim)
        let pairB = try GPUTensor.zeros(rt, floatCount: outDim)
        let ctx = GraphContext(rt)
        try ctx.begin()
        try ctx.matmulF16(weight: wa, x: xt, out: refA, inDim: inDim, outDim: outDim)
        try ctx.matmulF16(weight: wb, x: xt, out: refB, inDim: inDim, outDim: outDim)
        try ctx.matmulF16Pair(weightA: wa, weightB: wb, x: xt,
                              outA: pairA, outB: pairB, inDim: inDim, outDim: outDim)
        ctx.commit()
        for (reference, paired) in [(refA.floatArray(), pairA.floatArray()),
                                    (refB.floatArray(), pairB.floatArray())] {
            for i in reference.indices {
                XCTAssertEqual(reference[i].bitPattern, paired[i].bitPattern,
                               "paired F16 row \(i)")
            }
        }
    }
}
