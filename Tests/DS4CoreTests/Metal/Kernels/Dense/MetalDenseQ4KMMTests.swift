import XCTest
import Foundation
import DS4Core
@testable import DS4Metal

/// Validates the dense Q4_K prefill GEMM (kernel_mul_mm_q4_K_f32, used by the
/// batched prefill when DS4_DENSE_Q4/DS4_QKV_Q4 requantized the projections)
/// against the validated Q4_K matvec over the same quantized weight — plain
/// token-major form and the strided grouped form (ne0-as-row-stride contract).
final class MetalDenseQ4KMMTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private var seed: UInt64 = 0x4B4B
    private func rf() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

    private func q4Weight(_ rt: MetalRuntime, rows: Int, cols: Int) throws -> GPUTensor {
        var w = [Float](repeating: 0, count: rows * cols)
        for i in 0..<w.count { w[i] = rf() * 0.1 }
        var packed = [UInt8](repeating: 0, count: rows * (cols / 256) * 144)
        w.withUnsafeBufferPointer { xp in
            packed.withUnsafeMutableBytes { pp in
                Quantize.quantizeQ4_K(xp.baseAddress!, count: w.count, into: pp.baseAddress!)
            }
        }
        return try GPUTensor.bytes(rt, packed, elementCount: rows * cols)
    }

    func testDenseQ4KMMMatchesMatvec() throws {
        let rt = try makeRuntime()
        let inDim = 512, outDim = 128, nTok = 13   // bc paths: outDim%64==0, nTok%32!=0
        let weight = try q4Weight(rt, rows: outDim, cols: inDim)
        let act = (0..<(nTok * inDim)).map { _ in rf() }
        let actT = try GPUTensor.floats(rt, act)
        let ref = try GPUTensor.zeros(rt, floatCount: nTok * outDim)
        let got = try GPUTensor.zeros(rt, floatCount: nTok * outDim)

        let c = GraphContext(rt); try c.begin()
        for t in 0..<nTok {
            try c.matmulQ4_K(weight: weight,
                             x: actT.subview(byteOffset: t * inDim * 4, byteLength: inDim * 4, count: inDim),
                             out: ref.subview(byteOffset: t * outDim * 4, byteLength: outDim * 4, count: outDim),
                             inDim: inDim, outDim: outDim)
        }
        try c.encodeMMDenseQ4K(weight: weight, act: actT, actBase: 0, out: got,
                               inDim: inDim, outDim: outDim, nTok: nTok)
        c.commit()
        XCTAssertNil(c.lastError)

        let rv = ref.floatArray(nTok * outDim), gv = got.floatArray(nTok * outDim)
        var maxRel: Float = 0
        for i in 0..<rv.count { maxRel = max(maxRel, abs(gv[i] - rv[i]) / max(abs(rv[i]), 0.05)) }
        XCTAssertLessThan(maxRel, 2e-2, "dense Q4_K mm vs matvec max rel \(maxRel)")
    }

    func testDenseQ4KMMStridedGroups() throws {
        let rt = try makeRuntime()
        let inDim = 512, outDim = 128, nTok = 9, nGroups = 2
        // Weight: two group slabs, contiguous (the attnOutA layout).
        let weight = try q4Weight(rt, rows: nGroups * outDim, cols: inDim)
        let groupRowBytes = (inDim / 256) * 144
        let act = (0..<(nTok * nGroups * inDim)).map { _ in rf() }   // [tok][group][inDim]
        let actT = try GPUTensor.floats(rt, act)
        let ref = try GPUTensor.zeros(rt, floatCount: nTok * nGroups * outDim)
        let got = try GPUTensor.zeros(rt, floatCount: nTok * nGroups * outDim)

        let c = GraphContext(rt); try c.begin()
        for t in 0..<nTok {
            for g in 0..<nGroups {
                let wView = weight.subview(byteOffset: g * outDim * groupRowBytes,
                                           byteLength: outDim * groupRowBytes,
                                           count: outDim * inDim)
                try c.matmulQ4_K(weight: wView,
                                 x: actT.subview(byteOffset: (t * nGroups + g) * inDim * 4,
                                                 byteLength: inDim * 4, count: inDim),
                                 out: ref.subview(byteOffset: (t * nGroups + g) * outDim * 4,
                                                  byteLength: outDim * 4, count: outDim),
                                 inDim: inDim, outDim: outDim)
            }
        }
        for g in 0..<nGroups {
            try c.encodeMMDenseQ4KStrided(weight: weight, weightOffset: g * outDim * groupRowBytes,
                                          act: actT, actBase: g * inDim * 4,
                                          actRowStride: nGroups * inDim * 4,
                                          out: got, outBase: g * outDim * 4,
                                          outRowStrideElems: nGroups * outDim,
                                          inDim: inDim, outDim: outDim, nTok: nTok)
        }
        c.commit()
        XCTAssertNil(c.lastError)

        let rv = ref.floatArray(nTok * nGroups * outDim), gv = got.floatArray(nTok * nGroups * outDim)
        var maxRel: Float = 0
        for i in 0..<rv.count { maxRel = max(maxRel, abs(gv[i] - rv[i]) / max(abs(rv[i]), 0.05)) }
        XCTAssertLessThan(maxRel, 2e-2, "strided Q4_K mm vs matvec max rel \(maxRel)")
    }
}
