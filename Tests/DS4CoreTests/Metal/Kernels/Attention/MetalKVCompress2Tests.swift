import XCTest
import Foundation
@testable import DS4Metal

/// Phase 9 / Stage A4 (part 2): validates the remaining dsv4_kv.metal kernels
/// (hadamard_fp4, kv_fp8_store, compressor_store_one) vs CPU references.
final class MetalKVCompress2Tests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_kv.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testHadamardFP4() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x4D41
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let nRows = 5, n = 128
        var x = [Float](repeating: 0, count: nRows * n)
        for i in 0..<x.count { x[i] = rndF() }

        let gpu = try rt.indexerHadamardFP4(x, nRows: nRows)
        XCTAssertEqual(gpu.count, nRows * n)

        // CPU reference: in-place FWHT butterfly (same index pattern), scale, per-32 amax, E2M1.
        for r in 0..<nRows {
            var v = Array(x[r*n..<(r+1)*n])
            var stride = 1
            while stride < 128 {
                for tid in 0..<128 where (tid & stride) == 0 {
                    let base = (tid & ~(2*stride - 1)) + (tid & (stride - 1))
                    let a = v[base], b = v[base + stride]
                    v[base] = a + b; v[base + stride] = a - b
                }
                stride <<= 1
            }
            var rotated = [Float](repeating: 0, count: 128)
            for i in 0..<128 { rotated[i] = v[i] * 0.08838834764831845 }
            for block in 0..<4 {
                var amax: Float = 0
                for l in 0..<32 { amax = max(amax, abs(rotated[block*32+l])) }
                amax = max(amax, 7.052966104933725e-38)
                let scale = exp2((log2(amax / 6.0)).rounded(.up))
                for l in 0..<32 {
                    let idx = block*32+l
                    let q = MetalRuntime.e2m1Dequant(min(max(rotated[idx]/scale, -6), 6)) * scale
                    XCTAssertEqual(gpu[r*n+idx], q, accuracy: max(abs(q),1)*1e-4, "hadamard r=\(r) i=\(idx)")
                }
            }
        }
    }

    func testKVFP8Store() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x4D42
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 26) }

        let headDim = 512, nRot = 64, rawCap = 8, rawRow = 3
        var kv = [Float](repeating: 0, count: headDim)
        for i in 0..<headDim { kv[i] = rndF() }
        let rawCache = [Float](repeating: 0, count: rawCap * headDim)

        let (gkv, graw) = try rt.kvFP8Store(kv: kv, rawCache: rawCache, headDim: headDim, nRot: nRot, rawRow: rawRow, rawCap: rawCap)

        let nNope = headDim - nRot
        var refKv = kv
        var off = 0
        while off < nNope {
            var amax: Float = 0
            for j in 0..<64 where off+j < nNope { amax = max(amax, abs(kv[off+j])) }
            amax = max(amax, 1e-4)
            let scale = exp2((log2(amax / 448.0)).rounded(.up))
            for j in 0..<64 where off+j < nNope {
                refKv[off+j] = MetalRuntime.e4m3Dequant(min(max(kv[off+j]/scale, -448), 448)) * scale
            }
            off += 64
        }
        for i in 0..<headDim { XCTAssertEqual(gkv[i], refKv[i], accuracy: max(abs(refKv[i]),1)*1e-4, "kvfp8 kv \(i)") }
        for i in 0..<headDim {
            let expected = Float(Float16(refKv[i]))
            XCTAssertEqual(graw[rawRow*headDim+i], expected, accuracy: max(abs(expected),1)*1e-3, "kvfp8 raw \(i)")
        }
    }

    func testCompressorStoreOne() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x4D43
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let width = 200, ratio = 4, pos = 9
        let stateRows = 2 * ratio
        var kv = [Float](repeating: 0, count: width)
        var sc = [Float](repeating: 0, count: width)
        var ape = [Float](repeating: 0, count: ratio * width)
        for i in 0..<width { kv[i] = rndF(); sc[i] = rndF() }
        for i in 0..<ape.count { ape[i] = rndF() }
        let stateKv = [Float](repeating: -1, count: stateRows * width)
        let stateScore = [Float](repeating: -1, count: stateRows * width)

        let (gk, gs) = try rt.compressorStoreOne(kv: kv, score: sc, ape: ape, stateKv: stateKv, stateScore: stateScore,
                                                 width: width, ratio: ratio, pos: pos)
        let posMod = pos % ratio
        let dstRow = ratio == 4 ? ratio + posMod : posMod
        for g in 0..<width {
            let dst = dstRow * width + g
            XCTAssertEqual(gk[dst], kv[g], "store kv \(g)")
            XCTAssertEqual(gs[dst], sc[g] + ape[posMod*width+g], accuracy: max(abs(sc[g]),1)*1e-5, "store score \(g)")
        }
    }
}
