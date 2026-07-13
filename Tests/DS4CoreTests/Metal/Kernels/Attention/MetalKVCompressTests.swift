import XCTest
import Foundation
@testable import DS4Metal

/// Phase 9 / Stage A4: validates the real dsv4_kv.metal kernels
/// (kernel_dsv4_fp8_kv_quantize_f32, kernel_dsv4_ratio4_shift_f32) vs CPU.
final class MetalKVCompressTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_kv.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testFP8KVQuantize() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xF8A1
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 26) }

        let headDim = 512, nTok = 3, nRot = 64
        var x = [Float](repeating: 0, count: nTok * headDim)
        for i in 0..<x.count { x[i] = rndF() }

        let gpu = try rt.fp8KVQuantize(x, nTok: nTok, headDim: headDim, nRot: nRot)
        XCTAssertEqual(gpu.count, nTok * headDim)

        // CPU reference: per 64-block amax/scale, E4M3 dequant; RoPE tail unchanged.
        let nNope = headDim - nRot
        for t in 0..<nTok {
            let base = t * headDim
            var off = 0
            while off < nNope {
                var amax: Float = 0
                for j in 0..<64 where off + j < nNope { amax = max(amax, abs(x[base + off + j])) }
                amax = max(amax, 1e-4)
                let scale = exp2((log2(amax / 448.0)).rounded(.up))
                for j in 0..<64 where off + j < nNope {
                    let q = MetalRuntime.e4m3Dequant(min(max(x[base+off+j]/scale, -448), 448)) * scale
                    XCTAssertEqual(gpu[base+off+j], q, accuracy: max(abs(q),1)*1e-4, "fp8 t=\(t) i=\(off+j)")
                }
                off += 64
            }
            for i in nNope..<headDim { XCTAssertEqual(gpu[base+i], x[base+i], "rope tail t=\(t) i=\(i)") }
        }
    }

    func testRatio4Shift() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xF8A2
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let width = 70
        let n = 4 * width
        var kv = [Float](repeating: 0, count: 2 * n)
        var sc = [Float](repeating: 0, count: 2 * n)
        for i in 0..<kv.count { kv[i] = rndF() }
        for i in 0..<sc.count { sc[i] = rndF() }

        let (gk, gs) = try rt.ratio4Shift(stateKv: kv, stateScore: sc, width: width)
        for i in 0..<n {
            XCTAssertEqual(gk[i], kv[n + i], "kv shift \(i)")
            XCTAssertEqual(gs[i], sc[n + i], "score shift \(i)")
        }
    }
}
