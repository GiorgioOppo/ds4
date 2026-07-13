import XCTest
@testable import DS4Metal

/// Phase 9 / Stage A1: validates the real metal/dense.metal prefill matmul
/// kernels (kernel_mul_mm_q8_0_f32 / kernel_mul_mm_f16_f32) vs CPU, multi-token.
final class MetalMatmulMMTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dense.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private func quantizeQ8_0(_ row: [Float]) -> [UInt8] {
        var out: [UInt8] = []
        var b = 0
        while b < row.count {
            let block = row[b..<b+32]
            let amax = block.map { abs($0) }.max() ?? 0
            let d = amax / 127.0
            withUnsafeBytes(of: Float16(d).bitPattern.littleEndian) { out.append(contentsOf: $0) }
            for x in block {
                let q = d != 0 ? Int((x / d).rounded()) : 0
                out.append(UInt8(bitPattern: Int8(clamping: q)))
            }
            b += 32
        }
        return out
    }

    func testMulMMQ8AndF16() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x9911
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        // sizes that exercise both aligned and broadcast (bc_out) paths
        for (inDim, outDim, nTok) in [(256, 128, 32), (512, 300, 40)] {
            var wF = [Float](repeating: 0, count: inDim * outDim)
            for i in 0..<wF.count { wF[i] = rnd() }
            var x = [Float](repeating: 0, count: inDim * nTok)
            for i in 0..<x.count { x[i] = rnd() }

            // Q8_0
            var wq: [UInt8] = []
            for r in 0..<outDim { wq.append(contentsOf: quantizeQ8_0(Array(wF[r*inDim..<(r+1)*inDim]))) }
            let gq = try rt.matmulMMQ8_0(weight: wq, activation: x, inDim: inDim, outDim: outDim, nTok: nTok)
            // CPU ref reading the SAME quantized bytes
            let rowBytes = (inDim/32)*34
            var maxRelQ: Float = 0
            for t in 0..<nTok {
                for r in 0..<outDim {
                    var acc: Float = 0
                    for blk in 0..<(inDim/32) {
                        let base = r*rowBytes + blk*34
                        let d = Float(Float16(bitPattern: UInt16(wq[base]) | (UInt16(wq[base+1])<<8)))
                        var sq: Float = 0
                        for i in 0..<32 { sq += Float(Int8(bitPattern: wq[base+2+i])) * x[t*inDim + blk*32 + i] }
                        acc += sq * d
                    }
                    maxRelQ = max(maxRelQ, abs(gq[t*outDim+r] - acc)/max(abs(acc),1))
                }
            }
            XCTAssertLessThan(maxRelQ, 1e-3, "mul_mm q8_0 \(inDim)x\(outDim)x\(nTok) rel \(maxRelQ)")

            // F16
            var wh = [UInt16](repeating: 0, count: inDim*outDim)
            var whF = [Float](repeating: 0, count: inDim*outDim)
            for i in 0..<wh.count { let h = Float16(wF[i]); wh[i] = h.bitPattern; whF[i] = Float(h) }
            let gh = try rt.matmulMMF16(weight: wh, activation: x, inDim: inDim, outDim: outDim, nTok: nTok)
            var maxRelH: Float = 0
            for t in 0..<nTok {
                for r in 0..<outDim {
                    var acc: Float = 0
                    for i in 0..<inDim { acc += whF[r*inDim+i] * x[t*inDim+i] }
                    maxRelH = max(maxRelH, abs(gh[t*outDim+r] - acc)/max(abs(acc),1))
                }
            }
            XCTAssertLessThan(maxRelH, 2e-3, "mul_mm f16 \(inDim)x\(outDim)x\(nTok) rel \(maxRelH)")
        }
    }
}
