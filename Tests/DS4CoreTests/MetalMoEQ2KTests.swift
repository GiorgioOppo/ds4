import XCTest
@testable import DS4Metal

/// Phase 9 (MoE): validates the routed Q2_K matvec (real kernel_mul_mv_id_q2_K_f32)
/// vs a CPU reference that dequantizes the same random block_q2_K bytes canonically.
final class MetalMoEQ2KTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/moe.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private func half(_ w: [UInt8], _ off: Int) -> Float {
        Float(Float16(bitPattern: UInt16(w[off]) | (UInt16(w[off+1]) << 8)))
    }

    // Canonical ggml Q2_K dequant of one 256-element block (84 bytes):
    // scales[16] @0, qs[64] @16, d half @80, dmin half @82.
    private func dequantBlock(_ w: [UInt8], _ base: Int) -> [Float] {
        let d = half(w, base + 80), dmin = half(w, base + 82)
        let scales = base, qsBase = base + 16
        var out = [Float](repeating: 0, count: 256)
        var oi = 0, isb = 0
        var n = 0
        while n < 256 {
            let q = qsBase + (n / 128) * 32
            var shift = 0
            for _ in 0..<4 {
                var sc = w[scales + isb]; isb += 1
                var dl = d * Float(sc & 0xF); var ml = dmin * Float(sc >> 4)
                for l in 0..<16 { out[oi] = dl * Float((w[q + l] >> shift) & 3) - ml; oi += 1 }
                sc = w[scales + isb]; isb += 1
                dl = d * Float(sc & 0xF); ml = dmin * Float(sc >> 4)
                for l in 0..<16 { out[oi] = dl * Float((w[q + l + 16] >> shift) & 3) - ml; oi += 1 }
                shift += 2
            }
            n += 128
        }
        return out
    }

    func testMoEQ2KMatvecRoutes() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x2222
        func nextByte() -> UInt8 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return UInt8(truncatingIfNeeded: seed >> 40) }
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let inDim = 512, outDim = 128, nExperts = 8
        let nblk = inDim / 256
        let rowBytes = nblk * 84
        let expertBytes = rowBytes * outDim

        var experts = [UInt8](repeating: 0, count: expertBytes * nExperts)
        var off = 0
        for _ in 0..<(nExperts * outDim * nblk) {
            for i in 0..<16 { experts[off + i] = nextByte() }      // scales
            for i in 0..<64 { experts[off + 16 + i] = nextByte() } // qs
            let d = Float16(abs(rndF()) * 0.05), dmin = Float16(abs(rndF()) * 0.02)
            withUnsafeBytes(of: d.bitPattern.littleEndian) { experts[off+80] = $0[0]; experts[off+81] = $0[1] }
            withUnsafeBytes(of: dmin.bitPattern.littleEndian) { experts[off+82] = $0[0]; experts[off+83] = $0[1] }
            off += 84
        }
        var activation = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim { activation[i] = rndF() }

        let ids: [Int32] = [3, 0, 7, 5, 1, 6]
        let gpu = try rt.moeMatvecQ2_K(experts: experts, expertIds: ids, activation: activation,
                                       nExperts: nExperts, inDim: inDim, outDim: outDim)

        var maxRel: Float = 0
        for (slot, e) in ids.enumerated() {
            for o in 0..<outDim {
                let rowBase = Int(e) * expertBytes + o * rowBytes
                var ref: Float = 0
                for blk in 0..<nblk {
                    let dq = dequantBlock(experts, rowBase + blk * 84)
                    for i in 0..<256 { ref += dq[i] * activation[blk * 256 + i] }
                }
                maxRel = max(maxRel, abs(gpu[slot * outDim + o] - ref) / max(abs(ref), 1.0))
            }
        }
        XCTAssertLessThan(maxRel, 2e-3, "MoE Q2_K routed matvec max rel err \(maxRel)")
    }
}
