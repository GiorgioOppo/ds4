import XCTest
@testable import DS4Metal

/// Phase 9 (MoE): validates the expert-routed Q4_K matvec (real
/// kernel_mul_mv_id_q4_K_f32) against a CPU reference that dequantizes the SAME
/// random block_q4_K bytes the canonical way and dots with the activation.
/// Random bytes are fine: both sides read identical bytes, so a match proves the
/// Swift dequant reconstruction equals the kernel's.
final class MetalMoEQ4KTests: XCTestCase {
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

    // Canonical ggml Q4_K scale/min unpack.
    private func scaleMinK4(_ j: Int, _ s: [UInt8], _ base: Int) -> (Float, Float) {
        if j < 4 {
            return (Float(s[base + j] & 63), Float(s[base + j + 4] & 63))
        }
        let d = (s[base + j + 4] & 0xF) | ((s[base + j - 4] >> 6) << 4)
        let m = (s[base + j + 4] >> 4) | ((s[base + j] >> 6) << 4)
        return (Float(d), Float(m))
    }

    // Dequantize one 256-element block_q4_K (144 bytes) at blockBase.
    private func dequantBlock(_ w: [UInt8], _ blockBase: Int) -> [Float] {
        let d = half(w, blockBase), dmin = half(w, blockBase + 2)
        let scalesBase = blockBase + 4
        var q = blockBase + 16
        var out = [Float](repeating: 0, count: 256)
        var oi = 0, isb = 0, j = 0
        while j < 256 {
            let (sc1, m1) = scaleMinK4(isb, w, scalesBase); let d1 = d * sc1; let mm1 = dmin * m1
            let (sc2, m2) = scaleMinK4(isb + 1, w, scalesBase); let d2 = d * sc2; let mm2 = dmin * m2
            for l in 0..<32 { out[oi] = d1 * Float(w[q + l] & 0xF) - mm1; oi += 1 }
            for l in 0..<32 { out[oi] = d2 * Float(w[q + l] >> 4) - mm2; oi += 1 }
            q += 32; isb += 2; j += 64
        }
        return out
    }

    func testMoEQ4KMatvecRoutes() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x4444
        func nextByte() -> UInt8 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return UInt8(truncatingIfNeeded: seed >> 40) }
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let inDim = 512, outDim = 128, nExperts = 8
        let nblk = inDim / 256
        let rowBytes = nblk * 144
        let expertBytes = rowBytes * outDim

        // Random block_q4_K bytes per expert/row/block (moderate d/dmin).
        var experts = [UInt8](repeating: 0, count: expertBytes * nExperts)
        var off = 0
        for _ in 0..<(nExperts * outDim * nblk) {
            let d = Float16(abs(rndF()) * 0.05), dmin = Float16(abs(rndF()) * 0.02)
            withUnsafeBytes(of: d.bitPattern.littleEndian) { experts[off] = $0[0]; experts[off+1] = $0[1] }
            withUnsafeBytes(of: dmin.bitPattern.littleEndian) { experts[off+2] = $0[0]; experts[off+3] = $0[1] }
            for i in 0..<12 { experts[off + 4 + i] = nextByte() }
            for i in 0..<128 { experts[off + 16 + i] = nextByte() }
            off += 144
        }
        var activation = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim { activation[i] = rndF() }

        let ids: [Int32] = [3, 0, 7, 5, 1, 6]
        let gpu = try rt.moeMatvecQ4_K(experts: experts, expertIds: ids, activation: activation,
                                       nExperts: nExperts, inDim: inDim, outDim: outDim)
        XCTAssertEqual(gpu.count, ids.count * outDim)

        var maxRel: Float = 0
        for (slot, e) in ids.enumerated() {
            for o in 0..<outDim {
                let rowBase = Int(e) * expertBytes + o * rowBytes
                var ref: Float = 0
                for blk in 0..<nblk {
                    let dq = dequantBlock(experts, rowBase + blk * 144)
                    for i in 0..<256 { ref += dq[i] * activation[blk * 256 + i] }
                }
                let got = gpu[slot * outDim + o]
                maxRel = max(maxRel, abs(got - ref) / max(abs(ref), 1.0))
            }
        }
        XCTAssertLessThan(maxRel, 2e-3, "MoE Q4_K routed matvec max rel err \(maxRel)")
    }
}
