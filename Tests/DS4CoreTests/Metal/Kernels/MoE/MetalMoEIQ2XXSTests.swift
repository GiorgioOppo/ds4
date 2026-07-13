import XCTest
@testable import DS4Metal

/// Phase 9 (MoE): validates the routed IQ2_XXS matvec (real
/// kernel_mul_mv_id_iq2_xxs_f32) vs a CPU reference that dequantizes the same
/// random block_iq2_xxs bytes with the canonical ggml grid/sign codebook.
/// This is the kernel that gate+up routed experts use in the 2-bit GGUF, and
/// it cooperatively loads the codebook into threadgroup memory — so the test
/// also guards the threadgroup-memory size (2176 B), whose previous 256 B value
/// produced out-of-bounds writes and garbage output.
final class MetalMoEIQ2XXSTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/moe.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    static let kmask: [UInt8] = [1, 2, 4, 8, 16, 32, 64, 128]
    static let ksigns: [UInt8] = [
          0,129,130,  3,132,  5,  6,135,136,  9, 10,139, 12,141,142, 15,
        144, 17, 18,147, 20,149,150, 23, 24,153,154, 27,156, 29, 30,159,
        160, 33, 34,163, 36,165,166, 39, 40,169,170, 43,172, 45, 46,175,
         48,177,178, 51,180, 53, 54,183,184, 57, 58,187, 60,189,190, 63,
        192, 65, 66,195, 68,197,198, 71, 72,201,202, 75,204, 77, 78,207,
         80,209,210, 83,212, 85, 86,215,216, 89, 90,219, 92,221,222, 95,
         96,225,226, 99,228,101,102,231,232,105,106,235,108,237,238,111,
        240,113,114,243,116,245,246,119,120,249,250,123,252,125,126,255]
    static let grid: [UInt64] = [
        0x0808080808080808,0x080808080808082b,0x0808080808081919,0x0808080808082b08,
        0x0808080808082b2b,0x0808080808190819,0x0808080808191908,0x08080808082b0808,
        0x08080808082b082b,0x08080808082b2b08,0x08080808082b2b2b,0x0808080819080819,
        0x0808080819081908,0x0808080819190808,0x0808080819192b08,0x08080808192b0819,
        0x08080808192b1908,0x080808082b080808,0x080808082b08082b,0x080808082b082b2b,
        0x080808082b2b082b,0x0808081908080819,0x0808081908081908,0x0808081908190808,
        0x0808081908191919,0x0808081919080808,0x080808192b081908,0x080808192b192b08,
        0x0808082b08080808,0x0808082b0808082b,0x0808082b082b082b,0x0808082b2b08082b,
        0x0808190808080819,0x0808190808081908,0x0808190808190808,0x08081908082b0819,
        0x08081908082b1908,0x0808190819080808,0x080819081908082b,0x0808190819082b08,
        0x08081908192b0808,0x080819082b080819,0x080819082b081908,0x080819082b190808,
        0x080819082b2b1908,0x0808191908080808,0x080819190808082b,0x0808191908082b08,
        0x08081919082b0808,0x080819191908192b,0x08081919192b2b19,0x080819192b080808,
        0x080819192b190819,0x0808192b08082b19,0x0808192b08190808,0x0808192b19080808,
        0x0808192b2b081908,0x0808192b2b2b1908,0x08082b0808080808,0x08082b0808081919,
        0x08082b0808082b08,0x08082b0808191908,0x08082b08082b2b08,0x08082b0819080819,
        0x08082b0819081908,0x08082b0819190808,0x08082b081919082b,0x08082b082b082b08,
        0x08082b1908081908,0x08082b1919080808,0x08082b2b0808082b,0x08082b2b08191908,
        0x0819080808080819,0x0819080808081908,0x0819080808190808,0x08190808082b0819,
        0x0819080819080808,0x08190808192b0808,0x081908082b081908,0x081908082b190808,
        0x081908082b191919,0x0819081908080808,0x0819081908082b08,0x08190819082b0808,
        0x0819081919190808,0x0819081919192b2b,0x081908192b080808,0x0819082b082b1908,
        0x0819082b19081919,0x0819190808080808,0x0819190808082b08,0x08191908082b0808,
        0x08191908082b1919,0x0819190819082b19,0x081919082b080808,0x0819191908192b08,
        0x08191919192b082b,0x0819192b08080808,0x0819192b0819192b,0x08192b0808080819,
        0x08192b0808081908,0x08192b0808190808,0x08192b0819080808,0x08192b082b080819,
        0x08192b1908080808,0x08192b1908081919,0x08192b192b2b0808,0x08192b2b19190819,
        0x082b080808080808,0x082b08080808082b,0x082b080808082b2b,0x082b080819081908,
        0x082b0808192b0819,0x082b08082b080808,0x082b08082b08082b,0x082b0819082b2b19,
        0x082b081919082b08,0x082b082b08080808,0x082b082b0808082b,0x082b190808080819,
        0x082b190808081908,0x082b190808190808,0x082b190819080808,0x082b19081919192b,
        0x082b191908080808,0x082b191919080819,0x082b1919192b1908,0x082b192b2b190808,
        0x082b2b0808082b08,0x082b2b08082b0808,0x082b2b082b191908,0x082b2b2b19081908,
        0x1908080808080819,0x1908080808081908,0x1908080808190808,0x1908080808192b08,
        0x19080808082b0819,0x19080808082b1908,0x1908080819080808,0x1908080819082b08,
        0x190808081919192b,0x19080808192b0808,0x190808082b080819,0x190808082b081908,
        0x190808082b190808,0x1908081908080808,0x19080819082b0808,0x19080819192b0819,
        0x190808192b080808,0x190808192b081919,0x1908082b08080819,0x1908082b08190808,
        0x1908082b19082b08,0x1908082b1919192b,0x1908082b192b2b08,0x1908190808080808,
        0x1908190808082b08,0x19081908082b0808,0x190819082b080808,0x190819082b192b19,
        0x190819190819082b,0x19081919082b1908,0x1908192b08080808,0x19082b0808080819,
        0x19082b0808081908,0x19082b0808190808,0x19082b0819080808,0x19082b0819081919,
        0x19082b1908080808,0x19082b1919192b08,0x19082b19192b0819,0x19082b192b08082b,
        0x19082b2b19081919,0x19082b2b2b190808,0x1919080808080808,0x1919080808082b08,
        0x1919080808190819,0x1919080808192b19,0x19190808082b0808,0x191908082b080808,
        0x191908082b082b08,0x1919081908081908,0x191908191908082b,0x191908192b2b1908,
        0x1919082b2b190819,0x191919082b190808,0x191919082b19082b,0x1919191908082b2b,
        0x1919192b08080819,0x1919192b19191908,0x19192b0808080808,0x19192b0808190819,
        0x19192b0808192b19,0x19192b08192b1908,0x19192b1919080808,0x19192b2b08082b08,
        0x192b080808081908,0x192b080808190808,0x192b080819080808,0x192b0808192b2b08,
        0x192b081908080808,0x192b081919191919,0x192b082b08192b08,0x192b082b192b0808,
        0x192b190808080808,0x192b190808081919,0x192b191908190808,0x192b19190819082b,
        0x192b19192b081908,0x192b2b081908082b,0x2b08080808080808,0x2b0808080808082b,
        0x2b08080808082b2b,0x2b08080819080819,0x2b0808082b08082b,0x2b08081908081908,
        0x2b08081908192b08,0x2b08081919080808,0x2b08082b08190819,0x2b08190808080819,
        0x2b08190808081908,0x2b08190808190808,0x2b08190808191919,0x2b08190819080808,
        0x2b081908192b0808,0x2b08191908080808,0x2b0819191908192b,0x2b0819192b191908,
        0x2b08192b08082b19,0x2b08192b19080808,0x2b08192b192b0808,0x2b082b080808082b,
        0x2b082b1908081908,0x2b082b2b08190819,0x2b19080808081908,0x2b19080808190808,
        0x2b190808082b1908,0x2b19080819080808,0x2b1908082b2b0819,0x2b1908190819192b,
        0x2b1908192b080808,0x2b19082b19081919,0x2b19190808080808,0x2b191908082b082b,
        0x2b19190819081908,0x2b19191919190819,0x2b192b082b080819,0x2b192b19082b0808,
        0x2b2b08080808082b,0x2b2b080819190808,0x2b2b08082b081919,0x2b2b081908082b19,
        0x2b2b082b08080808,0x2b2b190808192b08,0x2b2b2b0819190808,0x2b2b2b1908081908]

    private func half(_ w: [UInt8], _ off: Int) -> Float {
        Float(Float16(bitPattern: UInt16(w[off]) | (UInt16(w[off+1]) << 8)))
    }
    private func u16(_ w: [UInt8], _ off: Int) -> UInt32 {
        UInt32(w[off]) | (UInt32(w[off+1]) << 8)
    }

    /// CPU reference for one output row over `inDim` activations, exactly
    /// mirroring kernel_mul_mv_iq2_xxs_f32_impl (including the final *0.25).
    private func rowDot(_ w: [UInt8], rowBase: Int, inDim: Int, y: [Float]) -> Float {
        let nblk = inDim / 256
        var acc: Float = 0
        for b in 0..<nblk {
            let base = rowBase + b * 66
            let dBlock = half(w, base)          // half d @0
            let qsBase = base + 2               // ushort qs[32] @2
            for ib in 0..<8 {                   // 8 sub-blocks of 32 elements
                let q0 = u16(w, qsBase + ib * 8 + 0)
                let q1 = u16(w, qsBase + ib * 8 + 2)
                let q2v = u16(w, qsBase + ib * 8 + 4)
                let q3 = u16(w, qsBase + ib * 8 + 6)
                let aux8: [Int] = [Int(q0 & 0xFF), Int(q0 >> 8), Int(q1 & 0xFF), Int(q1 >> 8)]
                let aux32 = q2v | (q3 << 16)
                let dSub = dBlock * (0.5 + Float(aux32 >> 28))
                var sub: Float = 0
                for l in 0..<4 {
                    let word = Self.grid[aux8[l]]
                    let signs = Self.ksigns[Int((aux32 >> (7 * l)) & 127)]
                    for j in 0..<8 {
                        let g = Float((word >> (8 * j)) & 0xFF)
                        let sign: Float = (signs & Self.kmask[j]) != 0 ? -1 : 1
                        sub += y[b * 256 + ib * 32 + l * 8 + j] * g * sign
                    }
                }
                acc += dSub * sub
            }
        }
        return acc * 0.25
    }

    func testIQ2XXSMatvecRoutes() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x1517
        func nextByte() -> UInt8 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return UInt8(truncatingIfNeeded: seed >> 40) }
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let inDim = 512, outDim = 128, nExperts = 8
        let nblk = inDim / 256
        let rowBytes = nblk * 66
        let expertBytes = rowBytes * outDim

        var experts = [UInt8](repeating: 0, count: expertBytes * nExperts)
        var off = 0
        for _ in 0..<(nExperts * outDim * nblk) {
            let d = Float16(abs(rndF()) * 0.05 + 0.001)
            withUnsafeBytes(of: d.bitPattern.littleEndian) { experts[off] = $0[0]; experts[off+1] = $0[1] }
            for i in 0..<64 { experts[off + 2 + i] = nextByte() }   // qs[32] uint16 = 64 bytes
            off += 66
        }
        var activation = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim { activation[i] = rndF() }

        let ids: [Int32] = [3, 0, 7, 5, 1, 6]
        let gpu = try rt.moeMatvecIQ2XXS(experts: experts, expertIds: ids, activation: activation,
                                         nExperts: nExperts, inDim: inDim, outDim: outDim)

        var maxAbs: Float = 0
        var maxRef: Float = 0
        for (slot, e) in ids.enumerated() {
            for o in 0..<outDim {
                let ref = rowDot(experts, rowBase: Int(e) * expertBytes + o * rowBytes, inDim: inDim, y: activation)
                let got = gpu[slot * outDim + o]
                maxAbs = max(maxAbs, abs(got - ref))
                maxRef = max(maxRef, abs(ref))
            }
        }
        // Relative tolerance: the kernel sums in f32 in a different order than the
        // reference, so allow small drift but require structural agreement.
        XCTAssertLessThan(maxAbs, max(2e-3, 1e-3 * maxRef), "iq2_xxs matvec diverges from CPU ref (maxAbs=\(maxAbs), maxRef=\(maxRef))")
    }
}
