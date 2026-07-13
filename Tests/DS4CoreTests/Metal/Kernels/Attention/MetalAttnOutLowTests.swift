import XCTest
import Foundation
@testable import DS4Metal

/// Stage E: validates the real moe.metal grouped low-rank attention-output kernel
/// (kernel_dsv4_attn_out_low_q8_0_f32) vs a CPU reference that dequantizes the
/// same Q8_0 group weights and does the per-group matvec.
final class MetalAttnOutLowTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/moe.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private func quantQ8(_ row: [Float]) -> [UInt8] {
        var out: [UInt8] = []; var b = 0
        while b < row.count {
            let blk = Array(row[b..<b+32]); let amax = blk.map { abs($0) }.max() ?? 0; let d = amax/127.0
            withUnsafeBytes(of: Float16(d).bitPattern.littleEndian) { out.append(contentsOf: $0) }
            for x in blk { out.append(UInt8(bitPattern: Int8(clamping: d != 0 ? Int((x/d).rounded()) : 0))) }
            b += 32
        }
        return out
    }

    func testAttnOutLowGroupedMatvec() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xA001
        func rf() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let nGroups = 4, groupDim = 512, rank = 128
        // weights: nGroups groups of (rank x groupDim) -> Q8
        var wF = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: groupDim), count: rank), count: nGroups)
        var outA: [UInt8] = []
        for g in 0..<nGroups {
            for r in 0..<rank {
                var row = [Float](repeating: 0, count: groupDim)
                for i in 0..<groupDim { row[i] = rf() }
                wF[g][r] = row
                outA += quantQ8(row)
            }
        }
        var heads = [Float](repeating: 0, count: nGroups * groupDim)
        for i in 0..<heads.count { heads[i] = rf() }

        let gpu = try rt.attnOutLowQ8(outputA: outA, heads: heads, nGroups: nGroups, groupDim: groupDim, rank: rank)
        XCTAssertEqual(gpu.count, nGroups * rank)

        // CPU ref: low[g][r] = dequant(wF[g][r]) . heads[g]  (Q8 = exact value used by kernel)
        var maxRel: Float = 0
        for g in 0..<nGroups {
            // recompute the Q8-rounded weight as the kernel sees it
            for r in 0..<rank {
                let qbytes = quantQ8(wF[g][r])
                var ref: Float = 0
                for blk in 0..<(groupDim/32) {
                    let o = blk * 34
                    let d = Float(Float16(bitPattern: UInt16(qbytes[o]) | (UInt16(qbytes[o+1]) << 8)))
                    for i in 0..<32 { ref += Float(Int8(bitPattern: qbytes[o+2+i])) * d * heads[g*groupDim + blk*32 + i] }
                }
                let got = gpu[g*rank + r]
                maxRel = max(maxRel, abs(got - ref) / max(abs(ref), 1))
            }
        }
        XCTAssertLessThan(maxRel, 2e-3, "attn_out_low grouped matvec max rel \(maxRel)")
    }
}
