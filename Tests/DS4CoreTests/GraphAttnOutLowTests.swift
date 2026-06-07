import XCTest
import Foundation
@testable import DS4Metal

/// Stage E: validates the FAITHFUL attention-output stage composed from validated
/// encode-ops — grouped low-rank (attnOutLowQ8) -> output_b matmulQ8 -> hcExpand4
/// — chained in one command buffer, vs a CPU reference. This is the replacement
/// for the single-matmul approximation in decodeRoute.
final class GraphAttnOutLowTests: XCTestCase {
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
            let blk = Array(row[b..<b+32]); let amax = blk.map { abs($0) }.max() ?? 0; let dd = amax/127.0
            withUnsafeBytes(of: Float16(dd).bitPattern.littleEndian) { out.append(contentsOf: $0) }
            for x in blk { out.append(UInt8(bitPattern: Int8(clamping: dd != 0 ? Int((x/dd).rounded()) : 0))) }
            b += 32
        }
        return out
    }
    private func dotQ8(_ q: [UInt8], _ base: Int, _ x: [Float], _ xoff: Int, _ inDim: Int) -> Float {
        var acc: Float = 0
        for blk in 0..<(inDim/32) {
            let o = base + blk*34
            let d = Float(Float16(bitPattern: UInt16(q[o]) | (UInt16(q[o+1]) << 8)))
            for i in 0..<32 { acc += Float(Int8(bitPattern: q[o+2+i])) * d * x[xoff + blk*32 + i] }
        }
        return acc
    }

    func testAttnOutputStageLowRank() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xA0B1
        func rf() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let nGroups = 4, groupDim = 512, rank = 128
        let lowDim = nGroups * rank   // 512
        let nEmbd = 512, nHC = 4
        let rowABytes = (groupDim/32)*34

        // output_a: nGroups groups of (rank x groupDim) Q8
        var oaBytes: [UInt8] = []
        for _ in 0..<(nGroups*rank) { var row = [Float](repeating: 0, count: groupDim); for i in 0..<groupDim { row[i] = rf() }; oaBytes += quantQ8(row) }
        // output_b: nEmbd rows of lowDim Q8
        var obBytes: [UInt8] = []
        for _ in 0..<nEmbd { var row = [Float](repeating: 0, count: lowDim); for i in 0..<lowDim { row[i] = rf() }; obBytes += quantQ8(row) }
        var heads = [Float](repeating: 0, count: nGroups*groupDim)
        for i in 0..<heads.count { heads[i] = rf() }
        var residual = [Float](repeating: 0, count: nHC*nEmbd); for i in 0..<residual.count { residual[i] = rf() }
        var post = [Float](repeating: 0, count: nHC); for i in 0..<post.count { post[i] = rf() }
        var comb = [Float](repeating: 0, count: nHC*nHC); for i in 0..<comb.count { comb[i] = rf() }

        let ctx = GraphContext(rt)
        let oa = try GPUTensor.bytes(rt, oaBytes, elementCount: 1)
        let ob = try GPUTensor.bytes(rt, obBytes, elementCount: 1)
        let ht = try GPUTensor.floats(rt, heads)
        let low = try GPUTensor.zeros(rt, floatCount: lowDim)
        let blockOut = try GPUTensor.zeros(rt, floatCount: nEmbd)
        let resT = try GPUTensor.floats(rt, residual)
        let postT = try GPUTensor.floats(rt, post)
        let combT = try GPUTensor.floats(rt, comb)
        let outT = try GPUTensor.zeros(rt, floatCount: nHC*nEmbd)

        try ctx.begin()
        try ctx.attnOutLowQ8(outputA: oa, heads: ht, low: low, nGroups: nGroups, groupDim: groupDim, rank: rank)
        try ctx.matmulQ8_0(weight: ob, x: low, out: blockOut, inDim: lowDim, outDim: nEmbd)
        try ctx.hcExpand4(blockOut: blockOut, residual: resT, post: postT, comb: combT, blockAdd: nil,
                          out: outT, nEmbd: nEmbd, nTokens: 1)
        ctx.commit()

        // CPU reference
        var lowRef = [Float](repeating: 0, count: lowDim)
        for g in 0..<nGroups { for r in 0..<rank {
            lowRef[g*rank + r] = dotQ8(oaBytes, (g*rank + r)*rowABytes, heads, g*groupDim, groupDim)
        } }
        var blk = [Float](repeating: 0, count: nEmbd)
        let obRow = (lowDim/32)*34
        for e in 0..<nEmbd { blk[e] = dotQ8(obBytes, e*obRow, lowRef, 0, lowDim) }
        let got = outT.floatArray(nHC*nEmbd)
        var maxRel: Float = 0
        for d in 0..<nEmbd {
            var r = [Float](repeating: 0, count: nHC)
            for j in 0..<nHC { r[j] = residual[d + j*nEmbd] }
            for k in 0..<nHC {
                var acc = blk[d] * post[k]
                for j in 0..<nHC { acc += comb[k + j*nHC] * r[j] }
                maxRel = max(maxRel, abs(got[d + k*nEmbd] - acc) / max(abs(acc), 0.5))
            }
        }
        XCTAssertLessThan(maxRel, 5e-3, "attn-output low-rank stage max rel \(maxRel)")
    }
}
