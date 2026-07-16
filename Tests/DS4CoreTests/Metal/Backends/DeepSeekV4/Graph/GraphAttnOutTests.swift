import XCTest
import Foundation
@testable import DS4Metal

/// Stage C: validates the attention-output stage composition — project attention
/// heads (matmulQ8) then expand into the 4 HC streams (hcExpand4) — chained on
/// GPUTensors in one command buffer, vs CPU.
final class GraphAttnOutTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_hc.metal"),
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
    private func mat(_ w: [UInt8], _ x: [Float], _ inDim: Int, _ outDim: Int) -> [Float] {
        let rb = (inDim/32)*34; var o = [Float](repeating: 0, count: outDim)
        for r in 0..<outDim {
            let base = r*rb; var acc: Float = 0
            for blk in 0..<(inDim/32) {
                let bo = base+blk*34; let d = Float(Float16(bitPattern: UInt16(w[bo]) | (UInt16(w[bo+1])<<8)))
                for i in 0..<32 { acc += Float(Int8(bitPattern: w[bo+2+i])) * d * x[blk*32+i] }
            }
            o[r] = acc
        }
        return o
    }

    func testAttnOutputStage() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xC0A7
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let inDim = 2048, nEmbd = 512, nHC = 4
        var heads = [Float](repeating: 0, count: inDim)
        for i in 0..<inDim { heads[i] = rndF() }
        var projB: [UInt8] = []
        for _ in 0..<nEmbd { var row = [Float](repeating: 0, count: inDim); for i in 0..<inDim { row[i] = rndF() }; projB += quantQ8(row) }
        var residual = [Float](repeating: 0, count: nHC * nEmbd)
        var post = [Float](repeating: 0, count: nHC)
        var comb = [Float](repeating: 0, count: nHC * nHC)
        for i in 0..<residual.count { residual[i] = rndF() }
        for i in 0..<post.count { post[i] = rndF() }
        for i in 0..<comb.count { comb[i] = rndF() }

        let ctx = GraphContext(rt)
        let ht = try GPUTensor.floats(rt, heads)
        let pw = try GPUTensor.bytes(rt, projB, elementCount: nEmbd*inDim)
        let blockOut = try GPUTensor.zeros(rt, floatCount: nEmbd)
        let resT = try GPUTensor.floats(rt, residual)
        let postT = try GPUTensor.floats(rt, post)
        let combT = try GPUTensor.floats(rt, comb)
        let outT = try GPUTensor.zeros(rt, floatCount: nHC * nEmbd)

        try ctx.begin()
        try ctx.matmulQ8_0(weight: pw, x: ht, out: blockOut, inDim: inDim, outDim: nEmbd)
        try ctx.hcExpand4(blockOut: blockOut, residual: resT, post: postT, comb: combT,
                          blockAdd: nil, out: outT, nEmbd: nEmbd, nTokens: 1)
        ctx.commit()

        // CPU
        let bo = mat(projB, heads, inDim, nEmbd)
        let got = outT.floatArray(nHC * nEmbd)
        var maxRel: Float = 0
        for d in 0..<nEmbd {
            var r = [Float](repeating: 0, count: nHC)
            for j in 0..<nHC { r[j] = residual[d + j*nEmbd] }
            for k in 0..<nHC {
                var acc = bo[d] * post[k]
                for j in 0..<nHC { acc += comb[k + j*nHC] * r[j] }
                let g = got[d + k*nEmbd]
                maxRel = max(maxRel, abs(g - acc) / max(abs(acc), 0.1))
            }
        }
        XCTAssertLessThan(maxRel, 5e-3, "attn output stage max rel \(maxRel)")
    }
}
