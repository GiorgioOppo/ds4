import XCTest
import Foundation
@testable import DS4Metal

/// Validates the ENCODE-FORM compressor pieces used inside decodeRoute
/// (compressorStoreOneEnc + compressorPoolEnc + ratio4ShiftEnc, with F16 APE —
/// the real model path) against the independent CPU oracle, driven recurrently
/// over N tokens through a GraphContext. The standalone helpers were validated
/// earlier; this guards the GraphContext re-implementations + the F16 APE read,
/// which the decode actually uses.
final class GraphCompressorTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"
    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_kv.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private func f16round(_ x: Float) -> Float { Float(Float16(x)) }

    // CPU oracle pooling (= compressor_pool_decode_state), two-lane for ratio-4.
    private func poolRef(_ kv: [Float], _ sc: [Float], headDim: Int, ratio: Int) -> [Float] {
        let coff = ratio == 4 ? 2 : 1, width = coff * headDim
        var out = [Float](repeating: 0, count: headDim)
        for j in 0..<headDim {
            var maxS: Float = -.infinity
            if ratio == 4 {
                for r in 0..<ratio { maxS = max(maxS, sc[r*width+j]); maxS = max(maxS, sc[(ratio+r)*width+headDim+j]) }
            } else {
                for r in 0..<ratio { maxS = max(maxS, sc[r*width+j]) }
            }
            if maxS <= -1e30 * 0.5 { out[j] = 0; continue }
            var denom: Float = 0, sum: Float = 0
            if ratio == 4 {
                for r in 0..<ratio {
                    let wp = Foundation.expf(sc[r*width+j] - maxS)
                    let wc = Foundation.expf(sc[(ratio+r)*width+headDim+j] - maxS)
                    denom += wp + wc
                    sum += wp * kv[r*width+j] + wc * kv[(ratio+r)*width+headDim+j]
                }
            } else {
                for r in 0..<ratio {
                    let w = Foundation.expf(sc[r*width+j] - maxS); denom += w; sum += w * kv[r*width+j]
                }
            }
            out[j] = denom > 0 ? sum/denom : 0
        }
        return out
    }
    private func shiftRatio4(_ s: inout [Float], width: Int) {
        for r in 0..<4 { for j in 0..<width { s[r*width+j] = s[(4+r)*width+j] } }
        for r in 0..<4 { for j in 0..<width { s[(4+r)*width+j] = s[r*width+j] } }
    }

    func testEncodeFormCompressorVsCPU() throws {
        let rt = try makeRuntime()
        let ratio = 4, headDim = 128, coff = 2, width = coff * headDim, rows = coff * ratio
        var seed: UInt64 = 0x5151
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        // F16 APE (ratio*width), and its F16-rounded float view for the CPU oracle.
        let apeVals = (0..<(ratio*width)).map { _ in rnd() * 0.5 }
        var apeBytes = [UInt8](); apeBytes.reserveCapacity(ratio*width*2)
        for v in apeVals { withUnsafeBytes(of: Float16(v).bitPattern.littleEndian) { apeBytes.append($0[0]); apeBytes.append($0[1]) } }
        let apeT = try GPUTensor.bytes(rt, apeBytes, elementCount: ratio*width)
        let apeF = apeVals.map { f16round($0) }

        let comp = try CompressorState(rt, ratio: ratio, headDim: headDim, maxComp: 8)
        var cKv = [Float](repeating: 0, count: rows*width), cSc = [Float](repeating: -1e30, count: rows*width)
        var maxAbs: Float = 0, emits = 0

        for pos in 0..<13 {
            let kvCur = (0..<width).map { _ in rnd() }
            let scCur = (0..<width).map { _ in rnd() * 2 }
            let kvT = try GPUTensor.floats(rt, kvCur), scT = try GPUTensor.floats(rt, scCur)
            // GPU encode-form store.
            let g1 = GraphContext(rt); try g1.begin()
            try g1.compressorStoreOneEnc(kvCur: kvT, scCur: scT, ape: apeT, apeType: 1,
                                         stateKv: comp.stateKv, stateScore: comp.stateScore, width: width, ratio: ratio, pos: pos)
            g1.commit()
            // CPU store.
            let posMod = pos % ratio, dstRow = ratio + posMod
            for j in 0..<width { cKv[dstRow*width+j] = kvCur[j]; cSc[dstRow*width+j] = scCur[j] + apeF[posMod*width+j] }

            guard (pos+1) % ratio == 0 else { continue }
            emits += 1
            let g2 = GraphContext(rt); try g2.begin()
            try g2.compressorPoolEnc(comp, out: comp.rowScratch)
            g2.commit()
            let gPooled = comp.rowScratch.floatArray(headDim)
            let cPooled = poolRef(cKv, cSc, headDim: headDim, ratio: ratio)
            for j in 0..<headDim { maxAbs = max(maxAbs, abs(gPooled[j] - cPooled[j])) }
            // shift both (GPU encode-form vs CPU).
            let g3 = GraphContext(rt); try g3.begin()
            try g3.ratio4ShiftEnc(stateKv: comp.stateKv, stateScore: comp.stateScore, width: width)
            g3.commit()
            shiftRatio4(&cKv, width: width); shiftRatio4(&cSc, width: width)
        }
        XCTAssertEqual(emits, 3)
        XCTAssertLessThan(maxAbs, 2e-3, "encode-form compressor diverges from CPU oracle (maxAbs=\(maxAbs))")
    }
}
