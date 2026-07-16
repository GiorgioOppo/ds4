import XCTest
import Foundation
@testable import DS4Metal

/// Validates the two-span extension of flashAttnCore (the decode path): attention
/// over raw rows (kvF32, nKeys) PLUS compressed rows (comp, nComp), concatenated in
/// kvF16. Compares to a CPU softmax attention over the union of keys. This is the
/// NSA attention used on compressed layers; previously untested.
final class GraphTwoSpanAttnTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"
    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/flash_attn.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testTwoSpanAttentionMatchesCPU() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x7A57
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let headDim = 512, nHead = 4, nKeys = 20, nComp = 5   // total 25 -> exercises kvpad
        let total = nKeys + nComp
        let q = (0..<(nHead*headDim)).map { _ in rnd() }
        let raw = (0..<(nKeys*headDim)).map { _ in rnd() }
        let comp = (0..<(nComp*headDim)).map { _ in rnd() }

        let sb = GraphContext.flashScratchBytes(nHead: nHead, nKeys: total)
        let qT = try GPUTensor.floats(rt, q)
        let rawT = try GPUTensor.floats(rt, raw)
        let compT = try GPUTensor.floats(rt, comp)
        let kvF16 = try GPUTensor.zerosBytes(rt, byteLength: sb.kvF16)
        let mask = try GPUTensor.zerosBytes(rt, byteLength: sb.mask)
        let sinks = try GPUTensor.zerosBytes(rt, byteLength: sb.sinks)
        let pad = try GPUTensor.zerosBytes(rt, byteLength: sb.pad)
        let tmp = try GPUTensor.zerosBytes(rt, byteLength: sb.tmp)
        let heads = try GPUTensor.zeros(rt, floatCount: nHead*headDim)

        let gc = GraphContext(rt); try gc.begin()
        try gc.flashAttnCore(q: qT, kvF32: rawT, kvF16: kvF16, mask: mask, sinks: sinks, pad: pad, tmp: tmp,
                             heads: heads, nHead: nHead, nKeys: nKeys, hasSinks: false, comp: compT, nComp: nComp)
        gc.commit()
        let gpu = heads.floatArray(nHead*headDim)

        // CPU reference: keys = raw ++ comp, F16-rounded (kernel attends in F16).
        var keys = [Float](repeating: 0, count: total*headDim)
        for i in 0..<(nKeys*headDim) { keys[i] = Float(Float16(raw[i])) }
        for i in 0..<(nComp*headDim) { keys[nKeys*headDim+i] = Float(Float16(comp[i])) }
        let scale = 1.0 / Float(headDim).squareRoot()
        var maxRel: Float = 0
        for h in 0..<nHead {
            var s = [Float](repeating: 0, count: total); var m = -Float.infinity
            for k in 0..<total { var dot: Float = 0; for d in 0..<headDim { dot += q[h*headDim+d]*keys[k*headDim+d] }; s[k] = dot*scale; m = max(m, s[k]) }
            var sum: Float = 0; for k in 0..<total { s[k] = expf(s[k]-m); sum += s[k] }
            var out = [Float](repeating: 0, count: headDim)
            for k in 0..<total { let w = s[k]/sum; for d in 0..<headDim { out[d] += w*keys[k*headDim+d] } }
            for d in 0..<headDim { maxRel = max(maxRel, abs(gpu[h*headDim+d]-out[d]) / max(abs(out[d]), 0.05)) }
        }
        XCTAssertLessThan(maxRel, 2e-2, "two-span attention max rel err \(maxRel)")
    }
}
