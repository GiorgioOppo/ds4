import XCTest
import Foundation
@testable import DS4Metal

/// Stage C: validates the encode-form FlashAttention core (cpy F32->F16 + vec +
/// reduce in ONE command buffer on GPUTensors) vs a CPU softmax-attention ref.
final class GraphAttnTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/flash_attn.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testFlashAttnCoreEncode() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xCA71
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let headDim = 512, nHead = 4, nKeys = 96
        var q = [Float](repeating: 0, count: nHead * headDim)
        var kv = [Float](repeating: 0, count: nKeys * headDim)
        for i in 0..<q.count { q[i] = rndF() }
        for i in 0..<kv.count { kv[i] = rndF() }

        let sb = GraphContext.flashScratchBytes(nHead: nHead, nKeys: nKeys)
        let ctx = GraphContext(rt)
        let qt = try GPUTensor.floats(rt, q)
        let kvt = try GPUTensor.floats(rt, kv)
        let kvF16 = try GPUTensor.zerosBytes(rt, byteLength: sb.kvF16)
        let mask = try GPUTensor.zerosBytes(rt, byteLength: sb.mask)
        let sinks = try GPUTensor.zerosBytes(rt, byteLength: sb.sinks)
        let pad = try GPUTensor.zerosBytes(rt, byteLength: sb.pad)
        let tmp = try GPUTensor.zerosBytes(rt, byteLength: sb.tmp)
        let heads = try GPUTensor.zeros(rt, floatCount: nHead * headDim)

        try ctx.begin()
        try ctx.flashAttnCore(q: qt, kvF32: kvt, kvF16: kvF16, mask: mask, sinks: sinks, pad: pad, tmp: tmp,
                              heads: heads, nHead: nHead, nKeys: nKeys)
        ctx.commit()

        var kvH = [Float](repeating: 0, count: kv.count)
        for i in 0..<kv.count { kvH[i] = Float(Float16(kv[i])) }
        let scale = 1.0 / Float(headDim).squareRoot()
        let got = heads.floatArray(nHead * headDim)
        var maxRel: Float = 0
        for h in 0..<nHead {
            var s = [Float](repeating: 0, count: nKeys); var m = -Float.infinity
            for k in 0..<nKeys {
                var dot: Float = 0
                for d in 0..<headDim { dot += q[h*headDim+d]*kvH[k*headDim+d] }
                s[k] = dot*scale; m = max(m, s[k])
            }
            var sum: Float = 0; for k in 0..<nKeys { s[k] = expf(s[k]-m); sum += s[k] }
            var out = [Float](repeating: 0, count: headDim)
            for k in 0..<nKeys { let w = s[k]/sum; for d in 0..<headDim { out[d] += w*kvH[k*headDim+d] } }
            for d in 0..<headDim { maxRel = max(maxRel, abs(got[h*headDim+d]-out[d]) / max(abs(out[d]),0.05)) }
        }
        XCTAssertLessThan(maxRel, 2e-2, "flash core encode max rel \(maxRel)")
    }

    func testFlashAttnCorePadPath() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xCA72
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let headDim = 512, nHead = 3, nKeys = 70  // NOT a multiple of 32 -> pad path
        var q = [Float](repeating: 0, count: nHead * headDim)
        var kv = [Float](repeating: 0, count: nKeys * headDim)
        for i in 0..<q.count { q[i] = rndF() }
        for i in 0..<kv.count { kv[i] = rndF() }

        let sb = GraphContext.flashScratchBytes(nHead: nHead, nKeys: nKeys)
        let ctx = GraphContext(rt)
        let qt = try GPUTensor.floats(rt, q)
        let kvt = try GPUTensor.floats(rt, kv)
        let kvF16 = try GPUTensor.zerosBytes(rt, byteLength: sb.kvF16)
        let mask = try GPUTensor.zerosBytes(rt, byteLength: sb.mask)
        let sinks = try GPUTensor.zerosBytes(rt, byteLength: sb.sinks)
        let pad = try GPUTensor.zerosBytes(rt, byteLength: sb.pad)
        let tmp = try GPUTensor.zerosBytes(rt, byteLength: sb.tmp)
        let heads = try GPUTensor.zeros(rt, floatCount: nHead * headDim)

        try ctx.begin()
        try ctx.flashAttnCore(q: qt, kvF32: kvt, kvF16: kvF16, mask: mask, sinks: sinks, pad: pad, tmp: tmp,
                              heads: heads, nHead: nHead, nKeys: nKeys)
        ctx.commit()

        var kvH = [Float](repeating: 0, count: kv.count)
        for i in 0..<kv.count { kvH[i] = Float(Float16(kv[i])) }
        let scale = 1.0 / Float(headDim).squareRoot()
        let got = heads.floatArray(nHead * headDim)
        var maxRel: Float = 0
        for h in 0..<nHead {
            var s = [Float](repeating: 0, count: nKeys); var m = -Float.infinity
            for k in 0..<nKeys {
                var dot: Float = 0
                for dd in 0..<headDim { dot += q[h*headDim+dd]*kvH[k*headDim+dd] }
                s[k] = dot*scale; m = max(m, s[k])
            }
            var sum: Float = 0; for k in 0..<nKeys { s[k] = expf(s[k]-m); sum += s[k] }
            var out = [Float](repeating: 0, count: headDim)
            for k in 0..<nKeys { let w = s[k]/sum; for dd in 0..<headDim { out[dd] += w*kvH[k*headDim+dd] } }
            for dd in 0..<headDim { maxRel = max(maxRel, abs(got[h*headDim+dd]-out[dd]) / max(abs(out[dd]),0.05)) }
        }
        XCTAssertLessThan(maxRel, 2e-2, "flash pad-path max rel \(maxRel)")
    }
}
