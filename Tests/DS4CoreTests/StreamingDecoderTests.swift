import XCTest
import Foundation
@testable import DS4Metal

/// Stage D: validates that the per-layer STREAMING decode (load layer -> compute
/// in its own command buffer -> evict) produces the SAME logits as the resident
/// decode (one command buffer). Proves the split-command-buffer streaming loop is
/// correct; the only difference vs resident is cb boundaries + per-layer eviction.
final class StreamingDecoderTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_hc.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private var seed: UInt64 = 0x57EA
    private func rf() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }
    private func q8(_ rows: Int, _ cols: Int) -> [UInt8] {
        var out = [UInt8](); for _ in 0..<rows { var c = 0; while c < cols {
            var amax: Float = 0; var blk = [Float](repeating: 0, count: 32)
            for i in 0..<32 { blk[i] = rf(); amax = max(amax, abs(blk[i])) }
            let dq = amax/127.0
            withUnsafeBytes(of: Float16(dq).bitPattern.littleEndian) { out.append($0[0]); out.append($0[1]) }
            for i in 0..<32 { out.append(UInt8(bitPattern: Int8(clamping: dq != 0 ? Int((blk[i]/dq).rounded()) : 0))) }
            c += 32 } }
        return out
    }

    func testStreamingMatchesResident() throws {
        let rt = try makeRuntime()
        let d = DSV4Dims(nEmbd: 512, nHC: 4, headDim: 512, nHead: 2, qRank: 256, qDim: 1024,
                         sharedFfn: 512, nExperts: 256, expertFfn: 256, k: 6, nRot: 64, vocab: 1024, nOutGroup: 2, nLoraO: 128)
        let hcDim = d.nHC * d.nEmbd
        let nKeys = 32, nLayer = 2

        func tf(_ n: Int) throws -> GPUTensor { var a = [Float](repeating: 0, count: n); for i in 0..<n { a[i] = rf() }; return try GPUTensor.floats(rt, a) }
        func tf16(_ n: Int) throws -> GPUTensor { var a = [UInt16](repeating: 0, count: n); for i in 0..<n { a[i] = Float16(rf()).bitPattern }; return try GPUTensor.bytes(rt, a.withUnsafeBytes { Array($0) }, elementCount: n) }
        func tq8(_ r: Int, _ c: Int) throws -> GPUTensor { try GPUTensor.bytes(rt, q8(r, c), elementCount: r*c) }
        func texp(_ e: Int, _ r: Int, _ inD: Int) throws -> GPUTensor { try GPUTensor.zerosBytes(rt, byteLength: e*r*(inD/256)*144) }
        func layer() throws -> LayerWeights {
            try LayerWeights(hcAttnFn: tf16(24*hcDim), attnScale: tf(3), attnBase: tf(24), attnNorm: tf(d.nEmbd),
                qA: tq8(d.qRank, d.nEmbd), qANorm: tf(d.qRank), qB: tq8(d.qDim, d.qRank), kvW: tq8(d.headDim, d.nEmbd),
                kvNorm: tf(d.headDim), attnSinks: tf(d.nHead), attnOutA: tq8(d.nOutGroup*d.nLoraO, d.attnGroupDim), attnOut: tq8(d.nEmbd, d.attnLowDim), hcFfnFn: tf16(24*hcDim), ffnScale: tf(3),
                ffnBase: tf(24), ffnNorm: tf(d.nEmbd), sharedGate: tq8(d.sharedFfn, d.nEmbd), sharedUp: tq8(d.sharedFfn, d.nEmbd),
                sharedDown: tq8(d.nEmbd, d.sharedFfn), routerW: tq8(d.nExperts, d.nEmbd),
                expGate: texp(d.nExperts, d.expertFfn, d.nEmbd), expUp: texp(d.nExperts, d.expertFfn, d.nEmbd),
                expDown: texp(d.nExperts, d.nEmbd, d.expertFfn))
        }
        let layers = [try layer(), try layer()]
        var emb = [UInt16](repeating: 0, count: d.vocab*d.nEmbd); for i in 0..<emb.count { emb[i] = Float16(rf()).bitPattern }
        let embedTable = try GPUTensor.bytes(rt, emb.withUnsafeBytes { Array($0) }, elementCount: d.vocab*d.nEmbd)
        let oh = OutputHeadWeights(hcFn: try tf16(d.nHC*hcDim), hcScaleScalar: 0.9, hcBase: try tf(d.nHC), norm: try tf(d.nEmbd), head: try tq8(d.vocab, d.nEmbd))
        let rope = RopeParams(nCtxOrig: 4096, freqBase: 10000, freqScale: 1, extFactor: 0, attnFactor: 1, betaFast: 32, betaSlow: 1)

        let resident = try DSV4Decoder(rt: rt, dims: d, rope: rope, layers: layers, embedTable: embedTable, out: oh, maxKeys: nKeys)
        let streaming = try StreamingDecoder(rt: rt, dims: d, rope: rope, nLayers: nLayer,
                                             layerProvider: { layers[$0] }, embedTable: embedTable, out: oh, maxKeys: nKeys)

        let lr = try resident.forward(token: 7, pos: 0, nKeys: nKeys)
        let ls = try streaming.forward(token: 7, pos: 0, nKeys: nKeys)
        XCTAssertEqual(lr.count, ls.count)
        var maxAbs: Float = 0
        for i in 0..<lr.count { maxAbs = max(maxAbs, abs(lr[i] - ls[i])) }
        XCTAssertLessThan(maxAbs, 1e-3, "streaming vs resident logit max abs diff \(maxAbs)")
    }
}
