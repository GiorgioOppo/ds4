import XCTest
import Foundation
@testable import DS4Metal

/// Stage C: smoke test of the full multi-layer decode forward (embed -> N layers
/// -> output head -> logits) on a tiny synthetic config. Confirms the whole
/// engine path executes end-to-end producing finite logits of size vocab.
/// Numerical fidelity is gated on the real model (>=64GB).
final class GraphDecoderTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_hc.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private var seed: UInt64 = 0xD0D0
    private func rf() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }
    private func q8(_ rows: Int, _ cols: Int) -> [UInt8] {
        var out = [UInt8](); out.reserveCapacity(rows*(cols/32)*34)
        for _ in 0..<rows { var c = 0; while c < cols {
            var amax: Float = 0; var blk = [Float](repeating: 0, count: 32)
            for i in 0..<32 { blk[i] = rf(); amax = max(amax, abs(blk[i])) }
            let d = amax/127.0
            withUnsafeBytes(of: Float16(d).bitPattern.littleEndian) { out.append($0[0]); out.append($0[1]) }
            for i in 0..<32 { out.append(UInt8(bitPattern: Int8(clamping: d != 0 ? Int((blk[i]/d).rounded()) : 0))) }
            c += 32 } }
        return out
    }

    func testFullDecodeForwardSmoke() throws {
        let rt = try makeRuntime()
        let d = DSV4Dims(nEmbd: 512, nHC: 4, headDim: 512, nHead: 2, qRank: 256, qDim: 1024,
                         sharedFfn: 512, nExperts: 256, expertFfn: 256, k: 6, nRot: 64, vocab: 1024, nOutGroup: 2, nLoraO: 128)
        let hcDim = d.nHC * d.nEmbd
        let nKeys = 32

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

        // embed table F16 [vocab x nEmbd]
        var emb = [UInt16](repeating: 0, count: d.vocab * d.nEmbd)
        for i in 0..<emb.count { emb[i] = Float16(rf()).bitPattern }
        let embedTable = try GPUTensor.bytes(rt, emb.withUnsafeBytes { Array($0) }, elementCount: d.vocab*d.nEmbd)

        let oh = OutputHeadWeights(hcFn: try GPUTensor.bytes(rt, { var a = [UInt16](repeating: 0, count: d.nHC*hcDim); for i in 0..<a.count { a[i] = Float16(rf()).bitPattern }; return a.withUnsafeBytes { Array($0) } }(), elementCount: d.nHC*hcDim),
                                   hcScaleScalar: 0.9, hcBase: try tf(d.nHC), norm: try tf(d.nEmbd), head: try tq8(d.vocab, d.nEmbd))

        let dec = try DSV4Decoder(rt: rt, dims: d, rope: RopeParams(nCtxOrig: 4096, freqBase: 10000, freqScale: 1, extFactor: 0, attnFactor: 1, betaFast: 32, betaSlow: 1),
                                  layers: layers, embedTable: embedTable, out: oh, maxKeys: nKeys)
        let logits = try dec.forward(token: 5, pos: 0, nKeys: nKeys)
        XCTAssertEqual(logits.count, d.vocab)
        var finite = true
        for v in logits where !v.isFinite { finite = false; break }
        XCTAssertTrue(finite, "full decode produced non-finite logits")

        // generate loop (decode-style prefill + autoregressive decode + sampling)
        let gen = try dec.generate(prompt: [5, 10, 15], maxNew: 5,
                                   sampling: DSV4Decoder.Sampling(temperature: 0)) // argmax
        XCTAssertEqual(gen.count, 5)
        for t in gen { XCTAssertTrue(t >= 0 && t < d.vocab, "token \(t) out of range") }
    }
}
