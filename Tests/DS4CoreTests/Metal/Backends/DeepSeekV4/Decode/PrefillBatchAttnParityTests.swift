import XCTest
import Foundation
@testable import DS4Metal

/// End-to-end parity of the BATCHED prefill attention (DS4_PREFILL_BATCH_ATTN,
/// default ON): a synthetic 2-layer model prefilled through the batched
/// expert-gather path (ONE multi-query FlashAttention per route run) must
/// produce the same last-token logits as the per-token forward() decode over
/// the same tokens. Everything except the attention kernel is dispatch-
/// identical; the attention differs only in accumulation order (MMA blocks vs
/// the vec kernel), so logits agree within a small tolerance, not bit-exactly.
final class PrefillBatchAttnParityTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private var seed: UInt64 = 0x9F11
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

    func testBatchedPrefillMatchesForward() throws {
        let rt = try makeRuntime()
        let d = DSV4Dims(nEmbd: 512, nHC: 4, headDim: 512, nHead: 2, qRank: 256, qDim: 1024,
                         sharedFfn: 512, nExperts: 256, expertFfn: 256, k: 6, nRot: 64, vocab: 1024,
                         nOutGroup: 2, nLoraO: 128)
        let hcDim = d.nHC * d.nEmbd
        let nKeys = 32, nLayer = 2
        let expertBytes = d.expertFfn * (d.nEmbd / 256) * 144   // gate/up == down byte size here

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

        // Reference: per-token forward() decode (resident zero experts — the
        // routed FFN contributes exactly zero on both paths, so parity covers
        // embed, HC, dense projections, ATTENTION, router and output head).
        let reference = try StreamingDecoder(rt: rt, dims: d, rope: rope, nLayers: nLayer,
                                             layerProvider: { layers[$0] }, embedTable: embedTable,
                                             out: oh, maxKeys: nKeys)
        // Batched prefill: the expert-gather path (packed zero experts) routes
        // phase A through encodeFlashRun when DS4_PREFILL_BATCH_ATTN is on.
        let prefiller = try StreamingDecoder(rt: rt, dims: d, rope: rope, nLayers: nLayer,
                                             layerProvider: { layers[$0] }, embedTable: embedTable,
                                             out: oh, maxKeys: nKeys,
                                             expertGather: { _, ids in
                                                 (try GPUTensor.zerosBytes(rt, byteLength: ids.count * expertBytes),
                                                  try GPUTensor.zerosBytes(rt, byteLength: ids.count * expertBytes),
                                                  try GPUTensor.zerosBytes(rt, byteLength: ids.count * expertBytes))
                                             })
        try XCTSkipUnless(prefiller.prefillBatchAttn, "DS4_PREFILL_BATCH_ATTN disabled in env")

        let tokens = (0..<12).map { _ in Int(abs(rf() * 1000)) % d.vocab }
        var want = [Float]()
        for (p, t) in tokens.enumerated() { want = try reference.forward(token: t, pos: p, nKeys: p + 1) }
        let got = try prefiller.prefill(tokens: tokens)

        XCTAssertGreaterThan(prefiller.profile.prefillFlashRuns, 0,
                             "batched attention path was never taken")
        if prefiller.prefillDenseMM {
            XCTAssertEqual(prefiller.profile.prefillDenseRuns, prefiller.profile.prefillFlashRuns,
                           "dense-GEMM path was expected on every run (all-Q8 fixture)")
        }
        XCTAssertEqual(want.count, got.count)
        var maxRel: Float = 0
        for i in 0..<want.count {
            maxRel = max(maxRel, abs(got[i] - want[i]) / max(abs(want[i]), 0.05))
        }
        XCTAssertLessThan(maxRel, 2e-2, "batched prefill vs forward logits max rel \(maxRel)")
    }
}
