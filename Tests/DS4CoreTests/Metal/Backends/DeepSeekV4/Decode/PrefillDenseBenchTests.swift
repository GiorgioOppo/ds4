import XCTest
import Foundation
@testable import DS4Metal

/// Manual timing harness for the batched-prefill route (real Flash dense
/// dimensions, synthetic weights, zero experts): compares the per-token
/// route (DS4_PREFILL_DENSE_MM=0) against the dense-GEMM route on the same
/// prompt. Skipped unless DS4_BENCH=1 — run with:
///   DS4_BENCH=1 swift test --filter PrefillDenseBenchTests 2>&1 | grep BENCH
final class PrefillDenseBenchTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    /// Fast synthetic Q8_0: one shared 32-int8 ramp payload, per-block scale
    /// varied by an LCG so rows are NOT degenerate (a row-indexing bug must
    /// show up in the parity check).
    private var lcg: UInt64 = 0x1234_5678
    private func nextScale() -> Float16 {
        lcg = lcg &* 6364136223846793005 &+ 1442695040888963407
        return Float16(0.004 + Float((lcg >> 33) % 64) * 0.0004)
    }
    private func q8fast(_ rows: Int, _ cols: Int) -> [UInt8] {
        var payload = [UInt8](repeating: 0, count: 32)
        for i in 0..<32 { payload[i] = UInt8(bitPattern: Int8(truncatingIfNeeded: i * 5 - 64)) }
        let blocks = rows * (cols / 32)
        var out = [UInt8](repeating: 0, count: blocks * 34)
        out.withUnsafeMutableBytes { p in
            payload.withUnsafeBytes { b in
                for i in 0..<blocks {
                    let s = nextScale().bitPattern.littleEndian
                    withUnsafeBytes(of: s) { sb in
                        p[i * 34] = sb[0]; p[i * 34 + 1] = sb[1]
                    }
                    memcpy(p.baseAddress! + i * 34 + 2, b.baseAddress!, 32)
                }
            }
        }
        return out
    }

    private func f16fast(_ n: Int) -> [UInt16] {
        // Cycle of 16 small values — cheap, non-degenerate.
        let cycle = (0..<16).map { Float16(0.005 + Float($0) * 0.003).bitPattern }
        var out = [UInt16](repeating: 0, count: n)
        for i in 0..<n { out[i] = cycle[i & 15] }
        return out
    }

    func testRouteDenseTiming() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["DS4_BENCH"] == "1",
                          "manual benchmark (DS4_BENCH=1)")
        let rt = try makeRuntime()
        // Real Flash route dims; MoE shrunk (zero experts — phase B is not
        // what lever 2 changed) and small vocab (output head irrelevant).
        let d = DSV4Dims(nEmbd: 4096, nHC: 4, headDim: 512, nHead: 64, qRank: 1536, qDim: 32768,
                         sharedFfn: 2048, nExperts: 256, expertFfn: 512, k: 6, nRot: 64, vocab: 2048,
                         nOutGroup: 8, nLoraO: 1024)
        let hcDim = d.nHC * d.nEmbd
        let maxKeys = 512, nLayer = 4   // layers 0,1 plain; 2 ratio-4 (+idx state); 3 ratio-128
        let expertBytes = d.expertFfn * (d.nEmbd / 256) * 144

        func tf(_ n: Int) throws -> GPUTensor { try GPUTensor.floats(rt, [Float](repeating: 0.01, count: n)) }
        func tf16(_ n: Int) throws -> GPUTensor { let a = f16fast(n); return try GPUTensor.bytes(rt, a.withUnsafeBytes { Array($0) }, elementCount: n) }
        func tq8(_ r: Int, _ c: Int) throws -> GPUTensor { try GPUTensor.bytes(rt, q8fast(r, c), elementCount: r*c) }
        // Lazy: the expert-gather path never binds these resident slabs, so
        // no physical RAM is committed for the 256-expert placeholders.
        func texp(_ e: Int, _ r: Int, _ inD: Int) throws -> GPUTensor { try GPUTensor.lazyZeros(rt, floatCount: e*r*(inD/256)*144 / 4) }
        func layer(_ il: Int) throws -> LayerWeights {
            var w = try LayerWeights(hcAttnFn: tf16(24*hcDim), attnScale: tf(3), attnBase: tf(24), attnNorm: tf(d.nEmbd),
                qA: tq8(d.qRank, d.nEmbd), qANorm: tf(d.qRank), qB: tq8(d.qDim, d.qRank), kvW: tq8(d.headDim, d.nEmbd),
                kvNorm: tf(d.headDim), attnSinks: tf(d.nHead), attnOutA: tq8(d.nOutGroup*d.nLoraO, d.attnGroupDim), attnOut: tq8(d.nEmbd, d.attnLowDim), hcFfnFn: tf16(24*hcDim), ffnScale: tf(3),
                ffnBase: tf(24), ffnNorm: tf(d.nEmbd), sharedGate: tq8(d.sharedFfn, d.nEmbd), sharedUp: tq8(d.sharedFfn, d.nEmbd),
                sharedDown: tq8(d.nEmbd, d.sharedFfn), routerW: tq8(d.nExperts, d.nEmbd),
                expGate: texp(d.nExperts, d.expertFfn, d.nEmbd), expUp: texp(d.nExperts, d.expertFfn, d.nEmbd),
                expDown: texp(d.nExperts, d.nEmbd, d.expertFfn))
            let ratio = DSV4Shape.compressRatio(layer: il)
            if ratio != 0 {
                let coff = ratio == 4 ? 2 : 1
                let width = coff * d.headDim
                w.compKv = try tf16(width * d.nEmbd)
                w.compGate = try tf16(width * d.nEmbd)
                w.compApe = try tf16(width * ratio)
                w.compNorm = try tf(d.headDim)
            }
            if ratio == 4 {
                let iw = 2 * d.nIndexerHeadDim
                w.idxKv = try tf16(iw * d.nEmbd)
                w.idxGate = try tf16(iw * d.nEmbd)
                w.idxApe = try tf16(iw * ratio)
                w.idxNorm = try tf(d.nIndexerHeadDim)
            }
            return w
        }
        let layers = try (0..<nLayer).map { try layer($0) }
        let embedTable = try tf16(d.vocab * d.nEmbd)
        let oh = OutputHeadWeights(hcFn: try tf16(d.nHC*hcDim), hcScaleScalar: 0.9, hcBase: try tf(d.nHC), norm: try tf(d.nEmbd), head: try tq8(d.vocab, d.nEmbd))
        let rope = RopeParams(nCtxOrig: 4096, freqBase: 10000, freqScale: 1, extFactor: 0, attnFactor: 1, betaFast: 32, betaSlow: 1)
        let tokens = (0..<256).map { ($0 * 37) % d.vocab }

        func run(_ dense: Bool, _ batchAttn: Bool) throws -> (tps: Double, flashRuns: Int, denseRuns: Int, logits: [Float]) {
            setenv("DS4_PREFILL_DENSE_MM", dense ? "1" : "0", 1)
            setenv("DS4_PREFILL_BATCH_ATTN", batchAttn ? "1" : "0", 1)
            let dec = try StreamingDecoder(rt: rt, dims: d, rope: rope, nLayers: nLayer,
                                           layerProvider: { layers[$0] }, embedTable: embedTable,
                                           out: oh, maxKeys: maxKeys,
                                           expertGather: { _, ids in
                                               (try GPUTensor.zerosBytes(rt, byteLength: ids.count * expertBytes),
                                                try GPUTensor.zerosBytes(rt, byteLength: ids.count * expertBytes),
                                                try GPUTensor.zerosBytes(rt, byteLength: ids.count * expertBytes))
                                           })
            do {
                _ = try dec.prefill(tokens: Array(tokens.prefix(32)))   // warm-up (PSO compile)
            } catch {
                print("BENCH warm-up FAILED (dense=\(dense) batchAttn=\(batchAttn)): \(error)")
                fflush(stdout)
                throw error
            }
            for c in dec.compStates { try c?.reset(rt) }
            for c in dec.indexStates { try c?.reset(rt) }
            dec.resetProfile()
            let t = Date()
            let logits = try dec.prefill(tokens: tokens)
            let dt = Date().timeIntervalSince(t)
            return (Double(tokens.count) / dt, dec.profile.prefillFlashRuns,
                    dec.profile.prefillDenseRuns, logits)
        }

        let perTok = try run(false, true)
        let dense = try run(true, true)
        let hist = try run(false, false)
        print(String(format: "BENCH route per-token+batchattn: %.1f tok/s (flash %d, dense %d)",
                     perTok.tps, perTok.flashRuns, perTok.denseRuns))
        print(String(format: "BENCH route dense-GEMM:          %.1f tok/s (flash %d, dense %d)",
                     dense.tps, dense.flashRuns, dense.denseRuns))
        print(String(format: "BENCH route historical:          %.1f tok/s (flash %d, dense %d)",
                     hist.tps, hist.flashRuns, hist.denseRuns))
        // Parity at REAL dims (incl. both compressed-layer kinds): the dense
        // route must agree with the historical per-token route within the
        // usual MMA-order tolerance.
        var maxRel: Float = 0
        for i in 0..<hist.logits.count {
            maxRel = max(maxRel, abs(dense.logits[i] - hist.logits[i]) / max(abs(hist.logits[i]), 0.05))
        }
        print(String(format: "BENCH dense-vs-historical logits max rel: %.4f", maxRel))
        XCTAssertLessThan(maxRel, 5e-2, "dense prefill diverges from the per-token route")
    }
}
