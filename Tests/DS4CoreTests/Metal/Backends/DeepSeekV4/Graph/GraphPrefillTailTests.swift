import XCTest
import Foundation
@testable import DS4Metal

/// The MM prefill paths' BATCHED routed tail (sum6 via unit-weight
/// hcWeightedSum + rows-add + one hcExpand4 over the run) must be
/// BIT-IDENTICAL to the historical per-token sequence (moeSum6 + add +
/// hcExpand4 per token): x·1.0 is exact and every accumulation order matches.
final class GraphPrefillTailTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testBatchedRoutedTailMatchesPerToken() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x7A11
        func rf() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let nEmbd = 64, nHC = 4, k = 6, nTok = 5
        let hcDim = nHC * nEmbd
        let down6 = try GPUTensor.floats(rt, (0..<(nTok * k * nEmbd)).map { _ in rf() })
        let sOut = try GPUTensor.floats(rt, (0..<(nTok * nEmbd)).map { _ in rf() })
        let attn = try GPUTensor.floats(rt, (0..<(nTok * hcDim)).map { _ in rf() })
        let split = try GPUTensor.floats(rt, (0..<(nTok * 24)).map { _ in rf() })
        // The unit-weight collapse reads k entries per token (zero token
        // stride): size like production's mm.ones (n >= k always there).
        let ones = try GPUTensor.floats(rt, [Float](repeating: 1, count: max(nTok, k)))

        // ── Batched tail (the encodeRoutedTailBatched sequence).
        let routedMat = try GPUTensor.zeros(rt, floatCount: nTok * nEmbd)
        let outBatch = try GPUTensor.zeros(rt, floatCount: nTok * hcDim)
        let cb = GraphContext(rt); try cb.begin()
        try cb.hcWeightedSum(x: down6, weights: ones, out: routedMat,
                             nEmbd: nEmbd, nHC: k, nTokens: nTok, weightsTokenStride: 0)
        try cb.add(sOut, routedMat, out: routedMat, width: nEmbd, rows: nTok)
        try cb.hcExpand4(blockOut: routedMat, residual: attn, post: split, comb: split,
                         blockAdd: nil, out: outBatch, nEmbd: nEmbd, nTokens: nTok,
                         postByteOffset: 4 * 4, combByteOffset: 8 * 4,
                         splitTokenStride: 24 * 4)
        cb.commit()
        XCTAssertNil(cb.lastError)

        // ── Per-token reference (moeSum6 + add + hcExpand4, one token each).
        let outRef = try GPUTensor.zeros(rt, floatCount: nTok * hcDim)
        let routedTmp = try GPUTensor.zeros(rt, floatCount: nEmbd)
        let ffnTmp = try GPUTensor.zeros(rt, floatCount: nEmbd)
        let cr = GraphContext(rt); try cr.begin()
        for t in 0..<nTok {
            let d6 = down6.subview(byteOffset: t * k * nEmbd * 4,
                                   byteLength: k * nEmbd * 4, count: k * nEmbd)
            let sh = sOut.subview(byteOffset: t * nEmbd * 4,
                                  byteLength: nEmbd * 4, count: nEmbd)
            let res = attn.subview(byteOffset: t * hcDim * 4,
                                   byteLength: hcDim * 4, count: hcDim)
            let sp = split.subview(byteOffset: t * 24 * 4, byteLength: 24 * 4, count: 24)
            let dst = outRef.subview(byteOffset: t * hcDim * 4,
                                     byteLength: hcDim * 4, count: hcDim)
            try cr.moeSum6(experts: d6, out: routedTmp, width: nEmbd)
            try cr.add(sh, routedTmp, out: ffnTmp, width: nEmbd)
            try cr.hcExpand4(blockOut: ffnTmp, residual: res, post: sp, comb: sp,
                             blockAdd: nil, out: dst, nEmbd: nEmbd, nTokens: 1,
                             postByteOffset: 4 * 4, combByteOffset: 8 * 4)
        }
        cr.commit()
        XCTAssertNil(cr.lastError)

        let got = outBatch.floatArray(nTok * hcDim)
        let ref = outRef.floatArray(nTok * hcDim)
        XCTAssertEqual(got, ref, "batched tail must be bit-identical to the per-token tail")
    }
}
