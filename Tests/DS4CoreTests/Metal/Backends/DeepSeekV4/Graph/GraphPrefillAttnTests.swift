import XCTest
import Foundation
@testable import DS4Metal

/// Validates the BATCHED prefill FlashAttention (flashAttnPrefill: one
/// multi-query MMA dispatch + per-query causal/window/comp mask, the
/// DS4_PREFILL_BATCH_ATTN path) against a CPU softmax reference that applies
/// exactly the per-token decode visibility: raw rows within the SWA window up
/// to the token's own position, comp rows emitted up to that token, plus the
/// per-head attention sink in the softmax denominator.
final class GraphPrefillAttnTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    /// CPU reference for one query row: softmax over the visible keys (F16
    /// rounded, like the GPU staging) with the sink term in the denominator.
    private func cpuAttnRow(q: [Float], qOff: Int, keys: [[Float]], visible: [Bool],
                            sink: Float, headDim: Int) -> [Float] {
        let scale = 1.0 / Float(headDim).squareRoot()
        var s = [Float](repeating: -.infinity, count: keys.count)
        var m = sink   // running max includes the sink logit
        for k in 0..<keys.count where visible[k] {
            var dot: Float = 0
            for d in 0..<headDim { dot += q[qOff + d] * keys[k][d] }
            s[k] = dot * scale
            m = max(m, s[k])
        }
        var sum: Float = expf(sink - m)
        for k in 0..<keys.count where visible[k] { s[k] = expf(s[k] - m); sum += s[k] }
        var out = [Float](repeating: 0, count: headDim)
        for k in 0..<keys.count where visible[k] {
            let w = s[k] / sum
            for d in 0..<headDim { out[d] += w * keys[k][d] }
        }
        return out
    }

    /// Causal + SWA window + comp visibility, kvpad (nKv % 64 != 0) and
    /// bc_mask (nQ % 8 != 0) paths both exercised.
    func testPrefillAttnMatchesCPUReference() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xBA7C
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let headDim = 512, nHead = 4, window = 16
        let nQ = 10, posFirst = 20                       // positions 20..29
        let rawStart = max(0, posFirst + 1 - window)     // 5
        let rawSpan = (posFirst + nQ) - rawStart         // 25
        let rawRows = 64                                 // full cache, no wrap
        // Comp rows: 1 pre-existing + emits at (pos+1)%4==0 (pos 23, 27).
        let nCompVis = (0..<nQ).map { r in 1 + (posFirst + r >= 23 ? 1 : 0) + (posFirst + r >= 27 ? 1 : 0) }
        let nComp = nCompVis[nQ - 1]                     // 3 -> nKv=28, kvpad path
        let nKv = rawSpan + nComp

        let q = (0..<(nQ * nHead * headDim)).map { _ in rnd() }
        var raw = [Float](repeating: 0, count: rawRows * headDim)
        for i in 0..<raw.count { raw[i] = rnd() }
        let comp = (0..<(nComp * headDim)).map { _ in rnd() }
        let sinks = (0..<nHead).map { _ in rnd() * 0.5 }

        let sb = GraphContext.flashPrefillScratchBytes(nHead: nHead, nQ: nQ, maxKv: nKv)
        let qT = try GPUTensor.floats(rt, q)
        let rawT = try GPUTensor.floats(rt, raw)
        let compT = try GPUTensor.floats(rt, comp)
        let sinksT = try GPUTensor.floats(rt, sinks)
        let kvF16 = try GPUTensor.zerosBytes(rt, byteLength: sb.kvF16)
        let mask = try GPUTensor.zerosBytes(rt, byteLength: sb.mask)
        let pad = try GPUTensor.zerosBytes(rt, byteLength: sb.pad)
        let blk = try GPUTensor.zerosBytes(rt, byteLength: sb.blk)
        let heads = try GPUTensor.zerosBytes(rt, byteLength: sb.heads)

        let maskPtr = (mask.buffer.contents() + mask.byteOffset)
            .bindMemory(to: UInt16.self, capacity: nQ * nKv)
        GraphContext.fillPrefillAttnMask(maskPtr, nQ: nQ, posFirst: posFirst,
                                         rawStart: rawStart, rawSpan: rawSpan,
                                         window: window, nCompVis: nCompVis, nComp: nComp)

        let ctx = GraphContext(rt)
        try ctx.begin()
        try ctx.flashAttnPrefill(q: qT, kvF32: rawT, kvF16: kvF16, mask: mask,
                                 sinks: sinksT, pad: pad, blk: blk, heads: heads,
                                 nHead: nHead, nQ: nQ, rawSpan: rawSpan,
                                 rawStartRow: rawStart, comp: compT, nComp: nComp)
        ctx.commit()
        XCTAssertNil(ctx.lastError)

        // F16-rounded key rows in span order (raw window slice, then comp).
        var keys = [[Float]]()
        for c in 0..<rawSpan {
            let base = (rawStart + c) * headDim
            keys.append((0..<headDim).map { Float(Float16(raw[base + $0])) })
        }
        for c in 0..<nComp {
            keys.append((0..<headDim).map { Float(Float16(comp[c * headDim + $0])) })
        }

        let got = heads.floatArray(nQ * nHead * headDim)
        var maxRel: Float = 0
        for r in 0..<nQ {
            let p = posFirst + r
            var visible = [Bool](repeating: false, count: nKv)
            for c in 0..<rawSpan {
                let pc = rawStart + c
                visible[c] = pc <= p && p - pc < window
            }
            for c in 0..<nComp { visible[rawSpan + c] = c < nCompVis[r] }
            for h in 0..<nHead {
                let ref = cpuAttnRow(q: q, qOff: (r * nHead + h) * headDim,
                                     keys: keys, visible: visible,
                                     sink: sinks[h], headDim: headDim)
                let base = (r * nHead + h) * headDim
                for d in 0..<headDim {
                    maxRel = max(maxRel, abs(got[base + d] - ref[d]) / max(abs(ref[d]), 0.05))
                }
            }
        }
        XCTAssertLessThan(maxRel, 2e-2, "batched prefill attention max rel \(maxRel)")
    }

    /// Ring-buffer raw cache: the span wraps the physical rows, exercising the
    /// kernel_dsv4_raw_ring_cpy_f32_f16 staging branch.
    func testPrefillAttnRingWrap() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xBA7D
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let headDim = 512, nHead = 3, window = 12
        let nQ = 4, posFirst = 26                        // positions 26..29
        let rawStart = max(0, posFirst + 1 - window)     // 15
        let rawSpan = (posFirst + nQ) - rawStart         // 15
        let rawRows = 16                                 // ring: physStart 15, wraps
        let nKv = rawSpan

        let q = (0..<(nQ * nHead * headDim)).map { _ in rnd() }
        // Ring content: absolute row pos pc lives at slot pc % rawRows.
        var byPos = [Int: [Float]]()
        var raw = [Float](repeating: 0, count: rawRows * headDim)
        for pc in rawStart..<(posFirst + nQ) {
            let row = (0..<headDim).map { _ in rnd() }
            byPos[pc] = row
            let slot = pc % rawRows
            for d in 0..<headDim { raw[slot * headDim + d] = row[d] }
        }
        let sinks = (0..<nHead).map { _ in rnd() * 0.5 }

        let sb = GraphContext.flashPrefillScratchBytes(nHead: nHead, nQ: nQ, maxKv: nKv)
        let qT = try GPUTensor.floats(rt, q)
        let rawT = try GPUTensor.floats(rt, raw)
        let sinksT = try GPUTensor.floats(rt, sinks)
        let kvF16 = try GPUTensor.zerosBytes(rt, byteLength: sb.kvF16)
        let mask = try GPUTensor.zerosBytes(rt, byteLength: sb.mask)
        let pad = try GPUTensor.zerosBytes(rt, byteLength: sb.pad)
        let blk = try GPUTensor.zerosBytes(rt, byteLength: sb.blk)
        let heads = try GPUTensor.zerosBytes(rt, byteLength: sb.heads)

        let maskPtr = (mask.buffer.contents() + mask.byteOffset)
            .bindMemory(to: UInt16.self, capacity: nQ * nKv)
        GraphContext.fillPrefillAttnMask(maskPtr, nQ: nQ, posFirst: posFirst,
                                         rawStart: rawStart, rawSpan: rawSpan,
                                         window: window, nCompVis: [Int](repeating: 0, count: nQ),
                                         nComp: 0)

        let ctx = GraphContext(rt)
        try ctx.begin()
        try ctx.flashAttnPrefill(q: qT, kvF32: rawT, kvF16: kvF16, mask: mask,
                                 sinks: sinksT, pad: pad, blk: blk, heads: heads,
                                 nHead: nHead, nQ: nQ, rawSpan: rawSpan,
                                 rawStartRow: rawStart, comp: nil, nComp: 0)
        ctx.commit()
        XCTAssertNil(ctx.lastError)

        var keys = [[Float]]()
        for c in 0..<rawSpan {
            let row = byPos[rawStart + c]!
            keys.append(row.map { Float(Float16($0)) })
        }

        let got = heads.floatArray(nQ * nHead * headDim)
        var maxRel: Float = 0
        for r in 0..<nQ {
            let p = posFirst + r
            var visible = [Bool](repeating: false, count: nKv)
            for c in 0..<rawSpan {
                let pc = rawStart + c
                visible[c] = pc <= p && p - pc < window
            }
            for h in 0..<nHead {
                let ref = cpuAttnRow(q: q, qOff: (r * nHead + h) * headDim,
                                     keys: keys, visible: visible,
                                     sink: sinks[h], headDim: headDim)
                let base = (r * nHead + h) * headDim
                for d in 0..<headDim {
                    maxRel = max(maxRel, abs(got[base + d] - ref[d]) / max(abs(ref[d]), 0.05))
                }
            }
        }
        XCTAssertLessThan(maxRel, 2e-2, "ring-wrap batched prefill max rel \(maxRel)")
    }

    /// Pure mask-fill check: exact expected visibility pattern per row.
    func testFillPrefillAttnMaskPattern() {
        let nQ = 3, posFirst = 6, window = 4
        let rawStart = max(0, posFirst + 1 - window)   // 3
        let rawSpan = (posFirst + nQ) - rawStart       // 6 (positions 3..8)
        let nCompVis = [2, 2, 3], nComp = 3
        let nKv = rawSpan + nComp
        var mask = [UInt16](repeating: 0xFFFF, count: nQ * nKv)
        mask.withUnsafeMutableBufferPointer {
            GraphContext.fillPrefillAttnMask($0.baseAddress!, nQ: nQ, posFirst: posFirst,
                                             rawStart: rawStart, rawSpan: rawSpan,
                                             window: window, nCompVis: nCompVis, nComp: nComp)
        }
        let negInf: UInt16 = 0xFC00
        for r in 0..<nQ {
            let p = posFirst + r
            for c in 0..<rawSpan {
                let pc = rawStart + c
                let vis = pc <= p && p - pc < window
                XCTAssertEqual(mask[r * nKv + c], vis ? 0 : negInf, "row \(r) raw col \(c)")
            }
            for c in 0..<nComp {
                XCTAssertEqual(mask[r * nKv + rawSpan + c], c < nCompVis[r] ? 0 : negInf,
                               "row \(r) comp col \(c)")
            }
        }
    }

    func testDSparkNoncausalMaskExposesWholeBlock() {
        let queries = 5, keys = 17
        var mask = [UInt16](repeating: 0xFC00, count: queries * keys)
        mask.withUnsafeMutableBufferPointer {
            GraphContext.fillDSparkNoncausalMask(
                $0.baseAddress!, queryRows: queries, keyRows: keys)
        }
        XCTAssertTrue(mask.allSatisfy { $0 == 0 })
    }
}
