import XCTest
import Foundation
@testable import DS4Metal

/// DS4_ADAPTIVE_SPLITK parity: the adaptive split-K dispatch of flashAttnCore
/// (nwg = min(32, ceil(total/32)) instead of the fixed 32) must produce heads
/// BIT-IDENTICAL to the historical dispatch — every active workgroup receives
/// exactly the same key chunks, only the empty partials disappear, and the
/// nwg == 1 case takes the vec kernel's self-normalizing direct write.
/// Covers: nwg==1 (with and without kvpad, with and without comp rows),
/// nwg==2/3/4/5/6/7 (including non-power-of-two guarded reduce lanes), and nwg==32
/// (adaptive == historical, trivially identical).
final class GraphAttnSplitKTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }   // embedded kernels
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    /// One flashAttnCore run at the given split policy -> heads floats.
    private func run(_ rt: MetalRuntime, adaptive: Bool, nHead: Int,
                     q: [Float], raw: [Float], comp: [Float], nKeys: Int, nComp: Int,
                     sinks: [Float]? = nil) throws -> [Float] {
        let headDim = 512
        let total = nKeys + nComp
        let saved = GraphContext.adaptiveSplitK
        GraphContext.adaptiveSplitK = adaptive
        defer { GraphContext.adaptiveSplitK = saved }

        let sb = GraphContext.flashScratchBytes(nHead: nHead, nKeys: total)
        let qT = try GPUTensor.floats(rt, q)
        let rawT = try GPUTensor.floats(rt, raw)
        let compT = nComp > 0 ? try GPUTensor.floats(rt, comp) : nil
        let kvF16 = try GPUTensor.zerosBytes(rt, byteLength: sb.kvF16)
        let mask = try GPUTensor.zerosBytes(rt, byteLength: sb.mask)
        let sinksT = sinks != nil ? try GPUTensor.floats(rt, sinks!)
                                  : try GPUTensor.zerosBytes(rt, byteLength: sb.sinks)
        let pad = try GPUTensor.zerosBytes(rt, byteLength: sb.pad)
        let tmp = try GPUTensor.zerosBytes(rt, byteLength: sb.tmp)
        let heads = try GPUTensor.zeros(rt, floatCount: nHead * headDim)

        let gc = GraphContext(rt); try gc.begin()
        try gc.flashAttnCore(q: qT, kvF32: rawT, kvF16: kvF16, mask: mask, sinks: sinksT,
                             pad: pad, tmp: tmp, heads: heads, nHead: nHead, nKeys: nKeys,
                             hasSinks: sinks != nil, comp: compT, nComp: nComp)
        gc.commit()
        return heads.floatArray(nHead * headDim)
    }

    func testSplitKWorkgroupSelectionUsesExactChunkCount() {
        let cases: [(rows: Int, expected: Int)] = [
            (0, 1),
            (1, 1),
            (31, 1),
            (32, 1),
            (33, 2),
            (95, 3),
            (96, 3),
            (97, 4),
            (128, 4),
            (129, 5),
            (160, 5),
            (161, 6),
            (1024, 32),
            (1025, 32),
        ]

        for testCase in cases {
            XCTAssertEqual(
                GraphContext.splitKWorkgroups(totalRows: testCase.rows, adaptive: true),
                testCase.expected,
                "unexpected split-K workgroup count at \(testCase.rows) rows"
            )
        }
        XCTAssertEqual(GraphContext.splitKWorkgroups(totalRows: 1, adaptive: false), 32)
        XCTAssertEqual(GraphContext.splitKWorkgroups(totalRows: 129, adaptive: false), 32)
    }

    func testAdaptiveSplitKMatchesFixedBitForBit() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x51EE7
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let headDim = 512, nHead = 4
        // (nKeys, nComp, sinks): totals chosen to hit nwg 1 (pad e non), 2,
        // the exact 128/129-row transition, and the historical upper cap.
        let cases: [(nKeys: Int, nComp: Int, withSinks: Bool)] = [
            (20, 5, false),    // total 25  -> nwg 1, kvpad, two-span
            (32, 0, false),    // total 32  -> nwg 1, no pad
            (32, 0, true),     // total 32  -> nwg 1 + sinks (iwg==0 path)
            (33, 0, false),    // total 33  -> nwg 2, kvpad (guarded reduce lanes)
            (64, 0, false),    // total 64  -> nwg 2, no pad
            (65, 30, false),   // total 95  -> nwg 3, two-span
            (100, 28, false),  // total 128 -> nwg 4, two-span
            (101, 28, false),  // total 129 -> first row in chunk 5
            (129, 32, false),  // total 161 -> nwg 6, two-span
            (161, 32, true),   // total 193 -> nwg 7 + sinks
            (1024, 32, true),  // total 1056 -> nchunks 33 -> nwg 32 == storico
        ]
        for c in cases {
            let q = (0..<(nHead * headDim)).map { _ in rnd() }
            let raw = (0..<(c.nKeys * headDim)).map { _ in rnd() }
            let comp = (0..<(max(1, c.nComp) * headDim)).map { _ in rnd() }
            let sinks = c.withSinks ? (0..<nHead).map { _ in rnd() } : nil

            let fixed = try run(rt, adaptive: false, nHead: nHead, q: q, raw: raw, comp: comp,
                                nKeys: c.nKeys, nComp: c.nComp, sinks: sinks)
            let adaptive = try run(rt, adaptive: true, nHead: nHead, q: q, raw: raw, comp: comp,
                                   nKeys: c.nKeys, nComp: c.nComp, sinks: sinks)
            XCTAssertEqual(fixed, adaptive,
                           "split-K adattivo diverge dal fisso a nKeys=\(c.nKeys) nComp=\(c.nComp) sinks=\(c.withSinks)")
        }
    }
}
