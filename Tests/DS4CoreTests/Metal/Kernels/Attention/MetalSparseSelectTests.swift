import XCTest
@testable import DS4Metal

/// Phase 9 / Stage A5: validates the real dsv4_misc.metal selection kernels
/// (topk_mask + scatter, sort_i32_rows_asc) vs CPU.
final class MetalSparseSelectTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        // Use the embedded source-of-truth so newly added kernels are exercised
        // even when the historical external checkout is absent or stale.
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testIndexerTopKMaskOneMatchesCPUHeap() throws {
        let rt = try makeRuntime()
        let nRaw = 127, nComp = 1800, nScores = 1703, topK = 512
        var seed: UInt64 = 0x1D3E_7E2
        var scores = [Float](repeating: 0, count: nScores)
        for i in 0..<scores.count {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            // Deliberately quantize values so tie-breaking by lower row id is tested.
            scores[i] = Float(Int32(truncatingIfNeeded: seed >> 48) % 97)
        }
        let expected = scores.withUnsafeBufferPointer {
            IndexerSelect.allowedTopK(scores: $0.baseAddress!, count: nScores, k: topK)
        }

        let scoreT = try GPUTensor.floats(rt, scores)
        let maskT = try GPUTensor.zerosBytes(rt, byteLength: (nRaw + nComp) * 2)
        let c = GraphContext(rt); try c.begin()
        try c.indexerTopKMask(scores: scoreT, mask: maskT, nRaw: nRaw,
                              nComp: nComp, nScores: nScores, topK: topK)
        c.commit()

        let p = (maskT.buffer.contents() + maskT.byteOffset)
            .bindMemory(to: UInt16.self, capacity: nRaw + nComp)
        let zero = Float16(0).bitPattern
        let negInf = Float16(-Float.infinity).bitPattern
        for i in 0..<nRaw { XCTAssertEqual(p[i], zero, "raw row \(i)") }
        for i in 0..<nComp {
            let selected = i < nScores && expected[i]
            XCTAssertEqual(p[nRaw + i], selected ? zero : negInf, "compressed row \(i)")
        }
    }

    func testIndexerTopKMaskSupportsProTop1024Without1024ThreadDispatch() throws {
        let rt = try makeRuntime()
        let nRaw = 128, nComp = 2600, nScores = 2417, topK = 1024
        var seed: UInt64 = 0xD54_1024
        var scores = [Float](repeating: 0, count: nScores)
        for i in scores.indices {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            scores[i] = Float(Int32(truncatingIfNeeded: seed >> 47) % 211)
        }
        let expected = scores.withUnsafeBufferPointer {
            IndexerSelect.allowedTopK(scores: $0.baseAddress!, count: nScores, k: topK)
        }

        let scoreT = try GPUTensor.floats(rt, scores)
        let maskT = try GPUTensor.zerosBytes(rt, byteLength: (nRaw + nComp) * 2)
        let c = GraphContext(rt); try c.begin()
        try c.indexerTopKMask(scores: scoreT, mask: maskT, nRaw: nRaw,
                              nComp: nComp, nScores: nScores, topK: topK)
        c.commit()

        let p = (maskT.buffer.contents() + maskT.byteOffset)
            .bindMemory(to: UInt16.self, capacity: nRaw + nComp)
        let zero = Float16(0).bitPattern
        let negInf = Float16(-Float.infinity).bitPattern
        for i in 0..<nRaw { XCTAssertEqual(p[i], zero, "raw row \(i)") }
        for i in 0..<nComp {
            let selected = i < nScores && expected[i]
            XCTAssertEqual(p[nRaw + i], selected ? zero : negInf,
                           "compressed row \(i)")
        }
    }

    func testTopkMaskScatter() throws {
        let rt = try makeRuntime()
        let nTokens = 3, topK = 4, nComp = 20
        // selected comp indices per token (distinct, in range)
        let topk: [Int32] = [
            5, 0, 12, 19,
            3, 3, 7, 1,     // duplicate 3 -> still just sets 0 twice
            18, 9, 2, 15,
        ]
        let mask = try rt.topkMaskAndScatter(topk: topk, nTokens: nTokens, topK: topK, nComp: nComp)
        XCTAssertEqual(mask.count, nTokens * nComp)
        for t in 0..<nTokens {
            var sel = Set<Int>()
            for k in 0..<topK { sel.insert(Int(topk[t*topK+k])) }
            for c in 0..<nComp {
                let v = mask[t*nComp+c]
                if sel.contains(c) { XCTAssertEqual(v, 0, "t=\(t) c=\(c) should be 0") }
                else { XCTAssertEqual(v, -Float.infinity, "t=\(t) c=\(c) should be -inf") }
            }
        }
    }

    func testSortI32RowsAsc() throws {
        let rt = try makeRuntime()
        let nTokens = 4, topK = 8
        var seed: UInt64 = 0x5071
        func nextIdx() -> Int32 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Int32(truncatingIfNeeded: seed >> 40) & 0x3FF }
        var src = [Int32](repeating: 0, count: nTokens * topK)
        for i in 0..<src.count { src[i] = nextIdx() }

        let sorted = try rt.sortI32RowsAsc(src, nTokens: nTokens, topK: topK)
        for t in 0..<nTokens {
            let ref = Array(src[t*topK..<(t+1)*topK]).sorted()
            let got = Array(sorted[t*topK..<(t+1)*topK])
            XCTAssertEqual(got, ref, "sort row \(t)")
        }
    }
}
