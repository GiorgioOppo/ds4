import XCTest
@testable import DS4Metal

/// Phase 9 / Stage A5: validates the real dsv4_misc.metal selection kernels
/// (topk_mask + scatter, sort_i32_rows_asc) vs CPU.
final class MetalSparseSelectTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_misc.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
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
