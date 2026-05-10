import XCTest
@testable import DeepSeekKit

final class AttentionIndicesTests: XCTestCase {

    /// Prefill with seqlen >= windowSize: each query attends to its `windowSize`
    /// preceding positions, with -1 padding when the ring wraps backwards past 0.
    func testSlidingWindowPrefill() {
        let win = 4
        let idxs = AttentionIndices.slidingWindow(windowSize: win, batch: 1,
                                                   seqlen: 6, startPos: 0)
        // Row 0: only position 0 is valid → [0, -1, -1, -1]
        XCTAssertEqual(idxs[0..<win].map { $0 }, [0, -1, -1, -1])
        // Row 3: window covers [0,1,2,3]
        XCTAssertEqual(idxs[3 * win ..< 4 * win].map { $0 }, [0, 1, 2, 3])
        // Row 5: window covers [2,3,4,5]
        XCTAssertEqual(idxs[5 * win ..< 6 * win].map { $0 }, [2, 3, 4, 5])
    }

    /// Decode after the ring has filled: indices wrap around the ring buffer.
    func testSlidingWindowDecodeWrap() {
        let win = 4
        // startPos = 5, ring positions in cyclic order are: from sp+1 (=2) through sp (=1).
        let idxs = AttentionIndices.slidingWindow(windowSize: win, batch: 1,
                                                   seqlen: 1, startPos: 5)
        // sp = 5 % 4 = 1, so indices = [2, 3, 0, 1]
        XCTAssertEqual(idxs.map { $0 }, [2, 3, 0, 1])
    }

    /// Prefill compressed indices respect causality: query s attends only to
    /// compressed tokens strictly before its compress block.
    func testCompressedPrefill() {
        let r = 4
        let (idxs, k) = AttentionIndices.compressed(ratio: r, batch: 1, seqlen: 12,
                                                     startPos: 0, offset: 100)
        XCTAssertEqual(k, 3)   // 12 / 4 = 3
        // Row 3: limit = 4 / 4 = 1, valid indices [100], rest -1.
        XCTAssertEqual(idxs[3 * k ..< 4 * k].map { $0 }, [100, -1, -1])
        // Row 7: limit = 8 / 4 = 2, valid indices [100, 101], -1 last.
        XCTAssertEqual(idxs[7 * k ..< 8 * k].map { $0 }, [100, 101, -1])
        // Row 11: limit = 12 / 4 = 3, all valid.
        XCTAssertEqual(idxs[11 * k ..< 12 * k].map { $0 }, [100, 101, 102])
    }

    func testCompressedDecode() {
        let r = 4
        let (idxs, k) = AttentionIndices.compressed(ratio: r, batch: 1, seqlen: 1,
                                                     startPos: 11, offset: 50)
        XCTAssertEqual(k, 3)   // (11 + 1) / 4 = 3
        XCTAssertEqual(idxs.map { $0 }, [50, 51, 52])
    }
}
