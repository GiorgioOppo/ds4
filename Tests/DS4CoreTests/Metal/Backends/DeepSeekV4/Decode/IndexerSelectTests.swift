import XCTest
@testable import DS4Metal

/// IndexerSelect.allowedTopK must select the EXACT same set as the full sort it
/// replaced (order: score DESC, index ASC — ties keep the lowest index). Pure
/// CPU, no Metal device needed.
final class IndexerSelectTests: XCTestCase {

    /// The previous implementation, kept as the reference oracle.
    private func reference(_ scores: [Float], k: Int) -> [Bool] {
        var order = Array(0..<scores.count)
        order.sort { scores[$0] != scores[$1] ? scores[$0] > scores[$1] : $0 < $1 }
        var allowed = [Bool](repeating: false, count: scores.count)
        for i in 0..<min(max(0, k), scores.count) { allowed[order[i]] = true }
        return allowed
    }

    private func heapSelect(_ scores: [Float], k: Int) -> [Bool] {
        scores.withUnsafeBufferPointer {
            IndexerSelect.allowedTopK(scores: $0.baseAddress!, count: scores.count, k: k)
        }
    }

    func testMatchesFullSortReference() {
        // Deterministic xorshift so failures are reproducible.
        var state: UInt64 = 0x1234_5678_9abc_def0
        func next() -> UInt64 { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state }
        for trial in 0..<50 {
            let n = 1 + Int(next() % 3000)
            // Values drawn from a SMALL set so ties are frequent (the hard case).
            let scores = (0..<n).map { _ in Float(next() % 17) - 8.0 }
            let k = Int(next() % UInt64(n + 8))
            XCTAssertEqual(heapSelect(scores, k: k), reference(scores, k: k),
                           "trial \(trial): n=\(n) k=\(k)")
        }
    }

    func testTiesKeepLowestIndex() {
        let scores: [Float] = [1, 1, 1, 1]
        XCTAssertEqual(heapSelect(scores, k: 2), [true, true, false, false])
        // A strictly better late element still wins over early ties.
        XCTAssertEqual(heapSelect([1, 1, 1, 5], k: 2), [true, false, false, true])
    }

    func testEdgeCases() {
        XCTAssertEqual(heapSelect([], k: 4), [])
        XCTAssertEqual(heapSelect([3, 2, 1], k: 0), [false, false, false])
        XCTAssertEqual(heapSelect([3, 2, 1], k: 3), [true, true, true])
        XCTAssertEqual(heapSelect([3, 2, 1], k: 99), [true, true, true])
        XCTAssertEqual(heapSelect([-1, -2, -3], k: 1), [true, false, false])
    }

    func testTypicalIndexerShape() {
        // indexerTopK=512 over a long-context row count.
        let n = 25_000, k = 512
        var state: UInt64 = 42
        func next() -> UInt64 { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state }
        let scores = (0..<n).map { _ in Float(Int(next() % 65536)) / 256.0 }
        let got = heapSelect(scores, k: k)
        XCTAssertEqual(got, reference(scores, k: k))
        XCTAssertEqual(got.filter { $0 }.count, k)
    }
}
