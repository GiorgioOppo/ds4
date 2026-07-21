import XCTest
@testable import DS4Engine

/// Pins the partial-selection scorer the GLM Correctness benchmark uses on
/// every evaluated position: descending order, lowest-id-wins ties (the same
/// rule as the greedy argmax kernels) and full-scan equivalence.
final class GLM52AccuracyCandidatesTests: XCTestCase {
    func testDescendingOrderAndCount() {
        let logits: [Float] = [0.1, 5.0, -2.0, 3.5, 4.9, 0.0]
        XCTAssertEqual(GLM52ChatService.topCandidates(logits, count: 3),
                       [1, 4, 3])
    }

    func testTiesKeepLowestId() {
        let logits: [Float] = [2.0, 7.0, 7.0, 7.0, 1.0]
        XCTAssertEqual(GLM52ChatService.topCandidates(logits, count: 3),
                       [1, 2, 3])
    }

    func testFewerElementsThanRequested() {
        XCTAssertEqual(GLM52ChatService.topCandidates([3.0, 9.0], count: 3),
                       [1, 0])
    }

    func testMatchesFullSortOnLargeVocabulary() {
        var generator = SystemRandomNumberGenerator()
        let logits = (0..<10_000).map { _ in
            Float.random(in: -10...10, using: &generator)
        }
        let expected = logits.enumerated()
            .sorted { $0.element == $1.element
                ? $0.offset < $1.offset
                : $0.element > $1.element }
            .prefix(3).map(\.offset)
        XCTAssertEqual(GLM52ChatService.topCandidates(logits, count: 3),
                       Array(expected))
    }
}
