import DS4Core
import XCTest
@testable import DS4Metal

/// The multi-block GPU top-k must reproduce the CPU oracle
/// (`GLM52IndexerReference.causalTopK`) on distinct scores — single-pass and
/// merge path — and must never select causal -infinity rows. Distinct scores
/// sidestep the one intended divergence: the bitonic network does not promise
/// the oracle's lowest-index tie order.
final class GLM52IndexerTopKTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    /// Distinct pseudo-random scores: a seeded shuffle of evenly spaced values,
    /// so every comparison in sort, merge and oracle is strict.
    private func distinctScores(count: Int, seed: UInt64) -> [Float] {
        var values = (0..<count).map { Float($0) * 0.125 - Float(count) * 0.0625 }
        var state = seed
        func next(_ bound: Int) -> Int {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int(state >> 33) % bound
        }
        for i in stride(from: count - 1, to: 0, by: -1) {
            values.swapAt(i, next(i + 1))
        }
        return values
    }

    func testSinglePassMatchesOracle() throws {
        let runtime = try makeRuntime()
        let rowCount = 96
        let topK = 8
        let scores = distinctScores(count: rowCount, seed: 11)

        let gpu = try runtime.glm52IndexerTopK(
            scores: scores, rowCount: rowCount, tokenCount: 1, topK: topK)
        let oracle = try GLM52IndexerReference.causalTopK(
            scores: scores, queryPosition: rowCount - 1, topK: topK)

        XCTAssertEqual(gpu.map(Int.init), oracle)
    }

    func testMultiBlockMergeMatchesOraclePerToken() throws {
        let runtime = try makeRuntime()
        let rowCount = 4_096
        let tokenCount = 3
        let topK = 100
        var scores = [Float]()
        scores.reserveCapacity(rowCount * tokenCount)
        for token in 0..<tokenCount {
            scores.append(contentsOf: distinctScores(
                count: rowCount, seed: 900 + UInt64(token)))
        }

        let gpu = try runtime.glm52IndexerTopK(
            scores: scores, rowCount: rowCount,
            tokenCount: tokenCount, topK: topK)
        XCTAssertEqual(gpu.count, tokenCount * topK)

        for token in 0..<tokenCount {
            let row = Array(scores[token * rowCount..<(token + 1) * rowCount])
            let oracle = try GLM52IndexerReference.causalTopK(
                scores: row, queryPosition: rowCount - 1, topK: topK)
            XCTAssertEqual(
                Array(gpu[token * topK..<(token + 1) * topK]).map(Int.init),
                oracle,
                "token \(token) diverges from the oracle")
        }
    }

    func testCausalInfinityRowsAreNeverSelected() throws {
        let runtime = try makeRuntime()
        let rowCount = 64
        let visible = 40
        let topK = 16
        var scores = distinctScores(count: visible, seed: 77)
        scores.append(contentsOf: [Float](
            repeating: -.infinity, count: rowCount - visible))

        let gpu = try runtime.glm52IndexerTopK(
            scores: scores, rowCount: rowCount, tokenCount: 1, topK: topK)
        XCTAssertTrue(gpu.allSatisfy { Int($0) < visible },
                      "selected a causal future row: \(gpu)")

        let oracle = try GLM52IndexerReference.causalTopK(
            scores: Array(scores[0..<visible]),
            queryPosition: visible - 1,
            topK: topK)
        XCTAssertEqual(gpu.map(Int.init), oracle)
    }

    func testValidationRejectsBadGeometry() throws {
        let runtime = try makeRuntime()
        let scores = distinctScores(count: 32, seed: 5)

        XCTAssertThrowsError(try runtime.glm52IndexerTopK(
            scores: scores, rowCount: 33, tokenCount: 1, topK: 4))
        XCTAssertThrowsError(try runtime.glm52IndexerTopK(
            scores: scores, rowCount: 32, tokenCount: 1, topK: 0))
        XCTAssertThrowsError(try runtime.glm52IndexerTopK(
            scores: scores, rowCount: 32, tokenCount: 1, topK: 33))
        XCTAssertThrowsError(try runtime.glm52IndexerTopK(
            scores: [], rowCount: 0, tokenCount: 1, topK: 1))
    }
}
