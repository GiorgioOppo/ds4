import DS4Core
import XCTest
@testable import DS4Metal

final class GLM52IndexerScoresTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    private func fixture(tokenCount: Int, rowCount: Int)
        -> (queries: [Float], weights: [Float], keys: [UInt16]) {
        var seed: UInt64 = 0x474C4D35_32494458
        func next() -> Float {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(truncatingIfNeeded: seed >> 32)) / Float(Int32.max)
        }
        let queries = (0..<tokenCount * GLM52IndexerScoresReference.queryWidth).map {
            _ in next() * 0.4
        }
        let weights = (0..<tokenCount * GLM52IndexerScoresReference.headCount).map {
            _ in 0.05 + abs(next())
        }
        let keys = (0..<rowCount * GLM52IndexerScoresReference.headDimension).map {
            _ in Half.bits(next() * 0.4)
        }
        return (queries, weights, keys)
    }

    func testReferenceAppliesReLUPerHead() throws {
        var queries = [Float](
            repeating: 0,
            count: GLM52IndexerScoresReference.queryWidth
        )
        for head in 0..<GLM52IndexerScoresReference.headCount {
            queries[head * GLM52IndexerScoresReference.headDimension] = 1
        }
        let weights = [Float](
            repeating: 1,
            count: GLM52IndexerScoresReference.headCount
        )
        var keys = [UInt16](
            repeating: Half.bits(0),
            count: 2 * GLM52IndexerScoresReference.headDimension
        )
        keys[0] = Half.bits(1)
        keys[GLM52IndexerScoresReference.headDimension] = Half.bits(-1)

        let output = try GLM52IndexerScoresReference.score(
            queries: queries,
            headWeights: weights,
            keyCacheBits: keys,
            pos0: 1,
            scale: 0.25
        )
        XCTAssertEqual(output[0, 0], 8, accuracy: 1e-6)
        XCTAssertEqual(output[0, 1], 0, accuracy: 1e-6)
    }

    func testReferenceMasksEveryCausalFutureRow() throws {
        let input = fixture(tokenCount: 3, rowCount: 6)
        let output = try GLM52IndexerScoresReference.score(
            queries: input.queries,
            headWeights: input.weights,
            keyCacheBits: input.keys,
            pos0: 2,
            scale: 0.125
        )
        for token in 0..<3 {
            for row in 0..<6 {
                if row <= 2 + token {
                    XCTAssertTrue(output[token, row].isFinite)
                } else {
                    XCTAssertEqual(output[token, row], -Float.infinity)
                }
            }
        }
    }

    func testMetalIndexerScoresMatchScalarOracle() throws {
        let runtime = try makeRuntime()
        let input = fixture(tokenCount: 3, rowCount: 7)
        let scale = GLM52IndexerScoresReference.defaultScale
        let expected = try GLM52IndexerScoresReference.score(
            queries: input.queries,
            headWeights: input.weights,
            keyCacheBits: input.keys,
            pos0: 2,
            scale: scale
        )
        let actual = try runtime.glm52IndexerScores(
            queries: input.queries,
            headWeights: input.weights,
            keyCacheBits: input.keys,
            pos0: 2,
            scale: scale
        )

        XCTAssertEqual(actual.tokenCount, expected.tokenCount)
        XCTAssertEqual(actual.rowCount, expected.rowCount)
        for index in expected.scores.indices {
            if expected.scores[index].isInfinite {
                XCTAssertEqual(actual.scores[index], expected.scores[index])
            } else {
                XCTAssertEqual(
                    actual.scores[index],
                    expected.scores[index],
                    accuracy: 5e-4,
                    "score index \(index)"
                )
            }
        }
    }

    func testRejectsWrongGeometryScaleAndCausalRange() throws {
        let input = fixture(tokenCount: 2, rowCount: 3)
        XCTAssertThrowsError(try GLM52IndexerScoresReference.score(
            queries: input.queries,
            headWeights: Array(input.weights.dropLast()),
            keyCacheBits: input.keys,
            pos0: 0,
            scale: 0.1
        ))
        XCTAssertThrowsError(try GLM52IndexerScoresReference.score(
            queries: input.queries,
            headWeights: input.weights,
            keyCacheBits: input.keys,
            pos0: 2,
            scale: 0.1
        ))
        XCTAssertThrowsError(try GLM52IndexerScoresReference.score(
            queries: input.queries,
            headWeights: input.weights,
            keyCacheBits: input.keys,
            pos0: 0,
            scale: .infinity
        ))
    }
}
