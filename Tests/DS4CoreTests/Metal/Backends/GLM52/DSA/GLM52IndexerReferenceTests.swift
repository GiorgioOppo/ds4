import XCTest
@testable import DS4Metal

final class GLM52IndexerReferenceTests: XCTestCase {
    private func queryAndWeights() -> (query: [Float], weights: [Float]) {
        var query = [Float](repeating: 0,
                            count: GLM52IndexerCPUReference.headCount
                                * GLM52IndexerCPUReference.headDimension)
        var weights = [Float](repeating: 0,
                              count: GLM52IndexerCPUReference.headCount)
        query[0] = 64
        weights[0] = 2
        query[GLM52IndexerCPUReference.headDimension + 1] = 64
        weights[1] = -0.5
        return (query, weights)
    }

    func testWeightedReluDotUsesTheGLMScale() throws {
        let input = queryAndWeights()
        let width = GLM52IndexerCPUReference.headDimension
        var keys = [Float](repeating: 0, count: 4 * width)

        keys[0 * width + 0] = 1       // 2 * ReLU(64 / 64) = 2
        keys[1 * width + 0] = -1      // ReLU(-64 / 64) = 0
        keys[2 * width + 0] = 2       // 2 * ReLU(128 / 64) = 4
        keys[2 * width + 1] = 1       // -0.5 * ReLU(64 / 64) = -0.5
        keys[3 * width + 1] = -5      // negative dot is clamped before weight

        XCTAssertEqual(GLM52IndexerCPUReference.scale, 1.0 / 64.0)
        XCTAssertEqual(
            try GLM52IndexerCPUReference.scores(
                query: input.query,
                headWeights: input.weights,
                keys: keys
            ),
            [2, 0, 3.5, 0]
        )
    }

    func testCausalTopKExcludesFutureRowsAndBreaksTiesByLowestIndex() throws {
        let scores: [Float] = [1, 10, 10, 99, 100]
        XCTAssertEqual(try GLM52IndexerCPUReference.causalTopK(
            scores: scores, queryPosition: 2, topK: 2), [1, 2])
        XCTAssertEqual(try GLM52IndexerCPUReference.causalTopK(
            scores: scores, queryPosition: 0, topK: 2), [0])

        // If all visible rows fit, preserve the natural causal range rather
        // than changing accumulation order based on scores.
        XCTAssertEqual(try GLM52IndexerCPUReference.causalTopK(
            scores: scores, queryPosition: 2, topK: 3), [0, 1, 2])
    }

    func testCausalTopKRejectsPositionOutsideStoredScores() {
        XCTAssertThrowsError(try GLM52IndexerCPUReference.causalTopK(
            scores: [1, 2], queryPosition: 2, topK: 1
        )) { error in
            XCTAssertEqual(
                error as? GLM52IndexerReferenceError,
                .invalidSelection(queryPosition: 2, scoreCount: 2, topK: 1)
            )
        }
        XCTAssertThrowsError(try GLM52IndexerCPUReference.causalTopK(
            scores: [1], queryPosition: .max, topK: 1
        ))
    }

    func testScoreAndSelectUsesOnlyVisibleRows() throws {
        let input = queryAndWeights()
        let width = GLM52IndexerCPUReference.headDimension
        var keys = [Float](repeating: 0, count: 4 * width)
        keys[0] = 1
        keys[width] = 2
        keys[2 * width] = 100 // future at query position 1
        keys[3 * width] = 200 // future at query position 1

        let result = try GLM52IndexerCPUReference.scoreAndSelect(
            query: input.query,
            headWeights: input.weights,
            keys: keys,
            queryPosition: 1,
            topK: 1
        )
        XCTAssertEqual(result.scores, [2, 4, 200, 400])
        XCTAssertEqual(result.selected, [1])
    }

    func testReferenceRejectsWrongShapesAndNonFiniteValues() {
        XCTAssertThrowsError(try GLM52IndexerCPUReference.scores(
            query: [0], headWeights: [0], keys: [0]))

        var input = queryAndWeights()
        input.query[7] = .nan
        let keys = [Float](repeating: 0,
                           count: GLM52IndexerCPUReference.headDimension)
        XCTAssertThrowsError(try GLM52IndexerCPUReference.scores(
            query: input.query, headWeights: input.weights, keys: keys)) { error in
            XCTAssertEqual(
                error as? GLM52IndexerReferenceError,
                .nonFiniteValue(component: "query", index: 7)
            )
        }
    }

    func testIndexShareStateReusesAndThenReplacesSelection() throws {
        var state = GLM52IndexShareState()
        _ = try state.resolve(layer: 0, computedRows: [0])
        _ = try state.resolve(layer: 1, computedRows: [1])
        let layer2 = try state.resolve(layer: 2, computedRows: [2, 4])
        XCTAssertEqual(layer2.sourceLayer, 2)

        for layer in 3...5 {
            XCTAssertEqual(try state.resolve(layer: layer), layer2)
        }

        let layer6 = try state.resolve(layer: 6, computedRows: [1, 3])
        XCTAssertEqual(layer6.sourceLayer, 6)
        XCTAssertEqual(try state.resolve(layer: 7), layer6)
        XCTAssertThrowsError(try state.resolve(layer: 7, computedRows: [9])) { error in
            XCTAssertEqual(
                error as? GLM52IndexerReferenceError,
                .unexpectedComputedSelection(layer: 7)
            )
        }
    }

    func testIndexShareStateRequiresInheritedSelectionForMidSlice() {
        var empty = GLM52IndexShareState()
        XCTAssertThrowsError(try empty.resolve(layer: 3)) { error in
            XCTAssertEqual(
                error as? GLM52IndexerReferenceError,
                .unavailableSharedSelection(layer: 3, expectedSourceLayer: 2)
            )
        }

        var inherited = GLM52IndexShareState(current: .init(
            sourceLayer: 2, rows: [4, 8]))
        XCTAssertEqual(try inherited.resolve(layer: 3).rows, [4, 8])
    }
}
