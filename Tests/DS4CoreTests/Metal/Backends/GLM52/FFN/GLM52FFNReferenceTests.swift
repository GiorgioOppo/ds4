import XCTest
@testable import DS4Metal

/// Device-free tests of the GLM 5.2 FFN oracles. The matvec is reimplemented
/// naively here; silu and rmsNorm are pinned by their own closed-form checks
/// and then reused inside composed expectations — so the composed tests prove
/// compositional properties (operation order, weight placement, sums), not
/// independently recomputed outputs. Closed-form checks are exact; composed
/// paths use small tolerances for summation-order drift.
final class GLM52FFNReferenceTests: XCTestCase {
    private let geometry = GLM52FFNGeometry(
        embeddingWidth: 8,
        denseHiddenWidth: 12,
        expertHiddenWidth: 6,
        expertsUsed: 8
    )

    private func pseudoRandom(count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(state >> 33) / Float(1 << 30) - 1.0
        }
    }

    private func naiveMatvec(_ rows: [Float], _ input: [Float]) -> [Float] {
        let rowCount = rows.count / input.count
        return (0..<rowCount).map { row in
            var accumulator: Float = 0
            for i in 0..<input.count {
                accumulator += rows[row * input.count + i] * input[i]
            }
            return accumulator
        }
    }

    func testV52GeometryMatchesArchitectureConstants() {
        let g = GLM52FFNGeometry.v5_2
        XCTAssertEqual(g.embeddingWidth, 6_144)
        XCTAssertEqual(g.denseHiddenWidth, 12_288)
        XCTAssertEqual(g.expertHiddenWidth, 2_048)
        XCTAssertEqual(g.expertsUsed, 8)
    }

    func testSiluAndSwiGLUClosedFormValues() throws {
        XCTAssertEqual(GLM52FFNCPUReference.silu(0), 0)
        XCTAssertEqual(GLM52FFNCPUReference.silu(1), 1 / (1 + exp(-1.0 as Float)),
                       accuracy: 1e-6)
        // Stable at extremes: no NaN, saturates to 0 / identity.
        XCTAssertEqual(GLM52FFNCPUReference.silu(-100), 0, accuracy: 1e-6)
        XCTAssertEqual(GLM52FFNCPUReference.silu(100), 100, accuracy: 1e-3)

        // GLM has NO clamp: a gate far above DeepSeek's clamp of 10 passes
        // through silu untouched.
        let out = try GLM52FFNCPUReference.swiGLU(gate: [50, -2], up: [3, 4])
        XCTAssertEqual(out[0], GLM52FFNCPUReference.silu(50) * 3, accuracy: 1e-3)
        XCTAssertEqual(out[1], GLM52FFNCPUReference.silu(-2) * 4, accuracy: 1e-6)
    }

    func testRMSNormMatchesClosedForm() throws {
        // All-equal input: scale = 1/sqrt(c^2 + eps) exactly.
        let c: Float = 2
        let out = try GLM52FFNCPUReference.rmsNorm(
            [c, c, c, c], weight: [1, 1, 0.5, 2])
        let scale = 1 / (c * c + 1e-5).squareRoot()
        XCTAssertEqual(out[0], c * scale, accuracy: 1e-7)
        XCTAssertEqual(out[2], c * scale * 0.5, accuracy: 1e-7)
        XCTAssertEqual(out[3], c * scale * 2, accuracy: 1e-7)
    }

    func testMatvecIdentityRowsReturnInput() throws {
        let input = pseudoRandom(count: 4, seed: 3)
        var identity = [Float](repeating: 0, count: 16)
        for i in 0..<4 { identity[i * 4 + i] = 1 }
        let out = try GLM52FFNCPUReference.matvec(
            rows: identity, input: input, rowCount: 4)
        XCTAssertEqual(out, input)
    }

    func testDenseFFNMatchesNaiveComposition() throws {
        let g = geometry
        let input = pseudoRandom(count: g.embeddingWidth, seed: 11)
        let gate = pseudoRandom(
            count: g.denseHiddenWidth * g.embeddingWidth, seed: 12)
        let up = pseudoRandom(
            count: g.denseHiddenWidth * g.embeddingWidth, seed: 13)
        let down = pseudoRandom(
            count: g.embeddingWidth * g.denseHiddenWidth, seed: 14)

        let out = try GLM52FFNCPUReference.denseFFN(
            geometry: g, input: input, gate: gate, up: up, down: down)

        let gateOut = naiveMatvec(gate, input)
        let upOut = naiveMatvec(up, input)
        let mid = (0..<g.denseHiddenWidth).map {
            GLM52FFNCPUReference.silu(gateOut[$0]) * upOut[$0]
        }
        let expected = naiveMatvec(down, mid)
        for i in 0..<out.count {
            XCTAssertEqual(out[i], expected[i], accuracy: 1e-5)
        }
    }

    func testRoutedFFNAppliesWeightBeforeDownAndSums() throws {
        let g = geometry
        let input = pseudoRandom(count: g.embeddingWidth, seed: 21)
        var experts = [GLM52ExpertWeights]()
        for e in 0..<g.expertsUsed {
            experts.append(GLM52ExpertWeights(
                gate: pseudoRandom(
                    count: g.expertHiddenWidth * g.embeddingWidth,
                    seed: 100 + UInt64(e)),
                up: pseudoRandom(
                    count: g.expertHiddenWidth * g.embeddingWidth,
                    seed: 200 + UInt64(e)),
                down: pseudoRandom(
                    count: g.embeddingWidth * g.expertHiddenWidth,
                    seed: 300 + UInt64(e))))
        }
        let weights = (0..<g.expertsUsed).map { Float($0 + 1) * 0.25 }

        let out = try GLM52FFNCPUReference.routedFFN(
            geometry: g, input: input, experts: experts, weights: weights)

        var expected = [Float](repeating: 0, count: g.embeddingWidth)
        for (rank, expert) in experts.enumerated() {
            let gateOut = naiveMatvec(expert.gate, input)
            let upOut = naiveMatvec(expert.up, input)
            let mid = (0..<g.expertHiddenWidth).map {
                GLM52FFNCPUReference.silu(gateOut[$0]) * upOut[$0] *
                    weights[rank]
            }
            let contribution = naiveMatvec(expert.down, mid)
            for i in 0..<expected.count { expected[i] += contribution[i] }
        }
        for i in 0..<out.count {
            XCTAssertEqual(out[i], expected[i], accuracy: 1e-4)
        }
    }

    func testSparseFFNIsRoutedPlusShared() throws {
        let g = geometry
        let input = pseudoRandom(count: g.embeddingWidth, seed: 31)
        let experts = (0..<g.expertsUsed).map { e in
            GLM52ExpertWeights(
                gate: pseudoRandom(
                    count: g.expertHiddenWidth * g.embeddingWidth,
                    seed: 400 + UInt64(e)),
                up: pseudoRandom(
                    count: g.expertHiddenWidth * g.embeddingWidth,
                    seed: 500 + UInt64(e)),
                down: pseudoRandom(
                    count: g.embeddingWidth * g.expertHiddenWidth,
                    seed: 600 + UInt64(e)))
        }
        let weights = [Float](repeating: 0.3125, count: g.expertsUsed)
        let sharedGate = pseudoRandom(
            count: g.expertHiddenWidth * g.embeddingWidth, seed: 41)
        let sharedUp = pseudoRandom(
            count: g.expertHiddenWidth * g.embeddingWidth, seed: 42)
        let sharedDown = pseudoRandom(
            count: g.embeddingWidth * g.expertHiddenWidth, seed: 43)

        let sparse = try GLM52FFNCPUReference.sparseFFN(
            geometry: g, input: input, experts: experts, weights: weights,
            sharedGate: sharedGate, sharedUp: sharedUp, sharedDown: sharedDown)
        let routed = try GLM52FFNCPUReference.routedFFN(
            geometry: g, input: input, experts: experts, weights: weights)
        let shared = try GLM52FFNCPUReference.sharedFFN(
            geometry: g, input: input, gate: sharedGate, up: sharedUp,
            down: sharedDown)

        for i in 0..<sparse.count {
            XCTAssertEqual(sparse[i], routed[i] + shared[i], accuracy: 1e-6)
        }
    }

    func testRouterWeightsFeedRoutedFFNUnchanged() throws {
        // Integration with the router oracle: its weights (already ×2.5
        // normalized) enter routedFFN as-is — the ×2.5 must survive.
        let logits = pseudoRandom(count: 256, seed: 51)
        let bias = pseudoRandom(count: 256, seed: 52).map { $0 * 0.01 }
        let routing = try GLM52RouterReference.route(logits: logits, bias: bias)
        XCTAssertEqual(routing.weights.count, 8)
        XCTAssertEqual(routing.weights.reduce(0, +), 2.5, accuracy: 1e-4)

        let g = geometry
        let input = pseudoRandom(count: g.embeddingWidth, seed: 53)
        let experts = (0..<g.expertsUsed).map { e in
            GLM52ExpertWeights(
                gate: pseudoRandom(
                    count: g.expertHiddenWidth * g.embeddingWidth,
                    seed: 700 + UInt64(e)),
                up: pseudoRandom(
                    count: g.expertHiddenWidth * g.embeddingWidth,
                    seed: 800 + UInt64(e)),
                down: pseudoRandom(
                    count: g.embeddingWidth * g.expertHiddenWidth,
                    seed: 900 + UInt64(e)))
        }
        let scaled = try GLM52FFNCPUReference.routedFFN(
            geometry: g, input: input, experts: experts,
            weights: routing.weights)
        let unit = try GLM52FFNCPUReference.routedFFN(
            geometry: g, input: input, experts: experts,
            weights: routing.weights.map { $0 / 2.5 })
        for i in 0..<scaled.count {
            XCTAssertEqual(scaled[i], unit[i] * 2.5, accuracy: 1e-4)
        }
    }

    func testOutputHeadIsNormThenVocabMatvec() throws {
        let hidden = pseudoRandom(count: 8, seed: 61)
        let normWeight = pseudoRandom(count: 8, seed: 62).map { $0 + 1.5 }
        let head = pseudoRandom(count: 5 * 8, seed: 63)

        let logits = try GLM52FFNCPUReference.outputHead(
            hidden: hidden, normWeight: normWeight, head: head,
            vocabularySize: 5)
        let normalized = try GLM52FFNCPUReference.rmsNorm(
            hidden, weight: normWeight)
        let expected = naiveMatvec(head, normalized)
        for i in 0..<logits.count {
            XCTAssertEqual(logits[i], expected[i], accuracy: 1e-6)
        }
    }

    func testRejectionsFireBeforeAnyComputation() {
        let g = geometry
        XCTAssertThrowsError(try GLM52FFNCPUReference.denseFFN(
            geometry: g, input: [1, 2], gate: [], up: [], down: []))
        XCTAssertThrowsError(try GLM52FFNCPUReference.routedFFN(
            geometry: g,
            input: [Float](repeating: 0.1, count: g.embeddingWidth),
            experts: [], weights: []))
        XCTAssertThrowsError(try GLM52FFNCPUReference.rmsNorm(
            [1, .nan], weight: [1, 1]))
        XCTAssertThrowsError(try GLM52FFNCPUReference.swiGLU(
            gate: [1], up: [1, 2]))
    }
}
