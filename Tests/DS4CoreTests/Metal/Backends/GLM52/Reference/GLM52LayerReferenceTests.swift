import XCTest
@testable import DS4Metal

/// Device-free tests of the layer/forward oracles. Composition is judged
/// against the already-pinned FFN primitives (compositional properties:
/// residual structure, norm placement, router wiring, chaining), with the
/// same small tolerances for summation-order drift.
final class GLM52LayerReferenceTests: XCTestCase {
    private let geometry = GLM52LayerGeometry(
        embeddingWidth: 8, headCount: 2, kvLoraRank: 6, ropeDimension: 4,
        valueDimension: 5, denseHiddenWidth: 12, expertHiddenWidth: 6,
        expertsUsed: 8)

    private func pseudoRandom(count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(state >> 33) / Float(1 << 30) - 1.0
        }
    }

    private func attentionWeights(seed: UInt64) -> GLM52LayerAttentionWeightsF32 {
        let g = geometry
        let headsWidth = g.headCount * g.valueDimension
        return GLM52LayerAttentionWeightsF32(
            attnNorm: pseudoRandom(count: g.embeddingWidth, seed: seed).map { $0 + 1.5 },
            kvA: pseudoRandom(count: g.kvRawWidth * g.embeddingWidth,
                              seed: seed &+ 1).map { $0 * 0.2 },
            kvANorm: pseudoRandom(count: g.kvLoraRank, seed: seed &+ 2).map { $0 + 1.5 },
            valueB: pseudoRandom(count: headsWidth * g.kvLoraRank,
                                 seed: seed &+ 3).map { $0 * 0.2 },
            attnOutput: pseudoRandom(count: g.embeddingWidth * headsWidth,
                                     seed: seed &+ 4).map { $0 * 0.2 })
    }

    private func expert(seed: UInt64) -> GLM52ExpertWeights {
        let g = geometry
        return GLM52ExpertWeights(
            gate: pseudoRandom(count: g.expertHiddenWidth * g.embeddingWidth,
                               seed: seed).map { $0 * 0.3 },
            up: pseudoRandom(count: g.expertHiddenWidth * g.embeddingWidth,
                             seed: seed &+ 1).map { $0 * 0.3 },
            down: pseudoRandom(count: g.embeddingWidth * g.expertHiddenWidth,
                               seed: seed &+ 2).map { $0 * 0.3 })
    }

    func testV52GeometryMatchesArchitectureConstants() {
        let g = GLM52LayerGeometry.v5_2
        XCTAssertEqual(g.kvRawWidth, 576)
        XCTAssertEqual(g.ffnGeometry, GLM52FFNGeometry.v5_2)
    }

    func testFirstTokenAttentionIsValueProjectionChain() throws {
        let g = geometry
        let x = pseudoRandom(count: g.embeddingWidth, seed: 3).map { $0 * 0.5 }
        let w = attentionWeights(seed: 10)

        let out = try GLM52LayerCPUReference.firstTokenAttention(
            geometry: g, input: x, weights: w)

        // Manual composition of the pinned primitives: pos-0 attention is the
        // token's own value projection — no Q path, no softmax weighting.
        let normed = try GLM52FFNCPUReference.rmsNorm(x, weight: w.attnNorm)
        let kvRaw = try GLM52FFNCPUReference.matvec(
            rows: w.kvA, input: normed, rowCount: g.kvRawWidth)
        let kvNorm = try GLM52FFNCPUReference.rmsNorm(
            Array(kvRaw[0..<g.kvLoraRank]), weight: w.kvANorm)
        let heads = try GLM52FFNCPUReference.matvec(
            rows: w.valueB, input: kvNorm,
            rowCount: g.headCount * g.valueDimension)
        let expected = try GLM52FFNCPUReference.matvec(
            rows: w.attnOutput, input: heads, rowCount: g.embeddingWidth)
        for i in 0..<out.count {
            XCTAssertEqual(out[i], expected[i], accuracy: 1e-6)
        }
    }

    func testDenseLayerHasPreNormResidualStructure() throws {
        let g = geometry
        let x = pseudoRandom(count: g.embeddingWidth, seed: 21).map { $0 * 0.5 }
        let attn = attentionWeights(seed: 22)
        let ffnNorm = pseudoRandom(count: g.embeddingWidth, seed: 23).map { $0 + 1.5 }
        let gate = pseudoRandom(count: g.denseHiddenWidth * g.embeddingWidth,
                                seed: 24).map { $0 * 0.3 }
        let up = pseudoRandom(count: g.denseHiddenWidth * g.embeddingWidth,
                              seed: 25).map { $0 * 0.3 }
        let down = pseudoRandom(count: g.embeddingWidth * g.denseHiddenWidth,
                                seed: 26).map { $0 * 0.3 }

        let (out, routing) = try GLM52LayerCPUReference.firstTokenLayer(
            geometry: g, input: x, attention: attn, ffnNorm: ffnNorm,
            ffn: .dense(gate: gate, up: up, down: down))
        XCTAssertNil(routing)

        let attnOut = try GLM52LayerCPUReference.firstTokenAttention(
            geometry: g, input: x, weights: attn)
        let afterAttn = (0..<x.count).map { x[$0] + attnOut[$0] }
        let ffnOut = try GLM52FFNCPUReference.denseFFN(
            geometry: g.ffnGeometry,
            input: GLM52FFNCPUReference.rmsNorm(afterAttn, weight: ffnNorm),
            gate: gate, up: up, down: down)
        for i in 0..<out.count {
            XCTAssertEqual(out[i], afterAttn[i] + ffnOut[i], accuracy: 1e-5)
        }
    }

    func testSparseLayerRoutesAndFetchesOnlySelectedExperts() throws {
        let g = geometry
        let x = pseudoRandom(count: g.embeddingWidth, seed: 31).map { $0 * 0.5 }
        let attn = attentionWeights(seed: 32)
        let ffnNorm = pseudoRandom(count: g.embeddingWidth, seed: 33).map { $0 + 1.5 }
        let routerRows = pseudoRandom(
            count: GLM52RouterReference.expertCount * g.embeddingWidth,
            seed: 34)
        let routerBias = pseudoRandom(
            count: GLM52RouterReference.expertCount, seed: 35).map { $0 * 0.01 }
        let shared = expert(seed: 36)

        var fetched = [UInt32]()
        let (out, routing) = try GLM52LayerCPUReference.firstTokenLayer(
            geometry: g, input: x, attention: attn, ffnNorm: ffnNorm,
            ffn: .sparse(routerRows: routerRows, routerBias: routerBias,
                         shared: shared,
                         expertProvider: { id in
                             fetched.append(id)
                             return self.expert(seed: 1_000 + UInt64(id))
                         }))

        // The provider saw exactly the router's selection, in rank order.
        let routed = try XCTUnwrap(routing)
        XCTAssertEqual(fetched, routed.selected.map { UInt32(bitPattern: $0) })

        // The routing is the router oracle on the manually computed logits.
        let attnOut = try GLM52LayerCPUReference.firstTokenAttention(
            geometry: g, input: x, weights: attn)
        let afterAttn = (0..<x.count).map { x[$0] + attnOut[$0] }
        let ffnIn = try GLM52FFNCPUReference.rmsNorm(afterAttn, weight: ffnNorm)
        let logits = try GLM52FFNCPUReference.matvec(
            rows: routerRows, input: ffnIn,
            rowCount: GLM52RouterReference.expertCount)
        let expectedRouting = try GLM52RouterReference.route(
            logits: logits, bias: routerBias)
        XCTAssertEqual(routed.selected, expectedRouting.selected)
        XCTAssertEqual(routed.weights, expectedRouting.weights)

        // And the output is afterAttn + sparseFFN with those experts/weights.
        let experts = routed.selected.map {
            self.expert(seed: 1_000 + UInt64(UInt32(bitPattern: $0)))
        }
        let ffnOut = try GLM52FFNCPUReference.sparseFFN(
            geometry: g.ffnGeometry, input: ffnIn, experts: experts,
            weights: routed.weights, sharedGate: shared.gate,
            sharedUp: shared.up, sharedDown: shared.down)
        for i in 0..<out.count {
            XCTAssertEqual(out[i], afterAttn[i] + ffnOut[i], accuracy: 1e-4)
        }
    }

    func testForwardChainsLayersSequentially() throws {
        let g = geometry
        let embedded = pseudoRandom(count: g.embeddingWidth, seed: 41).map { $0 * 0.5 }
        let denseLayer = (
            attention: attentionWeights(seed: 42),
            ffnNorm: pseudoRandom(count: g.embeddingWidth, seed: 43).map { $0 + 1.5 },
            ffn: GLM52LayerFFNWeightsF32.dense(
                gate: pseudoRandom(count: g.denseHiddenWidth * g.embeddingWidth,
                                   seed: 44).map { $0 * 0.3 },
                up: pseudoRandom(count: g.denseHiddenWidth * g.embeddingWidth,
                                 seed: 45).map { $0 * 0.3 },
                down: pseudoRandom(count: g.embeddingWidth * g.denseHiddenWidth,
                                   seed: 46).map { $0 * 0.3 }))
        let sparseLayer = (
            attention: attentionWeights(seed: 52),
            ffnNorm: pseudoRandom(count: g.embeddingWidth, seed: 53).map { $0 + 1.5 },
            ffn: GLM52LayerFFNWeightsF32.sparse(
                routerRows: pseudoRandom(
                    count: GLM52RouterReference.expertCount * g.embeddingWidth,
                    seed: 54),
                routerBias: pseudoRandom(
                    count: GLM52RouterReference.expertCount,
                    seed: 55).map { $0 * 0.01 },
                shared: expert(seed: 56),
                expertProvider: { self.expert(seed: 2_000 + UInt64($0)) }))

        let forward = try GLM52LayerCPUReference.firstTokenForward(
            geometry: g, embeddedToken: embedded,
            layers: [denseLayer, sparseLayer])

        var manual = embedded
        manual = try GLM52LayerCPUReference.firstTokenLayer(
            geometry: g, input: manual, attention: denseLayer.attention,
            ffnNorm: denseLayer.ffnNorm, ffn: denseLayer.ffn).output
        manual = try GLM52LayerCPUReference.firstTokenLayer(
            geometry: g, input: manual, attention: sparseLayer.attention,
            ffnNorm: sparseLayer.ffnNorm, ffn: sparseLayer.ffn).output
        for i in 0..<forward.count {
            XCTAssertEqual(forward[i], manual[i], accuracy: 1e-5)
        }

        // The output head composes on top of the forward hidden state.
        let normWeight = pseudoRandom(count: g.embeddingWidth, seed: 61).map { $0 + 1.5 }
        let head = pseudoRandom(count: 7 * g.embeddingWidth, seed: 62)
        let logits = try GLM52FFNCPUReference.outputHead(
            hidden: forward, normWeight: normWeight, head: head,
            vocabularySize: 7)
        XCTAssertEqual(logits.count, 7)
    }

    func testRejectionsFireOnWrongShapes() {
        let g = geometry
        let x = pseudoRandom(count: g.embeddingWidth, seed: 71)
        var attn = attentionWeights(seed: 72)
        attn = GLM52LayerAttentionWeightsF32(
            attnNorm: attn.attnNorm,
            kvA: Array(attn.kvA.dropLast()),
            kvANorm: attn.kvANorm,
            valueB: attn.valueB,
            attnOutput: attn.attnOutput)
        XCTAssertThrowsError(try GLM52LayerCPUReference.firstTokenAttention(
            geometry: g, input: x, weights: attn))
        XCTAssertThrowsError(try GLM52LayerCPUReference.firstTokenAttention(
            geometry: g, input: [1, 2], weights: attentionWeights(seed: 73)))
    }
}
