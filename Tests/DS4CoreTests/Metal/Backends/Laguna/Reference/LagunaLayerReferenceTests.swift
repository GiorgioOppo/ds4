import XCTest
import Foundation
import DS4Core
@testable import DS4Metal

/// Deterministic coverage of the Laguna CPU oracles. These references are the
/// correctness boundary the future Metal graph will be judged against, so the
/// tests pin the upstream semantics (softplus head gate, F16 ring KV, NeoX
/// pairs with YaRN only on full-attention blocks, 10-expert GLM-style router)
/// with hand-computable fixtures.
final class LagunaLayerReferenceTests: XCTestCase {
    private func tinySpec(rotationDims: Int = 4, cacheCapacity: Int = 8,
                          extrapolationFactor: Float = 0,
                          frequencyScale: Float = 1) -> LagunaAttentionSpec {
        LagunaAttentionSpec(
            headCount: 4, kvHeadCount: 2, headDim: 4,
            rotationDims: rotationDims, cacheCapacity: cacheCapacity,
            ropeFrequencyBase: 10_000, ropeFrequencyScale: frequencyScale,
            extrapolationFactor: extrapolationFactor, attentionFactor: 1,
            betaFast: 32, betaSlow: 1, ropeOriginalContext: 8_192,
            rmsEpsilon: 1e-6
        )
    }

    // MARK: Specs

    func testLayerSpecsFollowTheHeadCountAlternation() {
        let shape = LagunaShape.s2_1
        let full = LagunaAttentionSpec.spec(forLayer: 0, contextSize: 4_096)
        XCTAssertEqual(full.headCount, 48)
        XCTAssertEqual(full.rotationDims, 64)
        XCTAssertEqual(full.cacheCapacity, 4_096)
        XCTAssertEqual(full.ropeFrequencyBase, 500_000)
        XCTAssertEqual(full.ropeFrequencyScale, 1 / 32)
        XCTAssertEqual(full.extrapolationFactor, 1)
        XCTAssertEqual(full.ropeOriginalContext, 8_192)

        let sliding = LagunaAttentionSpec.spec(forLayer: 1, contextSize: 4_096)
        XCTAssertEqual(sliding.headCount, 72)
        XCTAssertEqual(sliding.rotationDims, 128)
        XCTAssertEqual(sliding.cacheCapacity, 512)
        XCTAssertEqual(sliding.ropeFrequencyBase, 10_000)
        XCTAssertEqual(sliding.extrapolationFactor, 0)
        XCTAssertEqual(sliding.ropeOriginalContext, Int(shape.contextLength))

        for layer in 0..<48 {
            XCTAssertEqual(
                LagunaAttentionSpec.spec(forLayer: layer, contextSize: 1_024).headCount,
                Int(shape.layerHeadCount(layer))
            )
        }
    }

    // MARK: Primitives

    func testRMSNormMatchesHandComputedValues() throws {
        let out = try LagunaLayerReference.rmsNorm([3, 4], weight: [1, 2],
                                                   epsilon: 0)
        // mean square = 12.5, inv = 1/sqrt(12.5)
        let inv = 1 / Float(12.5).squareRoot()
        XCTAssertEqual(out[0], 3 * inv, accuracy: 1e-6)
        XCTAssertEqual(out[1], 4 * inv * 2, accuracy: 1e-6)
    }

    func testSoftplusGateMatchesTheKernelGuard() {
        XCTAssertEqual(LagunaLayerReference.softplusGate(0), log(2), accuracy: 1e-6)
        XCTAssertEqual(LagunaLayerReference.softplusGate(25), 25,
                       "the kernel passes large gates through untouched")
        XCTAssertEqual(LagunaLayerReference.softplusGate(-30), 0, accuracy: 1e-6)
        // Continuity right at the guard boundary.
        XCTAssertEqual(LagunaLayerReference.softplusGate(20),
                       log(1 + exp(20)), accuracy: 1e-3)
    }

    func testSwiGLUHasNoClamp() throws {
        let out = try LagunaLayerReference.swiGLU(gate: [0, 100, -100],
                                                  up: [5, 2, 3])
        XCTAssertEqual(out[0], 0, accuracy: 1e-6)
        XCTAssertEqual(out[1], 200, accuracy: 1e-3,
                       "silu saturates to identity, never clamps")
        XCTAssertEqual(out[2], 0, accuracy: 1e-6)
    }

    // MARK: RoPE

    func testPlainRopeAtPositionZeroIsIdentityAfterNorm() throws {
        let spec = tinySpec()
        let head: [Float] = [1, 2, 3, 4]
        let weight: [Float] = [1, 1, 1, 1]
        let roped = try LagunaLayerReference.headRMSNormRope(
            head, weight: weight, spec: spec, position: 0
        )
        let normed = try LagunaLayerReference.rmsNorm(head, weight: weight,
                                                      epsilon: spec.rmsEpsilon)
        // theta = 0 for every pair at position 0 and mscale = 1 without YaRN.
        for i in 0..<4 {
            XCTAssertEqual(roped[i], normed[i], accuracy: 1e-6)
        }
    }

    func testPlainRopeRotatesNeoXPairsAndPreservesNorm() throws {
        let spec = tinySpec()
        let head: [Float] = [1, 0, 0, 1]
        let weight: [Float] = [1, 1, 1, 1]
        let roped = try LagunaLayerReference.headRMSNormRope(
            head, weight: weight, spec: spec, position: 3
        )
        let normed = try LagunaLayerReference.rmsNorm(head, weight: weight,
                                                      epsilon: spec.rmsEpsilon)
        // Pair 0 couples dims (0, 2) with theta = pos; pair 1 couples (1, 3)
        // with theta = pos * base^(-2/4).
        let theta0 = Float(3)
        let theta1 = Float(3) * pow(spec.ropeFrequencyBase, -2 / Float(4))
        XCTAssertEqual(roped[0], normed[0] * cos(theta0) - normed[2] * sin(theta0),
                       accuracy: 1e-5)
        XCTAssertEqual(roped[2], normed[0] * sin(theta0) + normed[2] * cos(theta0),
                       accuracy: 1e-5)
        XCTAssertEqual(roped[1], normed[1] * cos(theta1) - normed[3] * sin(theta1),
                       accuracy: 1e-5)
        XCTAssertEqual(roped[3], normed[1] * sin(theta1) + normed[3] * cos(theta1),
                       accuracy: 1e-5)
        // A rotation preserves the norm when mscale is 1.
        let before = normed.reduce(0) { $0 + $1 * $1 }
        let after = roped.reduce(0) { $0 + $1 * $1 }
        XCTAssertEqual(before, after, accuracy: 1e-4)
    }

    func testPartialRotationLeavesTheHeadTailUnrotated() throws {
        // rotationDims 2 on a 4-dim head: only pair (0, 1) rotates — NeoX
        // pairs live inside the rotation prefix, dims 2 and 3 stay put.
        let spec = tinySpec(rotationDims: 2)
        let head: [Float] = [1, 2, 3, 4]
        let weight: [Float] = [1, 1, 1, 1]
        let roped = try LagunaLayerReference.headRMSNormRope(
            head, weight: weight, spec: spec, position: 7
        )
        let normed = try LagunaLayerReference.rmsNorm(head, weight: weight,
                                                      epsilon: spec.rmsEpsilon)
        XCTAssertEqual(roped[2], normed[2], accuracy: 1e-6)
        XCTAssertEqual(roped[3], normed[3], accuracy: 1e-6)
        XCTAssertNotEqual(roped[0], normed[0])
    }

    func testYarnMagnitudeScaleAppliesOnFullAttentionBlocks() throws {
        // With YaRN active (ext_factor 1, freq_scale 1/32) even position 0
        // multiplies the rotated pairs by mscale = 1 + 0.1*ln(32).
        let spec = tinySpec(extrapolationFactor: 1, frequencyScale: 1 / 32)
        let head: [Float] = [1, 2, 3, 4]
        let weight: [Float] = [1, 1, 1, 1]
        let roped = try LagunaLayerReference.headRMSNormRope(
            head, weight: weight, spec: spec, position: 0
        )
        let normed = try LagunaLayerReference.rmsNorm(head, weight: weight,
                                                      epsilon: spec.rmsEpsilon)
        let mscale = 1 + 0.1 * Foundation.log(Float(32))
        for i in 0..<4 {
            XCTAssertEqual(roped[i], normed[i] * mscale, accuracy: 1e-5)
        }
    }

    func testYarnCorrectionDimsAreOrderedAndBounded() {
        let dims = LagunaLayerReference.ropeYarnCorrectionDims(
            rotationDims: 64, originalContext: 8_192, frequencyBase: 500_000,
            betaFast: 32, betaSlow: 1
        )
        XCTAssertGreaterThanOrEqual(dims.low, 0)
        XCTAssertLessThanOrEqual(dims.high, 63)
        XCTAssertLessThanOrEqual(dims.low, dims.high)
    }

    // MARK: Router

    func testRouterSelectsTenByBiasedScoreAndWeightsByUnbiasedSigmoid() throws {
        var logits = [Float](repeating: -20, count: 256)
        var bias = [Float](repeating: 0, count: 256)
        // Experts 3 and 7 have the top biased scores; expert 5 has a huge
        // bias but the weight must still come from the unbiased sigmoid.
        logits[3] = 4
        logits[7] = 2
        bias[5] = 10
        let route = try LagunaRouterReference.route(logits: logits, bias: bias)

        XCTAssertEqual(route.selected.count, 10)
        XCTAssertEqual(route.selected.first, 5, "biased score dominates selection")
        XCTAssertTrue(route.selected.contains(3))
        XCTAssertTrue(route.selected.contains(7))

        let probabilities = route.probabilities
        var sum: Float = 0
        for expert in route.selected { sum += probabilities[Int(expert)] }
        for (rank, expert) in route.selected.enumerated() {
            XCTAssertEqual(
                route.weights[rank],
                probabilities[Int(expert)] / sum * 2.5,
                accuracy: 1e-6
            )
        }
    }

    func testRouterTiesPreferTheLowerExpertID() throws {
        let logits = [Float](repeating: 0, count: 256)
        let bias = [Float](repeating: 0, count: 256)
        let route = try LagunaRouterReference.route(logits: logits, bias: bias)
        XCTAssertEqual(route.selected, (0..<10).map(Int32.init))
        // Uniform probabilities: every weight is scale / used.
        for weight in route.weights {
            XCTAssertEqual(weight, 2.5 / 10, accuracy: 1e-6)
        }
    }

    // MARK: Attention

    func testSingleKeyAttentionReturnsGatedValue() throws {
        let spec = tinySpec()
        let cache = LagunaReferenceKVCache(capacity: spec.cacheCapacity,
                                           rowWidth: spec.keyValueWidth)
        let keyRow = [Float](repeating: 0.5, count: spec.keyValueWidth)
        let valueRow: [Float] = (0..<spec.keyValueWidth).map { Float($0) }
        cache.store(position: 0, keyRow: keyRow, valueRow: valueRow)

        let query = [Float](repeating: 1, count: spec.queryWidth)
        let gate: [Float] = [0, 1, -1, 2]
        let out = try LagunaLayerReference.attend(
            query: query, gate: gate, cache: cache, position: 0, spec: spec
        )
        // One key: softmax is 1 regardless of the score; each head returns
        // its kv head's value row scaled by softplus(gate).
        let headsPerKV = spec.headCount / spec.kvHeadCount
        for head in 0..<spec.headCount {
            let kvHead = head / headsPerKV
            let gateScale = LagunaLayerReference.softplusGate(gate[head])
            for d in 0..<spec.headDim {
                let expected = cache.values[kvHead * spec.headDim + d] * gateScale
                XCTAssertEqual(out[head * spec.headDim + d], expected,
                               accuracy: 1e-4, "head \(head) dim \(d)")
            }
        }
    }

    func testUniformScoresAverageValuesAndRespectTheRingWindow() throws {
        let spec = tinySpec(cacheCapacity: 4)
        let cache = LagunaReferenceKVCache(capacity: 4,
                                           rowWidth: spec.keyValueWidth)
        // Identical keys make scores uniform; values encode their position.
        for position in 0...5 {
            cache.store(
                position: position,
                keyRow: [Float](repeating: 0.25, count: spec.keyValueWidth),
                valueRow: [Float](repeating: Float(position),
                                  count: spec.keyValueWidth)
            )
        }
        let query = [Float](repeating: 1, count: spec.queryWidth)
        let gate = [Float](repeating: 0, count: spec.headCount)
        let out = try LagunaLayerReference.attend(
            query: query, gate: gate, cache: cache, position: 5, spec: spec
        )
        // Window is the last 4 positions (2, 3, 4, 5): the mean is 3.5,
        // scaled by softplus(0) = ln 2.
        let expected = Float(3.5) * log(2)
        for value in out {
            XCTAssertEqual(value, expected, accuracy: 1e-3)
        }
    }

    func testStoredKVRoundsThroughF16LikeTheGPUCache() {
        let cache = LagunaReferenceKVCache(capacity: 2, rowWidth: 2)
        let exact: Float = 0.1 // not representable in F16
        cache.store(position: 0, keyRow: [exact, 1], valueRow: [exact, 1])
        XCTAssertEqual(cache.keys[0], Half.float(Half.bits(exact)))
        XCTAssertNotEqual(cache.keys[0], exact)
        XCTAssertEqual(cache.keys[1], 1)
    }

    // MARK: Blocks

    func testAttentionBlockAddsResidualAndUsesTheCache() throws {
        let spec = tinySpec()
        let width = 4
        // Identity-ish tiny weights: projections copy the normed input into
        // every head so the single-key case is fully hand-checkable.
        func replicated(_ rows: Int) -> [Float] {
            // Each output row r picks input dimension r % width.
            var w = [Float](repeating: 0, count: rows * width)
            for r in 0..<rows { w[r * width + (r % width)] = 1 }
            return w
        }
        // The output projection maps qDim -> width and keeps only the first
        // `width` head dimensions.
        var outputWeight = [Float](repeating: 0, count: width * spec.queryWidth)
        for r in 0..<width { outputWeight[r * spec.queryWidth + r] = 1 }
        let fixed = LagunaLayerReference.AttentionWeights(
            attnNorm: [1, 1, 1, 1],
            query: replicated(spec.queryWidth),
            key: replicated(spec.keyValueWidth),
            value: replicated(spec.keyValueWidth),
            gate: [Float](repeating: 0, count: spec.headCount * width),
            queryNorm: [1, 1, 1, 1],
            keyNorm: [1, 1, 1, 1],
            output: outputWeight
        )

        let cache = LagunaReferenceKVCache(capacity: spec.cacheCapacity,
                                           rowWidth: spec.keyValueWidth)
        let hidden: [Float] = [1, 2, 3, 4]
        let out = try LagunaLayerReference.attentionBlock(
            hidden: hidden, weights: fixed, cache: cache, position: 0,
            spec: spec, embeddingWidth: width
        )
        XCTAssertEqual(out.count, width)
        // The residual survives: output = hidden + projection of gated heads.
        // With gate logits 0 the gate scale is ln 2 > 0, so out != hidden.
        for i in 0..<width {
            XCTAssertNotEqual(out[i], hidden[i])
        }
        // Cache row 0 was written.
        XCTAssertNotEqual(cache.keys.reduce(0) { $0 + abs($1) }, 0)
    }

    func testDenseFFNBlockMatchesManualComposition() throws {
        let width = 2
        let inter = 3
        let hidden: [Float] = [1, -1]
        let ffnNorm: [Float] = [1, 1]
        let gate: [Float] = [1, 0, 0, 1, 1, 1]      // 3x2
        let up: [Float] = [1, 1, 2, 0, 0, 2]        // 3x2
        let down: [Float] = [1, 0, 0, 0, 1, 0]      // 2x3

        let out = try LagunaLayerReference.denseFFNBlock(
            hidden: hidden, ffnNorm: ffnNorm, gate: gate, up: up, down: down,
            hiddenWidth: width, intermediateWidth: inter, epsilon: 0
        )

        let normed = try LagunaLayerReference.rmsNorm(hidden, weight: ffnNorm,
                                                      epsilon: 0)
        let g = try LagunaLayerReference.matvec(weight: gate, inputWidth: width,
                                                outputWidth: inter, input: normed)
        let u = try LagunaLayerReference.matvec(weight: up, inputWidth: width,
                                                outputWidth: inter, input: normed)
        let mid = try LagunaLayerReference.swiGLU(gate: g, up: u)
        let expected = try LagunaLayerReference.matvec(
            weight: down, inputWidth: inter, outputWidth: width, input: mid
        )
        for i in 0..<width {
            XCTAssertEqual(out[i], hidden[i] + expected[i], accuracy: 1e-5)
        }
    }

    func testMoEBlockWeightsRoutedExpertsAndAddsTheSharedExpert() throws {
        let width = 2
        let expertWidth = 2
        let experts = LagunaRouterReference.expertCount

        // Router: experts 0..9 tie at the top (logit 1); everything else -20.
        var routerWeight = [Float](repeating: 0, count: experts * width)
        for e in 0..<experts {
            routerWeight[e * width] = e < 10 ? 1 : -20
        }
        let routerBias = [Float](repeating: 0, count: experts)

        // Every expert computes the same function so the routed sum equals
        // scale (2.5) times the single-expert output regardless of selection.
        let oneGate: [Float] = [1, 0, 0, 1]
        let oneUp: [Float] = [2, 0, 0, 2]
        let oneDown: [Float] = [1, 0, 0, 1]
        let weights = LagunaLayerReference.MoEWeights(
            ffnNorm: [1, 1],
            routerWeight: routerWeight,
            routerBias: routerBias,
            expertGate: Array(repeating: oneGate, count: experts).flatMap { $0 },
            expertUp: Array(repeating: oneUp, count: experts).flatMap { $0 },
            expertDown: Array(repeating: oneDown, count: experts).flatMap { $0 },
            sharedGate: oneGate,
            sharedUp: oneUp,
            sharedDown: oneDown
        )

        let hidden: [Float] = [1, 2]
        let out = try LagunaLayerReference.moeFFNBlock(
            hidden: hidden, weights: weights, hiddenWidth: width,
            expertWidth: expertWidth, sharedWidth: expertWidth, epsilon: 0
        )

        let normed = try LagunaLayerReference.rmsNorm(hidden, weight: [1, 1],
                                                      epsilon: 0)
        let g = try LagunaLayerReference.matvec(weight: oneGate, inputWidth: width,
                                                outputWidth: expertWidth,
                                                input: normed)
        let u = try LagunaLayerReference.matvec(weight: oneUp, inputWidth: width,
                                                outputWidth: expertWidth,
                                                input: normed)
        let mid = try LagunaLayerReference.swiGLU(gate: g, up: u)
        let one = try LagunaLayerReference.matvec(weight: oneDown,
                                                  inputWidth: expertWidth,
                                                  outputWidth: width, input: mid)
        // Routed weights sum to the 2.5 scale; the shared expert adds one
        // more unweighted copy.
        for i in 0..<width {
            XCTAssertEqual(out[i], hidden[i] + 2.5 * one[i] + one[i],
                           accuracy: 1e-4)
        }
    }

    func testOutputHeadNormalizesThenProjects() throws {
        let hidden: [Float] = [3, 4]
        let outputNorm: [Float] = [1, 1]
        var output = [Float](repeating: 0, count: 3 * 2)
        output[0] = 1          // token 0 reads dim 0
        output[1 * 2 + 1] = 1  // token 1 reads dim 1
        output[2 * 2] = 2      // token 2 reads 2*dim 0

        let logits = try LagunaLayerReference.outputHead(
            hidden: hidden, outputNorm: outputNorm, output: output,
            hiddenWidth: 2, vocabularySize: 3, epsilon: 0
        )
        let normed = try LagunaLayerReference.rmsNorm(hidden, weight: outputNorm,
                                                      epsilon: 0)
        XCTAssertEqual(logits[0], normed[0], accuracy: 1e-6)
        XCTAssertEqual(logits[1], normed[1], accuracy: 1e-6)
        XCTAssertEqual(logits[2], 2 * normed[0], accuracy: 1e-6)
    }
}
