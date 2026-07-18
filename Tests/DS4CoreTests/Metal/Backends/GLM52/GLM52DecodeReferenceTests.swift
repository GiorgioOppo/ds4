import DS4Core
import XCTest
@testable import DS4Metal

/// Contract tests of the decode-step CPU oracle on a small parameterized
/// geometry: cache write order and content (normed prefix, RAW tail), the
/// causal fill-range selection with the current token included, the top-k
/// branch, the IndexShare reuse rules, and the position-0 equivalence with
/// the first-token shortcut (where the Q path cancels under a one-row
/// softmax). No Metal device involved.
final class GLM52DecodeReferenceTests: XCTestCase {
    private let geometry = GLM52DecodeGeometry(
        layer: GLM52LayerGeometry(
            embeddingWidth: 64, headCount: 2, kvLoraRank: 16,
            ropeDimension: 8, valueDimension: 12, denseHiddenWidth: 48,
            expertHiddenWidth: 32, expertsUsed: 2),
        qLoraRank: 24, nopeDimension: 24,
        indexerHeadCount: 4, indexerHeadDimension: 16,
        indexerRotationDimension: 8, indexerTopK: 4)

    private func values(_ count: Int, seed: UInt64, scale: Float) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(truncatingIfNeeded: state >> 32))
                / Float(Int32.max) * scale
        }
    }

    private func attentionWeights(seed: UInt64) -> GLM52DecodeAttentionWeightsF32 {
        let g = geometry
        let layer = g.layer
        let headsWidth = layer.headCount * layer.valueDimension
        return GLM52DecodeAttentionWeightsF32(
            attnNorm: values(layer.embeddingWidth, seed: seed, scale: 0.5)
                .map { $0 + 1.5 },
            qA: values(g.qLoraRank * layer.embeddingWidth,
                       seed: seed &+ 1, scale: 0.2),
            qANorm: values(g.qLoraRank, seed: seed &+ 2, scale: 0.5)
                .map { $0 + 1.5 },
            qB: values(g.queryWidth * g.qLoraRank, seed: seed &+ 3, scale: 0.2),
            kvA: values(layer.kvRawWidth * layer.embeddingWidth,
                        seed: seed &+ 4, scale: 0.2),
            kvANorm: values(layer.kvLoraRank, seed: seed &+ 5, scale: 0.5)
                .map { $0 + 1.5 },
            keyB: values(layer.headCount * layer.kvLoraRank * g.nopeDimension,
                         seed: seed &+ 6, scale: 0.2),
            valueB: values(headsWidth * layer.kvLoraRank,
                           seed: seed &+ 7, scale: 0.2),
            attnOutput: values(layer.embeddingWidth * headsWidth,
                               seed: seed &+ 8, scale: 0.2))
    }

    private func indexerWeights(seed: UInt64) -> GLM52DecodeIndexerWeightsF32 {
        let g = geometry
        return GLM52DecodeIndexerWeightsF32(
            key: values(g.indexerHeadDimension * g.layer.embeddingWidth,
                        seed: seed, scale: 0.3),
            keyNorm: values(g.indexerHeadDimension, seed: seed &+ 1,
                            scale: 0.4).map { $0 + 1.2 },
            keyNormBias: values(g.indexerHeadDimension, seed: seed &+ 2,
                                scale: 0.2),
            queryB: values(g.indexerQueryWidth * g.qLoraRank,
                           seed: seed &+ 3, scale: 0.3),
            proj: values(g.indexerHeadCount * g.layer.embeddingWidth,
                         seed: seed &+ 4, scale: 0.3))
    }

    private func input(position: Int) -> [Float] {
        values(geometry.layer.embeddingWidth,
               seed: 900 + UInt64(position), scale: 0.6)
    }

    private func assertClose(_ got: [Float], _ expected: [Float],
                             label: String, tolerance: Float = 2e-3,
                             file: StaticString = #filePath,
                             line: UInt = #line) {
        XCTAssertEqual(got.count, expected.count, file: file, line: line)
        for i in 0..<got.count {
            XCTAssertEqual(got[i], expected[i],
                           accuracy: tolerance + abs(expected[i]) * tolerance,
                           "\(label) diverges at \(i)", file: file, line: line)
        }
    }

    func testPositionZeroMatchesFirstTokenShortcut() throws {
        let weights = attentionWeights(seed: 100)
        var cache = GLM52DecodeCacheStateF32()
        let x = input(position: 0)

        let decode = try GLM52DecodeCPUReference.decodeAttention(
            geometry: geometry, input: x, weights: weights,
            indexer: indexerWeights(seed: 200), reusedSelection: nil,
            cache: &cache, position: 0)
        // One visible row: softmax collapses to 1 and the Q path cancels,
        // so the decode output must reproduce the first-token shortcut up to
        // the F16 rounding of the cached row.
        let shortcut = try GLM52LayerCPUReference.firstTokenAttention(
            geometry: geometry.layer, input: x,
            weights: GLM52LayerAttentionWeightsF32(
                attnNorm: weights.attnNorm, kvA: weights.kvA,
                kvANorm: weights.kvANorm, valueB: weights.valueB,
                attnOutput: weights.attnOutput))

        XCTAssertEqual(decode.selection, [0])
        assertClose(decode.output, shortcut, label: "position-0 attention")
    }

    func testCompactCacheRowPinsNormedPrefixAndRawTail() throws {
        let weights = attentionWeights(seed: 300)
        var cache = GLM52DecodeCacheStateF32()
        let x = input(position: 0)
        _ = try GLM52DecodeCPUReference.decodeAttention(
            geometry: geometry, input: x, weights: weights,
            indexer: indexerWeights(seed: 400), reusedSelection: nil,
            cache: &cache, position: 0)

        // Recompute the projections through the same public primitives and
        // pin WHAT was stored: F16(rmsNorm(prefix)·w) then F16(raw tail) —
        // no norm and no rotation on the tail.
        let layer = geometry.layer
        let attnNorm = try GLM52FFNCPUReference.rmsNorm(
            x, weight: weights.attnNorm)
        let kvRaw = try GLM52FFNCPUReference.matvec(
            rows: weights.kvA, input: attnNorm, rowCount: layer.kvRawWidth)
        let expectedPrefix = try GLM52FFNCPUReference.rmsNorm(
            Array(kvRaw[0..<layer.kvLoraRank]), weight: weights.kvANorm)

        XCTAssertEqual(cache.compactRows.count,
                       layer.kvLoraRank + layer.ropeDimension)
        for i in 0..<layer.kvLoraRank {
            XCTAssertEqual(cache.compactRows[i],
                           Half.float(Half.bits(expectedPrefix[i])),
                           "cached prefix \(i) is not the F16 normed value")
        }
        for i in layer.kvLoraRank..<layer.kvRawWidth {
            XCTAssertEqual(cache.compactRows[i],
                           Half.float(Half.bits(kvRaw[i])),
                           "cached tail \(i) is not the F16 RAW projection")
        }
    }

    func testFillRangeThenTopKSelection() throws {
        let weights = attentionWeights(seed: 500)
        let indexer = indexerWeights(seed: 600)
        var cache = GLM52DecodeCacheStateF32()

        // Positions 0..3: visible <= topK(4) — plain causal range, current
        // token included.
        for position in 0..<4 {
            let step = try GLM52DecodeCPUReference.decodeAttention(
                geometry: geometry, input: input(position: position),
                weights: weights, indexer: indexer, reusedSelection: nil,
                cache: &cache, position: position)
            XCTAssertEqual(step.selection, Array(0...position),
                           "fill range at position \(position)")
        }
        // Position 4: visible(5) > topK(4) — scored top-k over all visible
        // rows, still bounded by the causal range.
        let scored = try GLM52DecodeCPUReference.decodeAttention(
            geometry: geometry, input: input(position: 4), weights: weights,
            indexer: indexer, reusedSelection: nil, cache: &cache, position: 4)
        XCTAssertEqual(scored.selection.count, 4)
        XCTAssertEqual(Set(scored.selection).count, 4)
        XCTAssertTrue(scored.selection.allSatisfy { (0..<5).contains($0) })
        XCTAssertEqual(cache.indexerKeys.count,
                       5 * geometry.indexerHeadDimension)
        XCTAssertEqual(
            cache.compactRows.count,
            5 * (geometry.layer.kvLoraRank + geometry.layer.ropeDimension))
    }

    func testTopKBranchMatchesIndependentComposition() throws {
        let weights = attentionWeights(seed: 700)
        let indexer = indexerWeights(seed: 800)
        var cache = GLM52DecodeCacheStateF32()
        for position in 0..<4 {
            _ = try GLM52DecodeCPUReference.decodeAttention(
                geometry: geometry, input: input(position: position),
                weights: weights, indexer: indexer, reusedSelection: nil,
                cache: &cache, position: position)
        }
        let x = input(position: 4)
        let step = try GLM52DecodeCPUReference.decodeAttention(
            geometry: geometry, input: x, weights: weights, indexer: indexer,
            reusedSelection: nil, cache: &cache, position: 4)

        // Independent composition from the public primitives: indexer query
        // from q_a_norm with PREFIX RoPE, head weights raw from the RAW
        // residual, ReLU on the scaled per-head dot, lowest-index-tie top-k.
        let g = geometry
        let attnNorm = try GLM52FFNCPUReference.rmsNorm(
            x, weight: weights.attnNorm)
        let qRank = try GLM52FFNCPUReference.matvec(
            rows: weights.qA, input: attnNorm, rowCount: g.qLoraRank)
        let qRankNorm = try GLM52FFNCPUReference.rmsNorm(
            qRank, weight: weights.qANorm)
        var indexerQuery = try GLM52FFNCPUReference.matvec(
            rows: indexer.queryB, input: qRankNorm,
            rowCount: g.indexerQueryWidth)
        indexerQuery = try GLM52RopeTailReference.rotatePrefix(
            values: indexerQuery, headCount: g.indexerHeadCount,
            headDimension: g.indexerHeadDimension,
            rotationDimension: g.indexerRotationDimension, position: 4)
        let headWeights = try GLM52FFNCPUReference.matvec(
            rows: indexer.proj, input: x, rowCount: g.indexerHeadCount)
        var scores = [Float](repeating: 0, count: 5)
        for row in 0..<5 {
            var score: Float = 0
            for head in 0..<g.indexerHeadCount {
                var dot: Float = 0
                for i in 0..<g.indexerHeadDimension {
                    dot += indexerQuery[head * g.indexerHeadDimension + i] *
                        cache.indexerKeys[row * g.indexerHeadDimension + i]
                }
                score += max(dot * g.indexerScale, 0) * headWeights[head]
            }
            scores[row] = score
        }
        let expected = try GLM52IndexerCPUReference.causalTopK(
            scores: scores, queryPosition: 4, topK: g.indexerTopK)
        XCTAssertEqual(step.selection, expected)
    }

    func testIndexShareReuseAndContractErrors() throws {
        let fullWeights = attentionWeights(seed: 900)
        let sharedWeights = attentionWeights(seed: 1_000)
        let indexer = indexerWeights(seed: 1_100)
        var fullCache = GLM52DecodeCacheStateF32()
        var sharedCache = GLM52DecodeCacheStateF32()

        // Drive both layers through positions 0..2; the IndexShare layer
        // reuses each selection verbatim and never stores indexer keys.
        for position in 0..<3 {
            let x = input(position: position)
            let full = try GLM52DecodeCPUReference.decodeAttention(
                geometry: geometry, input: x, weights: fullWeights,
                indexer: indexer, reusedSelection: nil,
                cache: &fullCache, position: position)
            let shared = try GLM52DecodeCPUReference.decodeAttention(
                geometry: geometry, input: x, weights: sharedWeights,
                indexer: nil, reusedSelection: full.selection,
                cache: &sharedCache, position: position)
            XCTAssertEqual(shared.selection, full.selection)
            XCTAssertTrue(sharedCache.indexerKeys.isEmpty)
        }
        // Same weights, same caches, IndexShare vs full: identical outputs,
        // because the selection is the only thing the indexer contributes.
        var referenceCache = GLM52DecodeCacheStateF32()
        var reuseCache = GLM52DecodeCacheStateF32()
        let x = input(position: 0)
        let computed = try GLM52DecodeCPUReference.decodeAttention(
            geometry: geometry, input: x, weights: fullWeights,
            indexer: indexer, reusedSelection: nil,
            cache: &referenceCache, position: 0)
        let reused = try GLM52DecodeCPUReference.decodeAttention(
            geometry: geometry, input: x, weights: fullWeights,
            indexer: nil, reusedSelection: computed.selection,
            cache: &reuseCache, position: 0)
        XCTAssertEqual(reused.output, computed.output)
        XCTAssertEqual(reuseCache.compactRows, referenceCache.compactRows)

        // Contract errors.
        var cache = GLM52DecodeCacheStateF32()
        XCTAssertThrowsError(try GLM52DecodeCPUReference.decodeAttention(
            geometry: geometry, input: x, weights: sharedWeights,
            indexer: nil, reusedSelection: nil,
            cache: &cache, position: 0)) {
            XCTAssertEqual($0 as? GLM52DecodeReferenceError,
                           .missingSharedSelection)
        }
        var cache2 = GLM52DecodeCacheStateF32()
        XCTAssertThrowsError(try GLM52DecodeCPUReference.decodeAttention(
            geometry: geometry, input: x, weights: fullWeights,
            indexer: indexer, reusedSelection: [0],
            cache: &cache2, position: 0)) {
            XCTAssertEqual($0 as? GLM52DecodeReferenceError,
                           .unexpectedSharedSelection)
        }
        var polluted = GLM52DecodeCacheStateF32()
        polluted.compactRows = [Float](
            repeating: 0,
            count: geometry.layer.kvLoraRank + geometry.layer.ropeDimension)
        polluted.indexerKeys = [Float](
            repeating: 0, count: geometry.indexerHeadDimension)
        XCTAssertThrowsError(try GLM52DecodeCPUReference.decodeAttention(
            geometry: geometry, input: x, weights: sharedWeights,
            indexer: nil, reusedSelection: [0],
            cache: &polluted, position: 1)) {
            XCTAssertEqual($0 as? GLM52DecodeReferenceError,
                           .indexerCacheOnSharedLayer)
        }
    }

    func testDecodeLayerMatchesFirstTokenLayerAtPositionZero() throws {
        let weights = attentionWeights(seed: 1_200)
        let layer = geometry.layer
        let ffnNorm = values(layer.embeddingWidth, seed: 1_300, scale: 0.5)
            .map { $0 + 1.5 }
        let gate = values(layer.denseHiddenWidth * layer.embeddingWidth,
                          seed: 1_400, scale: 0.2)
        let up = values(layer.denseHiddenWidth * layer.embeddingWidth,
                        seed: 1_500, scale: 0.2)
        let down = values(layer.embeddingWidth * layer.denseHiddenWidth,
                          seed: 1_600, scale: 0.2)
        let x = input(position: 0)

        var cache = GLM52DecodeCacheStateF32()
        let decode = try GLM52DecodeCPUReference.decodeLayer(
            geometry: geometry, input: x, attention: weights,
            indexer: indexerWeights(seed: 1_700), reusedSelection: nil,
            ffnNorm: ffnNorm, ffn: .dense(gate: gate, up: up, down: down),
            cache: &cache, position: 0)
        let firstToken = try GLM52LayerCPUReference.firstTokenLayer(
            geometry: layer, input: x,
            attention: GLM52LayerAttentionWeightsF32(
                attnNorm: weights.attnNorm, kvA: weights.kvA,
                kvANorm: weights.kvANorm, valueB: weights.valueB,
                attnOutput: weights.attnOutput),
            ffnNorm: ffnNorm, ffn: .dense(gate: gate, up: up, down: down))

        XCTAssertNil(decode.routing)
        XCTAssertEqual(decode.selection, [0])
        assertClose(decode.output, firstToken.output,
                    label: "position-0 layer")
    }
}
