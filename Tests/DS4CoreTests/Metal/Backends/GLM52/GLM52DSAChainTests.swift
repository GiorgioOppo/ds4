import DS4Core
import XCTest
@testable import DS4Metal

/// End-to-end composition of the isolated DSA primitives on GPU — indexer
/// scoring, multi-block top-k, then the staged compact attention — compared
/// against the same chain run through the CPU oracles. This is a fixture, not
/// a decode loop: it proves the pieces compose with identical selections and
/// tolerance-equal attention output.
final class GLM52DSAChainTests: XCTestCase {
    private let rowCount = 192
    private let topK = 24
    private let geometry = GLM52AttentionGeometry.v5_2

    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    private struct Fixture {
        // Indexer inputs (32x128 geometry, one query token).
        let indexerQuery: [Float]
        let headWeights: [Float]
        let indexerKeyBits: [UInt16]
        // Attention inputs (v5_2 geometry over the same row count).
        let attentionQuery: [Float]
        let keyB: [Float]
        let valueB: [Float]
        let cacheBits: [UInt16]
        let cacheAsFloat: [Float]
    }

    private func fixture() -> Fixture {
        var seed: UInt64 = 0x474C4D35_32445341
        func next() -> Float {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(truncatingIfNeeded: seed >> 32)) / Float(Int32.max)
        }
        let g = geometry
        let indexerQuery = (0..<GLM52IndexerScoresReference.queryWidth).map {
            _ in next() * 0.4
        }
        let headWeights = (0..<GLM52IndexerScoresReference.headCount).map {
            _ in 0.05 + abs(next())
        }
        let indexerKeyBits = (0..<rowCount * GLM52IndexerScoresReference.headDimension)
            .map { _ in Half.bits(next() * 0.4) }
        let attentionQuery = (0..<g.headCount * g.qkDimension).map { _ in next() * 0.25 }
        let keyB = (0..<g.headCount * g.kvLoraRank * g.nopeDimension).map {
            _ in next() * 0.05
        }
        let valueB = (0..<g.headCount * g.valueDimension * g.kvLoraRank).map {
            _ in next() * 0.05
        }
        let cacheBits = (0..<rowCount * g.cacheRowWidth).map {
            _ in Half.bits(next() * 0.25)
        }
        return Fixture(
            indexerQuery: indexerQuery,
            headWeights: headWeights,
            indexerKeyBits: indexerKeyBits,
            attentionQuery: attentionQuery,
            keyB: keyB,
            valueB: valueB,
            cacheBits: cacheBits,
            cacheAsFloat: cacheBits.map(Half.float)
        )
    }

    func testScoresTopKAttentionChainMatchesOracleChain() throws {
        let runtime = try makeRuntime()
        let f = fixture()

        // GPU chain. pos0 = rowCount - 1 makes every cache row visible to the
        // single query token, so the score row is fully finite.
        let gpuScores = try runtime.glm52IndexerScores(
            queries: f.indexerQuery,
            headWeights: f.headWeights,
            keyCacheBits: f.indexerKeyBits,
            pos0: rowCount - 1)
        let gpuSelection = try runtime.glm52IndexerTopK(
            scores: gpuScores.scores,
            rowCount: rowCount,
            tokenCount: 1,
            topK: topK)
        let gpuOutput = try runtime.glm52CompactAttention(
            query: f.attentionQuery,
            keyB: f.keyB,
            valueB: f.valueB,
            cacheBits: f.cacheBits,
            selection: gpuSelection)

        // Oracle chain on the same F16-rounded inputs.
        let oracleScores = try GLM52IndexerScoresReference.score(
            queries: f.indexerQuery,
            headWeights: f.headWeights,
            keyCacheBits: f.indexerKeyBits,
            pos0: rowCount - 1)
        let oracleSelection = try GLM52IndexerCPUReference.causalTopK(
            scores: oracleScores.scores,
            queryPosition: rowCount - 1,
            topK: topK)
        let oracleOutput = try GLM52AttentionCPUReference.absorbed(
            geometry: geometry,
            query: f.attentionQuery,
            keyB: f.keyB,
            valueB: f.valueB,
            cache: f.cacheAsFloat,
            selection: oracleSelection)

        // Selections must agree exactly (continuous fixtures: no score ties).
        XCTAssertEqual(gpuSelection.map(Int.init), oracleSelection)
        XCTAssertEqual(gpuOutput.count, oracleOutput.count)
        for i in 0..<gpuOutput.count {
            XCTAssertEqual(gpuOutput[i], oracleOutput[i], accuracy: 2e-3,
                           "DSA chain diverges from the oracle chain at \(i)")
        }
    }
}
