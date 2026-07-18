import DS4Core
import XCTest
@testable import DS4Metal

/// The decode tail semantics — raw cached K-RoPE tails rotated at attention
/// time with each row's OWN absolute position — must hold across all three
/// implementations: the two oracle evaluation orders and the GPU kernel flag.
/// Host-side pre-rotation of the cache is the independent equivalence anchor.
final class GLM52AttentionRotationTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    private func values(_ count: Int, seed: UInt64, scale: Float) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(truncatingIfNeeded: state >> 32))
                / Float(Int32.max) * scale
        }
    }

    /// Rotate every cache row's tail with the row's own position, host-side,
    /// through the independent RoPE oracle.
    private func preRotated(cache: [Float],
                            geometry: GLM52AttentionGeometry) throws -> [Float] {
        var rotated = cache
        let width = geometry.cacheRowWidth
        for row in 0..<(cache.count / width) {
            let tailBase = row * width + geometry.kvLoraRank
            let tail = try GLM52RopeTailReference.rotate(
                values: Array(cache[tailBase..<tailBase + geometry.ropeDimension]),
                headCount: 1, headDimension: geometry.ropeDimension,
                rotationDimension: geometry.ropeDimension, position: row)
            for i in 0..<geometry.ropeDimension { rotated[tailBase + i] = tail[i] }
        }
        return rotated
    }

    func testOracleOrdersAgreeAndMatchPreRotatedCache() throws {
        let geometry = GLM52AttentionGeometry(
            headCount: 3, nopeDimension: 12, ropeDimension: 8,
            kvLoraRank: 16, valueDimension: 10)
        let rowCount = 24
        let selection = [0, 3, 7, 11, 15, 22, 23]
        let query = values(geometry.headCount * geometry.qkDimension,
                           seed: 5, scale: 0.4)
        let keyB = values(geometry.headCount * geometry.kvLoraRank
                              * geometry.nopeDimension, seed: 6, scale: 0.2)
        let valueB = values(geometry.headCount * geometry.valueDimension
                                * geometry.kvLoraRank, seed: 7, scale: 0.2)
        let cache = values(rowCount * geometry.cacheRowWidth,
                           seed: 8, scale: 0.3)

        let absorbed = try GLM52AttentionCPUReference.absorbed(
            geometry: geometry, query: query, keyB: keyB, valueB: valueB,
            cache: cache, selection: selection, rotateTailByRowPosition: true)
        let expanded = try GLM52AttentionCPUReference.expanded(
            geometry: geometry, query: query, keyB: keyB, valueB: valueB,
            cache: cache, selection: selection, rotateTailByRowPosition: true)
        // Anchor: rotating the cache rows host-side FIRST, then attending
        // without the flag, must describe the same attention.
        let anchored = try GLM52AttentionCPUReference.absorbed(
            geometry: geometry, query: query, keyB: keyB, valueB: valueB,
            cache: preRotated(cache: cache, geometry: geometry),
            selection: selection)

        XCTAssertEqual(absorbed.count, expanded.count)
        for i in 0..<absorbed.count {
            XCTAssertEqual(absorbed[i], expanded[i], accuracy: 1e-4,
                           "absorbed/expanded orders diverge at \(i)")
            XCTAssertEqual(absorbed[i], anchored[i], accuracy: 1e-4,
                           "rotation-at-attention diverges from the "
                           + "pre-rotated anchor at \(i)")
        }
    }

    func testRotationChangesTheResult() throws {
        let geometry = GLM52AttentionGeometry(
            headCount: 2, nopeDimension: 8, ropeDimension: 4,
            kvLoraRank: 8, valueDimension: 6)
        let query = values(geometry.headCount * geometry.qkDimension,
                           seed: 15, scale: 0.5)
        let keyB = values(geometry.headCount * geometry.kvLoraRank
                              * geometry.nopeDimension, seed: 16, scale: 0.3)
        let valueB = values(geometry.headCount * geometry.valueDimension
                                * geometry.kvLoraRank, seed: 17, scale: 0.3)
        let cache = values(6 * geometry.cacheRowWidth, seed: 18, scale: 0.4)
        let selection = [1, 3, 5]

        let raw = try GLM52AttentionCPUReference.absorbed(
            geometry: geometry, query: query, keyB: keyB, valueB: valueB,
            cache: cache, selection: selection)
        let rotated = try GLM52AttentionCPUReference.absorbed(
            geometry: geometry, query: query, keyB: keyB, valueB: valueB,
            cache: cache, selection: selection, rotateTailByRowPosition: true)
        XCTAssertNotEqual(raw, rotated,
                          "row positions > 0 must change the rotated scores")
    }

    func testGPUKernelFlagMatchesOracle() throws {
        let runtime = try makeRuntime()
        let geometry = GLM52AttentionGeometry.v5_2
        let rowCount = 96
        let selection: [UInt32] = [0, 2, 5, 9, 17, 33, 48, 64, 80, 95]
        let query = values(geometry.headCount * geometry.qkDimension,
                           seed: 41, scale: 0.25)
        let keyB = values(geometry.headCount * geometry.kvLoraRank
                              * geometry.nopeDimension, seed: 42, scale: 0.05)
        let valueB = values(geometry.headCount * geometry.valueDimension
                                * geometry.kvLoraRank, seed: 43, scale: 0.05)
        let cacheBits = values(rowCount * geometry.cacheRowWidth,
                               seed: 44, scale: 0.25).map(Half.bits)

        let gpu = try runtime.glm52CompactAttention(
            query: query, keyB: keyB, valueB: valueB, cacheBits: cacheBits,
            selection: selection, rotateTailByRowPosition: true)
        let oracle = try GLM52AttentionCPUReference.absorbed(
            geometry: geometry, query: query, keyB: keyB, valueB: valueB,
            cache: cacheBits.map(Half.float),
            selection: selection.map(Int.init),
            rotateTailByRowPosition: true)

        XCTAssertEqual(gpu.count, oracle.count)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], oracle[i], accuracy: 2e-3,
                           "GPU rotated attention diverges at \(i)")
        }
    }
}
