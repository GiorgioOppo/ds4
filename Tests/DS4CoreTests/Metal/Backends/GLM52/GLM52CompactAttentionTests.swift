import DS4Core
import XCTest
@testable import DS4Metal

/// The staged compact-attention kernels must match the CPU oracle
/// (`GLM52AttentionCPUReference`) on the same F16-rounded cache: stage by
/// stage and chained end to end. Metal comparisons skip without a device; the
/// oracle itself is covered device-free in `DSA/GLM52AttentionReferenceTests`.
final class GLM52CompactAttentionTests: XCTestCase {
    private let geometry = GLM52AttentionGeometry.v5_2
    private let rowCount = 48
    private let selection: [UInt32] = [7, 0, 33, 12, 45, 3, 21, 9]

    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    private struct Fixture {
        let query: [Float]
        let keyB: [Float]
        let valueB: [Float]
        let cacheBits: [UInt16]
        /// The cache exactly as the GPU sees it: F16-rounded values.
        let cacheAsFloat: [Float]
    }

    private func fixture() -> Fixture {
        var seed: UInt64 = 0x474C4D35_32415454
        func next() -> Float {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(truncatingIfNeeded: seed >> 32)) / Float(Int32.max)
        }
        let g = geometry
        let query = (0..<g.headCount * g.qkDimension).map { _ in next() * 0.25 }
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
            query: query,
            keyB: keyB,
            valueB: valueB,
            cacheBits: cacheBits,
            cacheAsFloat: cacheBits.map(Half.float)
        )
    }

    func testQKLowRankMatchesScalarAbsorb() throws {
        let runtime = try makeRuntime()
        let f = fixture()
        let g = geometry

        let gpu = try runtime.glm52QKLowRank(query: f.query, keyB: f.keyB)
        XCTAssertEqual(gpu.count, g.headCount * g.kvLoraRank)

        for head in [0, 17, g.headCount - 1] {
            for j in [0, 255, g.kvLoraRank - 1] {
                var expected: Float = 0
                let rowBase = (head * g.kvLoraRank + j) * g.nopeDimension
                for i in 0..<g.nopeDimension {
                    expected += f.query[head * g.qkDimension + i] * f.keyB[rowBase + i]
                }
                XCTAssertEqual(gpu[head * g.kvLoraRank + j], expected,
                               accuracy: 1e-4,
                               "q_low diverges at head \(head) j \(j)")
            }
        }
    }

    func testValueProjectMatchesScalarDot() throws {
        let runtime = try makeRuntime()
        let f = fixture()
        let g = geometry
        let attnLora = (0..<g.headCount * g.kvLoraRank).map {
            Float($0 % 97) * 0.01 - 0.4
        }

        let gpu = try runtime.glm52ValueProject(attnLora: attnLora, valueB: f.valueB)
        XCTAssertEqual(gpu.count, g.headCount * g.valueDimension)

        for head in [0, 31, g.headCount - 1] {
            for d in [0, 100, g.valueDimension - 1] {
                var expected: Float = 0
                let rowBase = (head * g.valueDimension + d) * g.kvLoraRank
                for j in 0..<g.kvLoraRank {
                    expected += f.valueB[rowBase + j] * attnLora[head * g.kvLoraRank + j]
                }
                XCTAssertEqual(gpu[head * g.valueDimension + d], expected,
                               accuracy: 1e-4,
                               "value projection diverges at head \(head) d \(d)")
            }
        }
    }

    func testChainedStagesMatchAbsorbedOracle() throws {
        let runtime = try makeRuntime()
        let f = fixture()

        let gpu = try runtime.glm52CompactAttention(
            query: f.query,
            keyB: f.keyB,
            valueB: f.valueB,
            cacheBits: f.cacheBits,
            selection: selection)
        let oracle = try GLM52AttentionCPUReference.absorbed(
            geometry: geometry,
            query: f.query,
            keyB: f.keyB,
            valueB: f.valueB,
            cache: f.cacheAsFloat,
            selection: selection.map(Int.init))

        XCTAssertEqual(gpu.count, oracle.count)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], oracle[i], accuracy: 2e-3,
                           "chained attention diverges from the oracle at \(i)")
        }
    }

    func testSingleSelectedRowMatchesExpandedOracle() throws {
        let runtime = try makeRuntime()
        let f = fixture()

        let gpu = try runtime.glm52CompactAttention(
            query: f.query,
            keyB: f.keyB,
            valueB: f.valueB,
            cacheBits: f.cacheBits,
            selection: [11])
        let oracle = try GLM52AttentionCPUReference.expanded(
            geometry: geometry,
            query: f.query,
            keyB: f.keyB,
            valueB: f.valueB,
            cache: f.cacheAsFloat,
            selection: [11])

        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], oracle[i], accuracy: 2e-3)
        }
    }

    func testSelectionValidationRejectsMalformedInput() throws {
        let runtime = try makeRuntime()
        let f = fixture()

        XCTAssertThrowsError(try runtime.glm52CompactAttention(
            query: f.query, keyB: f.keyB, valueB: f.valueB,
            cacheBits: f.cacheBits, selection: []))
        XCTAssertThrowsError(try runtime.glm52CompactAttention(
            query: f.query, keyB: f.keyB, valueB: f.valueB,
            cacheBits: f.cacheBits, selection: [3, 3]))
        XCTAssertThrowsError(try runtime.glm52CompactAttention(
            query: f.query, keyB: f.keyB, valueB: f.valueB,
            cacheBits: f.cacheBits, selection: [UInt32(rowCount)]))
        XCTAssertThrowsError(try runtime.glm52CompactAttention(
            query: Array(f.query.dropLast()), keyB: f.keyB, valueB: f.valueB,
            cacheBits: f.cacheBits, selection: [0]))
    }
}
