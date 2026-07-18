import XCTest
@testable import DS4Metal

/// The compact DSA attention oracle must satisfy the properties the future
/// Metal kernels are judged against: the absorbed (low-rank) order agrees with
/// the expanded textbook order, selection order is irrelevant, and degenerate
/// selections reduce to plain value projections.
final class GLM52AttentionReferenceTests: XCTestCase {
    private let geometry = GLM52AttentionGeometry(
        headCount: 2,
        nopeDimension: 8,
        ropeDimension: 4,
        kvLoraRank: 6,
        valueDimension: 5
    )
    private let rowCount = 10

    /// Deterministic values in [-1, 1]: tests never depend on a global RNG.
    private func pseudoRandom(count: Int, seed: UInt64) -> [Float] {
        var state = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(state >> 33) / Float(1 << 30) - 1.0
        }
    }

    private func fixtures(seed: UInt64 = 9) -> (query: [Float], keyB: [Float],
                                                valueB: [Float], cache: [Float]) {
        let g = geometry
        return (
            query: pseudoRandom(count: g.headCount * g.qkDimension, seed: seed),
            keyB: pseudoRandom(count: g.headCount * g.kvLoraRank * g.nopeDimension,
                               seed: seed &+ 1),
            valueB: pseudoRandom(count: g.headCount * g.valueDimension * g.kvLoraRank,
                                 seed: seed &+ 2),
            cache: pseudoRandom(count: rowCount * g.cacheRowWidth, seed: seed &+ 3)
        )
    }

    func testV52GeometryMatchesArchitectureConstants() {
        let g = GLM52AttentionGeometry.v5_2
        XCTAssertEqual(g.qkDimension, 256)
        XCTAssertEqual(g.cacheRowWidth, 576)
        XCTAssertEqual(g.valueDimension, 256)
        XCTAssertEqual(g.scale, 1.0 / 16.0)
    }

    func testExpandedAndAbsorbedOrdersAgree() throws {
        let f = fixtures()
        let selection = [7, 0, 3, 5, 9]

        let expanded = try GLM52AttentionCPUReference.expanded(
            geometry: geometry, query: f.query, keyB: f.keyB,
            valueB: f.valueB, cache: f.cache, selection: selection)
        let absorbed = try GLM52AttentionCPUReference.absorbed(
            geometry: geometry, query: f.query, keyB: f.keyB,
            valueB: f.valueB, cache: f.cache, selection: selection)

        XCTAssertEqual(expanded.count, geometry.headCount * geometry.valueDimension)
        XCTAssertEqual(absorbed.count, expanded.count)
        for i in 0..<expanded.count {
            XCTAssertEqual(expanded[i], absorbed[i], accuracy: 1e-4,
                           "expanded/absorbed diverge at \(i)")
        }
    }

    func testSelectionOrderIsIrrelevant() throws {
        let f = fixtures(seed: 21)
        let ranked = [8, 2, 6, 0]
        let shuffled = [0, 6, 2, 8]

        // Explicit closures: unapplied function references would not carry
        // the defaulted rotateTailByRowPosition parameter.
        typealias Oracle = (GLM52AttentionGeometry, [Float], [Float],
                            [Float], [Float], [Int]) throws -> [Float]
        let oracles: [Oracle] = [
            { try GLM52AttentionCPUReference.expanded(
                geometry: $0, query: $1, keyB: $2, valueB: $3,
                cache: $4, selection: $5) },
            { try GLM52AttentionCPUReference.absorbed(
                geometry: $0, query: $1, keyB: $2, valueB: $3,
                cache: $4, selection: $5) },
        ]
        for oracle in oracles {
            let a = try oracle(geometry, f.query, f.keyB, f.valueB, f.cache, ranked)
            let b = try oracle(geometry, f.query, f.keyB, f.valueB, f.cache, shuffled)
            for i in 0..<a.count {
                XCTAssertEqual(a[i], b[i], accuracy: 1e-4)
            }
        }
    }

    func testSingleRowSelectionIsThatRowsValueProjection() throws {
        let f = fixtures(seed: 33)
        let g = geometry
        let row = 4

        let output = try GLM52AttentionCPUReference.expanded(
            geometry: g, query: f.query, keyB: f.keyB,
            valueB: f.valueB, cache: f.cache, selection: [row])

        for head in 0..<g.headCount {
            for d in 0..<g.valueDimension {
                var expected: Float = 0
                let rowBase = (head * g.valueDimension + d) * g.kvLoraRank
                for j in 0..<g.kvLoraRank {
                    expected += f.valueB[rowBase + j] *
                        f.cache[row * g.cacheRowWidth + j]
                }
                XCTAssertEqual(output[head * g.valueDimension + d], expected,
                               accuracy: 1e-5)
            }
        }
    }

    func testIdenticalCacheRowsCollapseToOneProjection() throws {
        let f = fixtures(seed: 47)
        let g = geometry
        let rowPattern = pseudoRandom(count: g.cacheRowWidth, seed: 55)
        var uniform = [Float]()
        for _ in 0..<rowCount { uniform.append(contentsOf: rowPattern) }

        let wide = try GLM52AttentionCPUReference.absorbed(
            geometry: g, query: f.query, keyB: f.keyB,
            valueB: f.valueB, cache: uniform, selection: [0, 3, 7, 9])
        let single = try GLM52AttentionCPUReference.absorbed(
            geometry: g, query: f.query, keyB: f.keyB,
            valueB: f.valueB, cache: uniform, selection: [5])

        for i in 0..<wide.count {
            XCTAssertEqual(wide[i], single[i], accuracy: 1e-4)
        }
    }

    func testRejectionsFireBeforeAnyComputation() {
        let f = fixtures(seed: 60)

        XCTAssertThrowsError(try GLM52AttentionCPUReference.expanded(
            geometry: geometry, query: f.query, keyB: f.keyB,
            valueB: f.valueB, cache: f.cache, selection: [])) {
            XCTAssertEqual($0 as? GLM52AttentionReferenceError, .emptySelection)
        }

        XCTAssertThrowsError(try GLM52AttentionCPUReference.expanded(
            geometry: geometry, query: f.query, keyB: f.keyB,
            valueB: f.valueB, cache: f.cache, selection: [0, rowCount])) {
            XCTAssertEqual($0 as? GLM52AttentionReferenceError,
                           .selectionOutOfRange(row: rowCount, rowCount: rowCount))
        }

        XCTAssertThrowsError(try GLM52AttentionCPUReference.absorbed(
            geometry: geometry, query: f.query, keyB: f.keyB,
            valueB: f.valueB, cache: f.cache, selection: [2, 5, 2])) {
            XCTAssertEqual($0 as? GLM52AttentionReferenceError,
                           .duplicateSelectedRow(2))
        }

        XCTAssertThrowsError(try GLM52AttentionCPUReference.absorbed(
            geometry: geometry, query: Array(f.query.dropLast()), keyB: f.keyB,
            valueB: f.valueB, cache: f.cache, selection: [0])) {
            XCTAssertEqual($0 as? GLM52AttentionReferenceError,
                           .invalidDimensions(
                               component: "query",
                               expected: geometry.headCount * geometry.qkDimension,
                               got: geometry.headCount * geometry.qkDimension - 1))
        }

        var poisoned = f.cache
        poisoned[3] = .nan
        XCTAssertThrowsError(try GLM52AttentionCPUReference.expanded(
            geometry: geometry, query: f.query, keyB: f.keyB,
            valueB: f.valueB, cache: poisoned, selection: [0])) {
            XCTAssertEqual($0 as? GLM52AttentionReferenceError,
                           .nonFiniteValue(component: "cache", index: 3))
        }
    }
}
