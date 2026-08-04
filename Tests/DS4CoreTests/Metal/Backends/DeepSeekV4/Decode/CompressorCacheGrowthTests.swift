import XCTest
@testable import DS4Metal

final class CompressorCacheGrowthTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testGrowthPreservesEmittedRowsWithoutAllocatingLogicalMaximum() throws {
        let rt = try makeRuntime()
        let state = try CompressorState(rt, ratio: 4, headDim: 4,
                                        maxComp: 64, initialComp: 2)
        XCTAssertEqual(state.cacheCapacity, 2)
        XCTAssertEqual(state.maxComp, 64)

        let expected = (0..<8).map { Float($0) + 0.25 }
        _ = expected.withUnsafeBytes {
            memcpy(state.cache.buffer.contents(), $0.baseAddress!, $0.count)
        }
        state.count = 2

        XCTAssertTrue(try state.ensureCacheCapacity(rt, requiredRows: 3))
        XCTAssertEqual(state.cacheCapacity, 4)
        XCTAssertEqual(state.cache.floatArray(expected.count), expected)

        XCTAssertTrue(try state.ensureCacheCapacity(rt, requiredRows: 64))
        XCTAssertEqual(state.cacheCapacity, 64)
        XCTAssertEqual(state.cache.floatArray(expected.count), expected)
        XCTAssertFalse(try state.ensureCacheCapacity(rt, requiredRows: 64))
    }
}
