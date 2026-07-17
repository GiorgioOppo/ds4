import DS4Core
import XCTest
@testable import DS4Metal

final class GLM52CompactKVTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    private func fixture(tokenCount: Int) -> [Float] {
        var rows = [Float](
            repeating: 0,
            count: tokenCount * GLM52CompactKVReference.rawWidth
        )
        for token in 0..<tokenCount {
            for column in 0..<GLM52CompactKVReference.rawWidth {
                let sign: Float = column.isMultiple(of: 3) ? -1 : 1
                rows[token * GLM52CompactKVReference.rawWidth + column] =
                    sign * Float((token + 1) * 1_000 + column) / 257
            }
        }
        return rows
    }

    func testReferenceStoresAtPos0AndPreservesOtherRows() throws {
        let capacity = 5
        let pos0 = 2
        let rows = fixture(tokenCount: 2)
        let kvSentinel = Half.bits(-7.25)
        let ropeSentinel = Half.bits(9.5)
        let initialKV = [UInt16](
            repeating: kvSentinel,
            count: capacity * GLM52CompactKVReference.kvLoRAWidth
        )
        let initialRoPE = [UInt16](
            repeating: ropeSentinel,
            count: capacity * GLM52CompactKVReference.kRoPEWidth
        )

        let output = try GLM52CompactKVReference.store(
            rows: rows,
            pos0: pos0,
            cacheCapacity: capacity,
            initialKVLoRABits: initialKV,
            initialKRoPEBits: initialRoPE
        )

        for position in [0, 1, 4] {
            XCTAssertTrue(output.kvLoRABits[
                position * GLM52CompactKVReference.kvLoRAWidth
                    ..< (position + 1) * GLM52CompactKVReference.kvLoRAWidth
            ].allSatisfy { $0 == kvSentinel })
            XCTAssertTrue(output.kRoPEBits[
                position * GLM52CompactKVReference.kRoPEWidth
                    ..< (position + 1) * GLM52CompactKVReference.kRoPEWidth
            ].allSatisfy { $0 == ropeSentinel })
        }
        XCTAssertEqual(
            output.kvLoRABits[pos0 * GLM52CompactKVReference.kvLoRAWidth + 17],
            Half.bits(rows[17])
        )
        XCTAssertEqual(
            output.kRoPEBits[pos0 * GLM52CompactKVReference.kRoPEWidth + 17],
            Half.bits(rows[GLM52CompactKVReference.kvLoRAWidth + 17])
        )
    }

    func testMetalCompactStoreMatchesHalfBitsExactly() throws {
        let runtime = try makeRuntime()
        let rows = fixture(tokenCount: 3)
        let capacity = 7
        let initialKV = [UInt16](
            repeating: Half.bits(-3),
            count: capacity * GLM52CompactKVReference.kvLoRAWidth
        )
        let initialRoPE = [UInt16](
            repeating: Half.bits(4),
            count: capacity * GLM52CompactKVReference.kRoPEWidth
        )
        let expected = try GLM52CompactKVReference.store(
            rows: rows,
            pos0: 3,
            cacheCapacity: capacity,
            initialKVLoRABits: initialKV,
            initialKRoPEBits: initialRoPE
        )
        let actual = try runtime.glm52StoreCompactKV(
            rows: rows,
            pos0: 3,
            cacheCapacity: capacity,
            initialKVLoRABits: initialKV,
            initialKRoPEBits: initialRoPE
        )
        XCTAssertEqual(actual, expected)
    }

    func testRejectsMalformedRowsAndOutOfCapacityWrites() throws {
        XCTAssertThrowsError(try GLM52CompactKVReference.store(
            rows: [Float](repeating: 0, count: 575),
            pos0: 0,
            cacheCapacity: 1
        ))
        XCTAssertThrowsError(try GLM52CompactKVReference.store(
            rows: fixture(tokenCount: 2),
            pos0: 2,
            cacheCapacity: 3
        ))
        XCTAssertThrowsError(try GLM52CompactKVReference.store(
            rows: fixture(tokenCount: 1),
            pos0: -1,
            cacheCapacity: 3
        ))
        XCTAssertThrowsError(try GLM52CompactKVReference.store(
            rows: fixture(tokenCount: 1),
            pos0: 0,
            cacheCapacity: 3,
            initialKVLoRABits: [0]
        ))
    }
}
