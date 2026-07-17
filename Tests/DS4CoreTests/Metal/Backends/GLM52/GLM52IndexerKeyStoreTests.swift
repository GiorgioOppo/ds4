import DS4Core
import XCTest
@testable import DS4Metal

final class GLM52IndexerKeyStoreTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    private func fixture(tokenCount: Int)
        -> (keys: [Float], weight: [Float], bias: [Float]) {
        var keys = [Float](
            repeating: 0,
            count: tokenCount * GLM52IndexerKeyStoreReference.headDimension
        )
        for token in 0..<tokenCount {
            for column in 0..<GLM52IndexerKeyStoreReference.headDimension {
                let sign: Float = (token + column).isMultiple(of: 4) ? -1 : 1
                keys[token * GLM52IndexerKeyStoreReference.headDimension + column] =
                    sign * Float((token + 2) * 13 + column % 23) / 19
            }
        }
        let weight = (0..<GLM52IndexerKeyStoreReference.headDimension).map {
            0.75 + Float($0 % 11) / 29
        }
        let bias = (0..<GLM52IndexerKeyStoreReference.headDimension).map {
            Float(Int($0 % 7) - 3) / 41
        }
        return (keys, weight, bias)
    }

    func testReferenceUsesCenteredLayerNormRatherThanRMSNorm() throws {
        let input = fixture(tokenCount: 1)
        let shifted = input.keys.map { $0 + 17 }
        let base = try GLM52IndexerKeyStoreReference.store(
            rawKeys: input.keys,
            weight: input.weight,
            bias: input.bias,
            pos0: 0,
            cacheCapacity: 1
        )
        let translated = try GLM52IndexerKeyStoreReference.store(
            rawKeys: shifted,
            weight: input.weight,
            bias: input.bias,
            pos0: 0,
            cacheCapacity: 1
        )
        for index in base.indices {
            XCTAssertEqual(
                Half.float(base[index]),
                Half.float(translated[index]),
                accuracy: 0.002,
                "centered normalization mismatch at \(index)"
            )
        }
    }

    func testReferenceRotatesPrefixAndPreservesRowsOutsideWriteRange() throws {
        let input = fixture(tokenCount: 1)
        let sentinel = Half.bits(-9.25)
        let capacity = 4
        let outputAtZero = try GLM52IndexerKeyStoreReference.store(
            rawKeys: input.keys,
            weight: input.weight,
            bias: input.bias,
            pos0: 0,
            cacheCapacity: capacity,
            initialCacheBits: [UInt16](
                repeating: sentinel,
                count: capacity * GLM52IndexerKeyStoreReference.headDimension
            )
        )
        let outputAtTwo = try GLM52IndexerKeyStoreReference.store(
            rawKeys: input.keys,
            weight: input.weight,
            bias: input.bias,
            pos0: 2,
            cacheCapacity: capacity,
            initialCacheBits: [UInt16](
                repeating: sentinel,
                count: capacity * GLM52IndexerKeyStoreReference.headDimension
            )
        )

        for position in [0, 1, 3] {
            XCTAssertTrue(outputAtTwo[
                position * GLM52IndexerKeyStoreReference.headDimension
                    ..< (position + 1) * GLM52IndexerKeyStoreReference.headDimension
            ].allSatisfy { $0 == sentinel })
        }
        let rowAtTwo = 2 * GLM52IndexerKeyStoreReference.headDimension
        XCTAssertNotEqual(outputAtTwo[rowAtTwo], outputAtZero[0])
        for column in GLM52IndexerKeyStoreReference.rotationDimension
            ..< GLM52IndexerKeyStoreReference.headDimension {
            XCTAssertEqual(outputAtTwo[rowAtTwo + column], outputAtZero[column])
        }
    }

    func testMetalIndexerKeyStoreMatchesScalarOracle() throws {
        let runtime = try makeRuntime()
        let input = fixture(tokenCount: 3)
        let capacity = 7
        let initial = [UInt16](
            repeating: Half.bits(6.5),
            count: capacity * GLM52IndexerKeyStoreReference.headDimension
        )
        let expected = try GLM52IndexerKeyStoreReference.store(
            rawKeys: input.keys,
            weight: input.weight,
            bias: input.bias,
            pos0: 2,
            cacheCapacity: capacity,
            initialCacheBits: initial
        )
        let actual = try runtime.glm52StoreIndexerKeys(
            rawKeys: input.keys,
            weight: input.weight,
            bias: input.bias,
            pos0: 2,
            cacheCapacity: capacity,
            initialCacheBits: initial
        )
        XCTAssertEqual(actual.count, expected.count)
        for index in expected.indices {
            XCTAssertEqual(
                Half.float(actual[index]),
                Half.float(expected[index]),
                accuracy: 0.003,
                "cache index \(index)"
            )
        }
    }

    func testRejectsMalformedGeometryAndCapacity() throws {
        let input = fixture(tokenCount: 2)
        XCTAssertThrowsError(try GLM52IndexerKeyStoreReference.store(
            rawKeys: Array(input.keys.dropLast()),
            weight: input.weight,
            bias: input.bias,
            pos0: 0,
            cacheCapacity: 2
        ))
        XCTAssertThrowsError(try GLM52IndexerKeyStoreReference.store(
            rawKeys: input.keys,
            weight: Array(input.weight.dropLast()),
            bias: input.bias,
            pos0: 0,
            cacheCapacity: 2
        ))
        XCTAssertThrowsError(try GLM52IndexerKeyStoreReference.store(
            rawKeys: input.keys,
            weight: input.weight,
            bias: input.bias,
            pos0: 1,
            cacheCapacity: 2
        ))
        XCTAssertThrowsError(try GLM52IndexerKeyStoreReference.store(
            rawKeys: input.keys,
            weight: input.weight,
            bias: input.bias,
            pos0: 0,
            cacheCapacity: 2,
            initialCacheBits: [0]
        ))
    }
}
