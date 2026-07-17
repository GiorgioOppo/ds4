import DS4Core
import XCTest
@testable import DS4Metal

final class GLM52KVLoRANormTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    private func fixture(tokenCount: Int) -> (rows: [Float], weight: [Float]) {
        var rows = [Float](
            repeating: 0,
            count: tokenCount * GLM52KVLoRANormReference.rawWidth
        )
        for token in 0..<tokenCount {
            for column in 0..<GLM52KVLoRANormReference.rawWidth {
                let sign: Float = (token + column).isMultiple(of: 3) ? -1 : 1
                rows[token * GLM52KVLoRANormReference.rawWidth + column] =
                    sign * Float((token + 3) * 31 + column % 29) / 37
            }
        }
        let weight = (0..<GLM52KVLoRANormReference.kvLoRAWidth).map {
            0.5 + Float($0 % 17) / 23
        }
        return (rows, weight)
    }

    func testReferenceNormalizesOnlyPrefixAndPreservesRawTailBits() throws {
        var rows = [Float](
            repeating: 2,
            count: GLM52KVLoRANormReference.rawWidth
        )
        for index in 0..<GLM52KVLoRANormReference.kRoPEWidth {
            rows[GLM52KVLoRANormReference.kvLoRAWidth + index] =
                Float(bitPattern: 0x8000_0000 | UInt32(index))
        }
        let weight = (0..<GLM52KVLoRANormReference.kvLoRAWidth).map {
            Float($0 + 1) / 512
        }
        let output = try GLM52KVLoRANormReference.normalize(
            rawRows: rows,
            weight: weight
        )
        let inverseRMS: Float = 1 / sqrt(4 + GLM52KVLoRANormReference.epsilon)
        for column in 0..<GLM52KVLoRANormReference.kvLoRAWidth {
            XCTAssertEqual(
                output[column],
                2 * inverseRMS * weight[column],
                accuracy: 1e-6
            )
        }
        for column in GLM52KVLoRANormReference.kvLoRAWidth
            ..< GLM52KVLoRANormReference.rawWidth {
            XCTAssertEqual(output[column].bitPattern, rows[column].bitPattern)
        }
    }

    func testMetalKVLoRANormMatchesScalarOracle() throws {
        let runtime = try makeRuntime()
        let input = fixture(tokenCount: 3)
        let expected = try GLM52KVLoRANormReference.normalize(
            rawRows: input.rows,
            weight: input.weight
        )
        let actual = try runtime.glm52NormalizeKVLoRA(
            rawRows: input.rows,
            weight: input.weight
        )
        XCTAssertEqual(actual.count, expected.count)
        for token in 0..<3 {
            let row = token * GLM52KVLoRANormReference.rawWidth
            for column in 0..<GLM52KVLoRANormReference.kvLoRAWidth {
                XCTAssertEqual(
                    actual[row + column],
                    expected[row + column],
                    accuracy: 3e-5,
                    "token \(token), column \(column)"
                )
            }
            for column in GLM52KVLoRANormReference.kvLoRAWidth
                ..< GLM52KVLoRANormReference.rawWidth {
                XCTAssertEqual(
                    actual[row + column].bitPattern,
                    input.rows[row + column].bitPattern
                )
            }
        }
    }

    func testRejectsMalformedRowsAndWeight() throws {
        XCTAssertThrowsError(try GLM52KVLoRANormReference.normalize(
            rawRows: [Float](repeating: 0, count: 575),
            weight: [Float](repeating: 1, count: 512)
        ))
        XCTAssertThrowsError(try GLM52KVLoRANormReference.normalize(
            rawRows: [Float](repeating: 0, count: 576),
            weight: [Float](repeating: 1, count: 511)
        ))
    }
}
