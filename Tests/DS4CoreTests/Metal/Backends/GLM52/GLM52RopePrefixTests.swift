import DS4Core
import XCTest
@testable import DS4Metal

/// The prefix RoPE — the indexer convention (rot_offset 0) — must rotate ONLY
/// the leading span of each head, invert cleanly, agree with the tail rotation
/// when the whole head rotates, and the GPU kernel must match the oracle.
final class GLM52RopePrefixTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    private func values(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(truncatingIfNeeded: state >> 32)) / Float(Int32.max)
        }
    }

    func testPrefixRotatesLeadingSpanOnly() throws {
        let input = values(2 * 8, seed: 3)
        let rotated = try GLM52RopeTailReference.rotatePrefix(
            values: input, headCount: 2, headDimension: 8,
            rotationDimension: 4, position: 9)

        for head in 0..<2 {
            let base = head * 8
            // The trailing half is untouched…
            for i in 4..<8 {
                XCTAssertEqual(rotated[base + i], input[base + i],
                               "head \(head) tail element \(i) moved")
            }
            // …and the leading half moved (position 9 is far from identity).
            XCTAssertNotEqual(Array(rotated[base..<base + 4]),
                              Array(input[base..<base + 4]))
        }
    }

    func testPositionZeroIsIdentityAndInverseRoundTrips() throws {
        let input = values(3 * 16, seed: 11)
        let identity = try GLM52RopeTailReference.rotatePrefix(
            values: input, headCount: 3, headDimension: 16,
            rotationDimension: 8, position: 0)
        XCTAssertEqual(identity, input)

        let rotated = try GLM52RopeTailReference.rotatePrefix(
            values: input, headCount: 3, headDimension: 16,
            rotationDimension: 8, position: 1_234)
        let restored = try GLM52RopeTailReference.rotatePrefix(
            values: rotated, headCount: 3, headDimension: 16,
            rotationDimension: 8, position: 1_234, inverse: true)
        for i in 0..<input.count {
            XCTAssertEqual(restored[i], input[i], accuracy: 1e-5,
                           "inverse round trip diverges at \(i)")
        }
    }

    func testFullHeadRotationMatchesTailConvention() throws {
        // When the rotated span covers the whole head, prefix and tail
        // conventions describe the same operation.
        let input = values(4 * 6, seed: 21)
        let prefix = try GLM52RopeTailReference.rotatePrefix(
            values: input, headCount: 4, headDimension: 6,
            rotationDimension: 6, position: 57)
        let tail = try GLM52RopeTailReference.rotate(
            values: input, headCount: 4, headDimension: 6,
            rotationDimension: 6, position: 57)
        XCTAssertEqual(prefix, tail)
    }

    func testGPUPrefixKernelMatchesOracle() throws {
        let runtime = try makeRuntime()
        // The real indexer geometry: 32 heads of 128, rotated prefix 64.
        let input = values(32 * 128, seed: 77)
        let gpu = try runtime.glm52RopePrefix(
            values: input, headCount: 32, headDimension: 128,
            rotationDimension: 64, position: 913)
        let oracle = try GLM52RopeTailReference.rotatePrefix(
            values: input, headCount: 32, headDimension: 128,
            rotationDimension: 64, position: 913)

        XCTAssertEqual(gpu.count, oracle.count)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], oracle[i], accuracy: 1e-3,
                           "prefix RoPE diverges from the oracle at \(i)")
        }
        // The unrotated halves must be bit-identical to the input.
        for head in 0..<32 {
            let base = head * 128
            for i in 64..<128 {
                XCTAssertEqual(gpu[base + i], input[base + i],
                               "head \(head) tail element \(i) moved on GPU")
            }
        }
    }

    func testValidationRejectsBadGeometry() throws {
        XCTAssertThrowsError(try GLM52RopeTailReference.rotatePrefix(
            values: [1, 2, 3], headCount: 1, headDimension: 3,
            rotationDimension: 3, position: 0))
        XCTAssertThrowsError(try GLM52RopeTailReference.rotatePrefix(
            values: [1, 2], headCount: 1, headDimension: 2,
            rotationDimension: 4, position: 0))
        XCTAssertThrowsError(try GLM52RopeTailReference.rotatePrefix(
            values: [1, 2], headCount: 1, headDimension: 2,
            rotationDimension: 2, position: -1))
    }
}
