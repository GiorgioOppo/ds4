import DS4Core
import XCTest
@testable import DS4Metal

/// The tail RoPE oracle must behave like upstream's linear GLM rope (identity
/// at position 0, per-pair norm preservation, inverse composition) and the
/// GPU kernel must match it within closed-form-vs-iterative theta drift.
/// Positions stay moderate: fp32 trigonometry of huge angles diverges by
/// argument reduction — the same documented caveat as the DeepSeek rope.
final class GLM52RopeTailTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    private func values(count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(state >> 33) / Float(1 << 30) - 1.0
        }
    }

    func testPositionZeroIsIdentity() throws {
        let input = values(count: 2 * 96, seed: 3)
        let out = try GLM52RopeTailReference.rotate(
            values: input, headCount: 2, headDimension: 96,
            rotationDimension: 64, position: 0)
        XCTAssertEqual(out, input)
    }

    func testNopePrefixIsUntouchedAndPairsPreserveNorm() throws {
        let headDim = 96, rot = 64, nope = headDim - rot
        let input = values(count: headDim, seed: 7)
        let out = try GLM52RopeTailReference.rotate(
            values: input, headCount: 1, headDimension: headDim,
            rotationDimension: rot, position: 1_234)

        XCTAssertEqual(Array(out[0..<nope]), Array(input[0..<nope]))
        for i in stride(from: nope, to: headDim, by: 2) {
            let before = input[i] * input[i] + input[i + 1] * input[i + 1]
            let after = out[i] * out[i] + out[i + 1] * out[i + 1]
            XCTAssertEqual(after, before, accuracy: 1e-5,
                           "pair at \(i) changed norm")
        }
    }

    func testInverseComposesToIdentity() throws {
        let input = values(count: 4 * 64, seed: 11)
        let rotated = try GLM52RopeTailReference.rotate(
            values: input, headCount: 4, headDimension: 64,
            rotationDimension: 64, position: 777)
        let restored = try GLM52RopeTailReference.rotate(
            values: rotated, headCount: 4, headDimension: 64,
            rotationDimension: 64, position: 777, inverse: true)
        for i in 0..<input.count {
            XCTAssertEqual(restored[i], input[i], accuracy: 1e-4)
        }
    }

    func testKernelMatchesOracleOnQueryHeadsAndKRow() throws {
        let runtime = try makeRuntime()
        for (heads, headDim, position) in [(64, 256, 1), (64, 256, 4_095),
                                           (1, 64, 2_048)] {
            let input = values(count: heads * headDim,
                               seed: UInt64(97 + position))
            let gpu = try runtime.glm52RopeTail(
                values: input, headCount: heads, headDimension: headDim,
                rotationDimension: 64, position: position)
            let oracle = try GLM52RopeTailReference.rotate(
                values: input, headCount: heads, headDimension: headDim,
                rotationDimension: 64, position: position)
            for i in 0..<gpu.count {
                XCTAssertEqual(gpu[i], oracle[i],
                               accuracy: 1e-3 + abs(oracle[i]) * 1e-3,
                               "heads \(heads) pos \(position) diverges at \(i)")
            }
        }
    }

    func testRejectionsFireOnBadGeometry() {
        XCTAssertThrowsError(try GLM52RopeTailReference.rotate(
            values: [1, 2, 3], headCount: 1, headDimension: 4,
            rotationDimension: 4, position: 0))
        XCTAssertThrowsError(try GLM52RopeTailReference.rotate(
            values: [1, 2, 3, 4], headCount: 1, headDimension: 4,
            rotationDimension: 3, position: 0))
        XCTAssertThrowsError(try GLM52RopeTailReference.rotate(
            values: [1, 2, 3, 4], headCount: 1, headDimension: 4,
            rotationDimension: 6, position: 0))
    }
}
