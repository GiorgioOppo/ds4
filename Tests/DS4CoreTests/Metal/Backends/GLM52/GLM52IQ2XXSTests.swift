import DS4Core
import XCTest
@testable import DS4Metal

/// IQ2_XXS — the routed-expert format of the published GLM 5.2 GGUF. The CPU
/// dequant is pinned structurally (known-cell decoding, byte-flip
/// sensitivity) and the GPU dot kernel is judged against a CPU matvec over
/// the dequantized rows — two independent decoders of the same tables.
final class GLM52IQ2XXSTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch MetalError.noDevice { throw XCTSkip("Metal device unavailable") }
    }

    /// One 66-byte superblock from raw parts.
    private func block(d: Float, words: [UInt16]) -> [UInt8] {
        precondition(words.count == 32)
        var bytes: [UInt8] = []
        withUnsafeBytes(of: Half.bits(d).littleEndian) {
            bytes.append(contentsOf: $0)
        }
        for word in words {
            withUnsafeBytes(of: word.littleEndian) {
                bytes.append(contentsOf: $0)
            }
        }
        return bytes
    }

    private func dequant(_ bytes: [UInt8], count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        bytes.withUnsafeBytes {
            Quantize.dequantIQ2_XXS($0.baseAddress!, count: count, into: &out)
        }
        return out
    }

    /// Random words with a FIXED finite f16 scale per superblock, so random
    /// bit patterns can never produce NaN/Inf weights.
    private func randomBlocks(count: Int, seed: UInt64) -> [UInt8] {
        var state = seed
        var bytes = (0..<count).map { _ -> UInt8 in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt8(truncatingIfNeeded: state >> 33)
        }
        let scale = Half.bits(0.02)
        for block in stride(from: 0, to: count, by: 66) {
            bytes[block] = UInt8(scale & 0xFF)
            bytes[block + 1] = UInt8(scale >> 8)
        }
        return bytes
    }

    func testKnownCellDecoding() throws {
        // All-zero words: every cell uses grid[0] (all magnitudes 8), sign
        // index 0 and cell scale 0 → db = d * 0.5 * 0.25. With d = 1 every
        // value has magnitude 1.0 exactly.
        let zeros = dequant(block(d: 1, words: [UInt16](repeating: 0,
                                                        count: 32)),
                            count: 256)
        XCTAssertEqual(zeros.count, 256)
        for (i, value) in zeros.enumerated() {
            XCTAssertEqual(abs(value), 1.0, "cell magnitude at \(i)")
        }
        // Raising the first group's cell scale nibble to 15 rescales ONLY
        // that group: db = (0.5 + 15) * 0.25 = 3.875.
        var words = [UInt16](repeating: 0, count: 32)
        words[3] = 0xF000
        let scaled = dequant(block(d: 1, words: words), count: 256)
        for i in 0..<32 {
            XCTAssertEqual(abs(scaled[i]), 3.875, "scaled group at \(i)")
        }
        for i in 32..<256 {
            XCTAssertEqual(abs(scaled[i]), 1.0, "untouched group at \(i)")
        }
    }

    func testEveryByteParticipates() throws {
        let original = randomBlocks(count: 66 * 2, seed: 9)
        let baseline = dequant(original, count: 512)
        for index in 0..<original.count {
            var mutated = original
            mutated[index] ^= 0x40
            XCTAssertNotEqual(dequant(mutated, count: 512), baseline,
                              "byte \(index) does not affect the output")
        }
    }

    func testGPUDotMatchesDequantizedOracle() throws {
        let runtime = try makeRuntime()
        let width = 512
        let rows = 96
        let rowBytes = (width / 256) * 66
        let weightBytes = randomBlocks(count: rows * rowBytes, seed: 41)
        var input = [Float](repeating: 0, count: width)
        var state: UInt64 = 77
        for i in 0..<width {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            input[i] = (Float(state >> 33) / Float(1 << 30) - 1) * 0.5
        }

        let gpu = try runtime.glm52MoEDown(
            mid: input, downRows: weightBytes,
            weightType: GLM52TensorSchema.iq2_XXS, outputWidth: rows)

        var dequantized = [Float](repeating: 0, count: rows * width)
        weightBytes.withUnsafeBytes {
            Quantize.dequantIQ2_XXS($0.baseAddress!, count: rows * width,
                                    into: &dequantized)
        }
        let oracle = try GLM52FFNCPUReference.matvec(
            rows: dequantized, input: input, rowCount: rows)

        XCTAssertEqual(gpu.count, oracle.count)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], oracle[i],
                           accuracy: 1e-3 + abs(oracle[i]) * 1e-3,
                           "IQ2_XXS dot diverges at row \(i)")
        }
    }

    func testProviderAcceptsIQ2XXSExperts() throws {
        XCTAssertTrue(GLM52StreamedExpertProvider.supportedTypes
            .contains(GLM52TensorSchema.iq2_XXS))
        XCTAssertNotNil(MetalRuntime.glm52KQuantRowBytes(
            type: GLM52TensorSchema.iq2_XXS, width: 512))
        XCTAssertEqual(MetalRuntime.glm52KQuantRowBytes(
            type: GLM52TensorSchema.iq2_XXS, width: 512), 132)
    }
}
