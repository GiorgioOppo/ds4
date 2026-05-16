import XCTest
@testable import DeepSeekKit

/// Tests the GGUF header parser end-to-end. Builds a synthetic GGUF
/// file in memory (no external fixtures) and verifies metadata
/// round-trip + tensor info reconstruction.
final class GGUFTests: XCTestCase {

    /// Round-trip: build a GGUF byte stream with 2 KV pairs and 1
    /// F32 tensor, parse it back, verify everything matches.
    func testHeaderRoundTrip() throws {
        var builder = GGUFBuilder()
        builder.writeMagic("GGUF")
        builder.writeU32(3)                          // version
        builder.writeU64(1)                          // tensor_count
        builder.writeU64(2)                          // kv_count
        // KV 1: general.alignment = 32 (uint64)
        builder.writeGGUFString("general.alignment")
        builder.writeU32(GGUFValueType.uint64.rawValue)
        builder.writeU64(32)
        // KV 2: general.name = "toy" (string)
        builder.writeGGUFString("general.name")
        builder.writeU32(GGUFValueType.string.rawValue)
        builder.writeGGUFString("toy")
        // Tensor info: name="weight", shape=[2,3], type=F32, offset=0
        builder.writeGGUFString("weight")
        builder.writeU32(2)                          // n_dims
        builder.writeU64(2)                          // dim 0
        builder.writeU64(3)                          // dim 1
        builder.writeI32(GGUFType.f32.rawValue)
        builder.writeU64(0)                          // relative offset

        // Pad to alignment then write 6 floats.
        builder.padTo(32)
        let values: [Float] = [1, 2, 3, 4, 5, 6]
        for v in values {
            builder.writeU32(v.bitPattern)
        }

        let data = builder.data
        let header = try data.withUnsafeBytes { GGUFHeader.parse(buffer: $0) }

        XCTAssertEqual(header.version, 3)
        XCTAssertEqual(header.alignment, 32)
        XCTAssertEqual(header.tensors.count, 1)
        XCTAssertEqual(header.tensors[0].name, "weight")
        XCTAssertEqual(header.tensors[0].shape, [2, 3])
        XCTAssertEqual(header.tensors[0].type, .f32)
        XCTAssertEqual(header.tensors[0].byteCount, 24)
        if case .string(let name)? = header.metadata["general.name"] {
            XCTAssertEqual(name, "toy")
        } else {
            XCTFail("expected string metadata for general.name")
        }
    }

    /// Bad magic must throw.
    func testRejectsBadMagic() {
        var builder = GGUFBuilder()
        builder.writeMagic("XXXX")
        builder.writeU32(3)
        builder.writeU64(0)
        builder.writeU64(0)
        let data = builder.data
        XCTAssertThrowsError(try data.withUnsafeBytes { GGUFHeader.parse(buffer: $0) }) { err in
            guard case GGUFError.badMagic = err else {
                return XCTFail("expected badMagic, got \(err)")
            }
        }
    }

    /// Unsupported version must throw.
    func testRejectsBadVersion() {
        var builder = GGUFBuilder()
        builder.writeMagic("GGUF")
        builder.writeU32(99)
        builder.writeU64(0)
        builder.writeU64(0)
        let data = builder.data
        XCTAssertThrowsError(try data.withUnsafeBytes { GGUFHeader.parse(buffer: $0) }) { err in
            guard case GGUFError.unsupportedVersion = err else {
                return XCTFail("expected unsupportedVersion, got \(err)")
            }
        }
    }

    /// Block sizing for a couple of quantised types: Q8_0 has 32-elem
    /// blocks at 34 bytes, Q4_K has 256-elem blocks at 144 bytes.
    func testQuantBlockSizing() {
        XCTAssertEqual(GGUFType.q8_0.blockSize, 32)
        XCTAssertEqual(GGUFType.q8_0.bytesPerBlock, 34)
        XCTAssertEqual(GGUFType.q4_0.blockSize, 32)
        XCTAssertEqual(GGUFType.q4_0.bytesPerBlock, 18)
        XCTAssertEqual(GGUFType.q4_K.blockSize, 256)
        XCTAssertEqual(GGUFType.q4_K.bytesPerBlock, 144)
    }

    /// Pass-through dtypes (F32/F16/BF16/I32/I8) report
    /// `isPassThroughDType == true`; quantised types are false.
    func testPassThroughClassification() {
        for t in [GGUFType.f32, .f16, .bf16, .i32, .i8] {
            XCTAssertTrue(t.isPassThroughDType, "\(t) should be pass-through")
        }
        for t in [GGUFType.q4_0, .q4_K, .q8_0, .q6_K] {
            XCTAssertFalse(t.isPassThroughDType, "\(t) requires dequant")
        }
    }
}

// MARK: - Test helper: minimal GGUF byte stream builder

private struct GGUFBuilder {
    private(set) var data = Data()

    mutating func writeMagic(_ s: String) {
        data.append(contentsOf: Array(s.utf8))
    }
    mutating func writeU32(_ v: UInt32) {
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
    }
    mutating func writeI32(_ v: Int32) {
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
    }
    mutating func writeU64(_ v: UInt64) {
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
    }
    mutating func writeGGUFString(_ s: String) {
        let bytes = Array(s.utf8)
        writeU64(UInt64(bytes.count))
        data.append(contentsOf: bytes)
    }
    mutating func padTo(_ alignment: Int) {
        let pad = (alignment - (data.count % alignment)) % alignment
        data.append(contentsOf: [UInt8](repeating: 0, count: pad))
    }
}
