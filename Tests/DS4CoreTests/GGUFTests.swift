import XCTest
@testable import DS4Core

/// Appends little-endian scalars / GGUF strings to build a synthetic GGUF.
private struct GGUFWriter {
    var data = Data()
    mutating func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    mutating func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    mutating func f32(_ v: Float)  { withUnsafeBytes(of: v.bitPattern.littleEndian) { data.append(contentsOf: $0) } }
    mutating func u8(_ v: UInt8)   { data.append(v) }
    mutating func str(_ s: String) {
        let bytes = Array(s.utf8)
        u64(UInt64(bytes.count))
        data.append(contentsOf: bytes)
    }
}

final class GGUFTests: XCTestCase {

    // MARK: Synthetic GGUF — exact ground truth built in the test.

    func testSyntheticGGUF() throws {
        var w = GGUFWriter()
        w.u32(0x4655_4747)        // magic "GGUF"
        w.u32(3)                  // version
        w.u64(2)                  // n_tensors
        w.u64(6)                  // n_kv

        // KV table
        w.str("general.alignment"); w.u32(4); w.u32(32)
        w.str("general.name");      w.u32(8); w.str("tiny")          // type 8 = string
        w.str("deepseek4.block_count"); w.u32(4); w.u32(2)
        w.str("test.u64");          w.u32(10); w.u64(123_456_789)     // type 10 = u64
        w.str("test.bool");         w.u32(7); w.u8(1)                 // type 7 = bool
        w.str("test.f32");          w.u32(6); w.f32(1.5)             // type 6 = f32

        // Tensor directory: a.weight [4] f32, b.weight [2,3] f32
        w.str("a.weight"); w.u32(1); w.u64(4); w.u32(0); w.u64(0)            // rel 0, 16 bytes
        w.str("b.weight"); w.u32(2); w.u64(2); w.u64(3); w.u32(0); w.u64(64) // rel 64, 24 bytes

        // Pad to alignment (32) then tensor data region (enough for rel 64 + 24).
        while w.data.count % 32 != 0 { w.u8(0) }
        let dataStart = w.data.count
        w.data.append(Data(count: 128))   // room for both tensors

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("synthetic-\(UUID().uuidString).gguf")
        try w.data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let m = try GGUFModel(path: url.path, metalMapping: false)
        XCTAssertEqual(m.version, 3)
        XCTAssertEqual(m.n_kv, 6)
        XCTAssertEqual(m.n_tensors, 2)
        XCTAssertEqual(m.alignment, 32)

        XCTAssertEqual(m.string("general.name"), "tiny")
        XCTAssertEqual(m.u32("deepseek4.block_count"), 2)
        XCTAssertEqual(m.u64("test.u64"), 123_456_789)
        XCTAssertEqual(m.u64Compat("deepseek4.block_count"), 2)   // u32 read as u64
        XCTAssertEqual(m.bool("test.bool"), true)
        XCTAssertEqual(m.f32Compat("test.f32"), 1.5)
        XCTAssertEqual(m.f32Compat("deepseek4.block_count"), 2.0) // u32 -> f32
        XCTAssertNil(m.u32("test.u64"))                            // wrong type -> nil

        let a = try XCTUnwrap(m.findTensor("a.weight"))
        XCTAssertEqual(a.dims, [4]); XCTAssertEqual(a.elements, 4)
        XCTAssertEqual(a.typeName, "f32"); XCTAssertEqual(a.bytes, 16)
        XCTAssertEqual(Int(a.absOffset), dataStart)

        let b = try XCTUnwrap(m.findTensor("b.weight"))
        XCTAssertEqual(b.dims, [2, 3]); XCTAssertEqual(b.elements, 6)
        XCTAssertEqual(b.bytes, 24)
        XCTAssertEqual(Int(b.absOffset), dataStart + 64)
        XCTAssertEqual(m.tensorDataPos, UInt64(dataStart))
        XCTAssertNil(m.findTensor("missing"))
    }

    func testRejectsNonGGUF() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("notgguf-\(UUID().uuidString).bin")
        try Data(repeating: 0xAB, count: 64).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(try GGUFModel(path: url.path, metalMapping: false))
    }

    // MARK: Real GGUF — cross-checked against `ds4 --inspect` (the C parser).

    /// Ground truth from: ./ds4 --inspect on the v2 Q4 Flash GGUF.
    func testRealGGUFMatchesCInspect() throws {
        let path = "/Users/oppog/Downloads/ds4-main/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
        try XCTSkipUnless(FileManager.default.fileExists(atPath: path), "real GGUF not present")

        let m = try GGUFModel(path: path, metalMapping: false)
        XCTAssertEqual(m.version, 3)
        XCTAssertEqual(m.n_kv, 62)
        XCTAssertEqual(m.n_tensors, 1328)

        XCTAssertEqual(m.string("general.name"), "DeepSeek V4 Flash")
        XCTAssertEqual(m.string("general.architecture"), "deepseek4")
        XCTAssertEqual(m.u32("deepseek4.block_count"), 43)
        XCTAssertEqual(m.u32("deepseek4.attention.head_count"), 64)
        XCTAssertEqual(m.u32("deepseek4.attention.head_count_kv"), 1)
        XCTAssertEqual(m.u32("deepseek4.attention.key_length"), 512)
        XCTAssertEqual(m.u32("deepseek4.attention.sliding_window"), 128)
        XCTAssertEqual(m.u32("deepseek4.attention.indexer.head_count"), 64)
        XCTAssertEqual(m.u32("deepseek4.attention.indexer.key_length"), 128)
        XCTAssertEqual(m.u32("deepseek4.attention.indexer.top_k"), 512)
        XCTAssertEqual(m.u32("deepseek4.expert_count"), 256)
        XCTAssertEqual(m.u32("deepseek4.expert_used_count"), 6)

        // Tensor-type breakdown must match the C inspect output exactly.
        var counts: [String: Int] = [:]
        var totalBytes: UInt64 = 0
        for t in m.tensors {
            counts[t.typeName, default: 0] += 1
            totalBytes += t.bytes
        }
        XCTAssertEqual(counts["f32"], 492)
        XCTAssertEqual(counts["f16"], 359)
        XCTAssertEqual(counts["q8_0"], 345)
        XCTAssertEqual(counts["q4_k"], 129)
        XCTAssertEqual(counts["i32"], 3)
        XCTAssertEqual(counts.values.reduce(0, +), 1328)

        // 153.32 GiB of described tensor bytes (within rounding of the report).
        let gib = Double(totalBytes) / 1_073_741_824.0
        XCTAssertEqual(gib, 153.32, accuracy: 0.02)

        // Sanity: core tensors resolve and stay inside the file.
        XCTAssertNotNil(m.findTensor("token_embd.weight"))
        XCTAssertNotNil(m.findTensor("output.weight"))
        for t in m.tensors where t.bytes != 0 {
            XCTAssertLessThanOrEqual(t.absOffset + t.bytes, m.size)
        }
    }
}
