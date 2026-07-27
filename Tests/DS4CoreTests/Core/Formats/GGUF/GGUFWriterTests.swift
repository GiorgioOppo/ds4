import XCTest
@testable import DS4Core

/// Round-trips the pure-Swift GGUFWriter against the GGUFModel reader: anything
/// written must parse back to identical metadata, tensor shapes and bytes. This
/// is the reader's inverse, so a passing round-trip pins the on-disk layout.
final class GGUFWriterTests: XCTestCase {

    private func tempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("ggufw-\(UUID().uuidString).gguf")
    }

    func testRoundTripMetadataAndTensors() throws {
        // Two tiny f32 tensors with known bytes.
        let aVals: [Float] = [1, 2, 3, 4]
        let bVals: [Float] = [10, 20, 30, 40, 50, 60]
        func f32Data(_ xs: [Float]) -> Data {
            var d = Data()
            for x in xs { withUnsafeBytes(of: x.bitPattern.littleEndian) { d.append(contentsOf: $0) } }
            return d
        }

        var w = try GGUFWriter(alignment: 32)
        w.put("general.alignment", .uint32(32))
        w.put("general.name", .text("tiny"))
        w.put("general.architecture", .text("deepseek4"))
        w.put("deepseek4.block_count", .uint32(2))
        w.put("test.u64", .uint64(123_456_789))
        w.put("test.i32", .int32(-7))
        w.put("test.bool", .bool(true))
        w.put("test.f32", .float32(1.5))
        w.put("test.arr", .array(elementType: .int32,
                                 elements: [.int32(5), .int32(-6), .int32(7)]))
        w.add(.init(name: "a.weight", dims: [4], type: 0, data: f32Data(aVals)))
        w.add(.init(name: "b.weight", dims: [2, 3], type: 0, data: f32Data(bVals)))

        let url = tempURL()
        try w.write(to: url.path)
        defer { try? FileManager.default.removeItem(at: url) }

        let m = try GGUFModel(path: url.path, metalMapping: false)
        XCTAssertEqual(m.version, 3)
        XCTAssertEqual(m.n_kv, 9)
        XCTAssertEqual(m.n_tensors, 2)
        XCTAssertEqual(m.alignment, 32)

        XCTAssertEqual(m.string("general.name"), "tiny")
        XCTAssertEqual(m.string("general.architecture"), "deepseek4")
        XCTAssertEqual(m.u32("deepseek4.block_count"), 2)
        XCTAssertEqual(m.u64("test.u64"), 123_456_789)
        XCTAssertEqual(m.bool("test.bool"), true)
        XCTAssertEqual(m.f32Compat("test.f32"), 1.5)
        XCTAssertEqual(m.intArray("test.arr"), [5, -6, 7])

        let a = try XCTUnwrap(m.findTensor("a.weight"))
        XCTAssertEqual(a.dims, [4]); XCTAssertEqual(a.typeName, "f32"); XCTAssertEqual(a.bytes, 16)
        XCTAssertEqual(m.tensorData(a), f32Data(aVals))

        let b = try XCTUnwrap(m.findTensor("b.weight"))
        XCTAssertEqual(b.dims, [2, 3]); XCTAssertEqual(b.bytes, 24)
        XCTAssertEqual(m.tensorData(b), f32Data(bVals))

        // Each tensor's data offset is alignment-aligned.
        XCTAssertEqual(a.absOffset % 32, 0)
        XCTAssertEqual(b.absOffset % 32, 0)
    }

    /// build() (single Data) and write(to:) (streamed) must be byte-identical.
    func testBuildMatchesStreamedWrite() throws {
        var w = try GGUFWriter()
        w.put("general.name", .text("x"))
        w.add(.init(name: "t", dims: [8], type: 0, data: Data(count: 32)))
        let built = try w.build()
        let url = tempURL()
        try w.write(to: url.path)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertEqual(built, try Data(contentsOf: url))
    }

    /// Reading a file and re-writing its metadata + tensors reproduces a file the
    /// reader parses identically (the requantizer's read->write backbone).
    func testReadReExportRoundTrip() throws {
        var w = try GGUFWriter()
        w.put("general.alignment", .uint32(32))
        w.put("general.architecture", .text("deepseek4"))
        w.put("tokens", .array(elementType: .string,
                               elements: [.text("a"), .text("bb"), .text("ccc")]))
        w.add(.init(name: "w0", dims: [16], type: 0, data: Data(count: 64)))
        let src = tempURL()
        try w.write(to: src.path)
        defer { try? FileManager.default.removeItem(at: src) }

        let m0 = try GGUFModel(path: src.path, metalMapping: false)
        let meta = try m0.allMetadata()
        let tensors = m0.tensors.map { m0.tensorInputPassthrough($0) }
        let w2 = try GGUFWriter(metadata: meta, tensors: tensors)

        let dst = tempURL()
        try w2.write(to: dst.path)
        defer { try? FileManager.default.removeItem(at: dst) }

        let m1 = try GGUFModel(path: dst.path, metalMapping: false)
        XCTAssertEqual(m1.n_kv, m0.n_kv)
        XCTAssertEqual(m1.n_tensors, m0.n_tensors)
        XCTAssertEqual(m1.string("general.architecture"), "deepseek4")
        XCTAssertEqual(m1.stringArrayBytes("tokens")?.map { Array($0) },
                       [Array("a".utf8), Array("bb".utf8), Array("ccc".utf8)])
        let t = try XCTUnwrap(m1.findTensor("w0"))
        XCTAssertEqual(t.dims, [16]); XCTAssertEqual(t.bytes, 64)
    }

    func testRejectsWrongTensorDataSize() throws {
        var w = try GGUFWriter()
        w.add(.init(name: "bad", dims: [4], type: 0, data: Data(count: 8)))  // needs 16
        XCTAssertThrowsError(try w.build())
    }
}
