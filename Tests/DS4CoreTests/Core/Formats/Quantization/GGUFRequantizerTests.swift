import XCTest
@testable import DS4Core

/// End-to-end offline requantization: write an F32 source GGUF, requantize it
/// through GGUFRequantizer, read the result back and check the dequantized values
/// track the originals. Exercises the full read -> dequant -> requant -> write
/// path with the byte-exact QuantEncode encoders.
final class GGUFRequantizerTests: XCTestCase {

    private func tempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("ggufrq-\(UUID().uuidString).gguf")
    }

    private func f32Data(_ xs: [Float]) -> Data {
        var d = Data()
        for x in xs { withUnsafeBytes(of: x.bitPattern.littleEndian) { d.append(contentsOf: $0) } }
        return d
    }

    /// Build an F32 source GGUF: one requantizable weight [256 x 2] plus a small
    /// i32 tensor that must pass through untouched.
    private func writeSource(_ vals: [Float]) throws -> URL {
        var w = try GGUFWriter(alignment: 32)
        w.put("general.architecture", .text("deepseek4"))
        w.add(.init(name: "blk.0.ffn_down.weight", dims: [256, 2], type: 0, data: f32Data(vals)))
        w.add(.init(name: "blk.0.router.weight", dims: [4], type: 26,
                    data: Data([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0])))  // i32 [4]
        let url = tempURL()
        try w.write(to: url.path)
        return url
    }

    private func dequantize(_ m: GGUFModel, _ name: String) throws -> [Float] {
        let t = try XCTUnwrap(m.findTensor(name))
        return try XCTUnwrap(GGUFRequantizer.dequantizeToF32(
            type: t.type, src: m.mapBase.advanced(by: Int(t.absOffset)), count: Int(t.elements)))
    }

    func testRequantF32toQ8_0RoundTrip() throws {
        let vals = (0..<512).map { Float(sin(Double($0) * 0.1)) }   // in [-1, 1]
        let src = try writeSource(vals)
        defer { try? FileManager.default.removeItem(at: src) }
        let m0 = try GGUFModel(path: src.path, metalMapping: false)

        let dst = tempURL()
        defer { try? FileManager.default.removeItem(at: dst) }
        let report = try GGUFRequantizer.requantize(
            source: m0,
            options: .remap([0: 8], include: { $0.hasSuffix("ffn_down.weight") }),  // f32 -> q8_0
            to: dst.path)
        XCTAssertEqual(report.requantized, 1)
        XCTAssertEqual(report.passthrough, 1)   // the i32 tensor
        XCTAssertEqual(report.skipped, 0)

        let m1 = try GGUFModel(path: dst.path, metalMapping: false)
        let w = try XCTUnwrap(m1.findTensor("blk.0.ffn_down.weight"))
        XCTAssertEqual(w.typeName, "q8_0")
        XCTAssertEqual(w.dims, [256, 2])

        let back = try dequantize(m1, "blk.0.ffn_down.weight")
        XCTAssertEqual(back.count, 512)
        for i in 0..<512 { XCTAssertEqual(back[i], vals[i], accuracy: 0.02) }   // Q8_0 step

        // i32 tensor untouched (bytes identical).
        let r0 = try XCTUnwrap(m0.findTensor("blk.0.router.weight"))
        let r1 = try XCTUnwrap(m1.findTensor("blk.0.router.weight"))
        XCTAssertEqual(r1.typeName, "i32")
        XCTAssertEqual(m0.tensorData(r0), m1.tensorData(r1))
    }

    func testRequantF32toQ4_K() throws {
        let vals = (0..<512).map { Float(cos(Double($0) * 0.05)) }
        let src = try writeSource(vals)
        defer { try? FileManager.default.removeItem(at: src) }
        let m0 = try GGUFModel(path: src.path, metalMapping: false)

        let dst = tempURL()
        defer { try? FileManager.default.removeItem(at: dst) }
        let report = try GGUFRequantizer.requantize(
            source: m0, options: .remap([0: 12]), to: dst.path)   // f32 -> q4_k
        XCTAssertEqual(report.requantized, 1)

        let m1 = try GGUFModel(path: dst.path, metalMapping: false)
        let w = try XCTUnwrap(m1.findTensor("blk.0.ffn_down.weight"))
        XCTAssertEqual(w.typeName, "q4_k")
        XCTAssertEqual(w.bytes, GGUF.tensorNBytes(type: 12, elements: 512))

        let back = try dequantize(m1, "blk.0.ffn_down.weight")
        var maxErr: Float = 0
        for i in 0..<512 { maxErr = max(maxErr, abs(back[i] - vals[i])) }
        XCTAssertLessThan(maxErr, 0.1)   // Q4_K on a smooth [-1,1] signal
    }

    /// A target that needs an imatrix but has none is skipped, not corrupted.
    func testIq2WithoutImatrixIsSkipped() throws {
        let vals = (0..<512).map { Float($0 % 7) * 0.1 }
        let src = try writeSource(vals)
        defer { try? FileManager.default.removeItem(at: src) }
        let m0 = try GGUFModel(path: src.path, metalMapping: false)

        let dst = tempURL()
        defer { try? FileManager.default.removeItem(at: dst) }
        let report = try GGUFRequantizer.requantize(
            source: m0,
            options: .remap([0: 16], include: { $0.hasSuffix("ffn_down.weight") }),  // -> iq2_xxs
            to: dst.path)
        XCTAssertEqual(report.requantized, 0)
        XCTAssertEqual(report.skipped, 1)

        // The weight stays F32 (passed through unchanged).
        let m1 = try GGUFModel(path: dst.path, metalMapping: false)
        XCTAssertEqual(try XCTUnwrap(m1.findTensor("blk.0.ffn_down.weight")).typeName, "f32")
    }
}
