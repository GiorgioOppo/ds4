import XCTest
@testable import DS4Core

/// Byte-exact pinning of the Swift quantization ENCODERS against fixtures
/// produced by the compiled C reference (ds4 `gguf-tools/quants.c`, built
/// with `-ffp-contract=off`). Any divergence — a rounding change, a search
/// tweak, a packing bug — shows up as a byte diff, not a quality drift.
final class QuantEncodeTests: XCTestCase {
    private static let input: [Float] =
        QuantEncodeFixtures.inputBits.map(Float.init(bitPattern:))
    private static let imatrix: [Float] =
        QuantEncodeFixtures.imatrixBits.map(Float.init(bitPattern:))

    private func encode(type: UInt32, start: Int = 0, rows: Int, columns: Int,
                        weighted: Bool) -> [UInt8] {
        let rowBytes = QuantEncode.rowSize(type: type, columns: columns)
        let startRow = start / columns
        var out = [UInt8](repeating: 0, count: (startRow + rows) * rowBytes)
        let written = Self.input.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBytes { dst in
                if weighted {
                    return Self.imatrix.withUnsafeBufferPointer { im in
                        QuantEncode.quantizeChunk(
                            type: type, src: src.baseAddress!,
                            dst: dst.baseAddress!, start: start,
                            rows: rows, columns: columns,
                            imatrix: im.baseAddress!)
                    }
                }
                return QuantEncode.quantizeChunk(
                    type: type, src: src.baseAddress!,
                    dst: dst.baseAddress!, start: start,
                    rows: rows, columns: columns, imatrix: nil)
            }
        }
        XCTAssertEqual(written, rows * rowBytes)
        return out
    }

    func testQ8_0MatchesReference() {
        XCTAssertEqual(encode(type: 8, rows: 2, columns: 64, weighted: false),
                       QuantEncodeFixtures.q8_0)
    }

    func testQ8_0StartOffsetMatchesReference() {
        // Row 1 only, via the element-offset start: row 0 stays zeroed.
        XCTAssertEqual(encode(type: 8, start: 64, rows: 1, columns: 64,
                              weighted: false),
                       QuantEncodeFixtures.q8_0_row1)
    }

    func testQ8KMatchesReference() {
        XCTAssertEqual(encode(type: 15, rows: 2, columns: 512, weighted: false),
                       QuantEncodeFixtures.q8_K)
    }

    func testQ4KReferenceVariantMatchesReference() {
        XCTAssertEqual(encode(type: 12, rows: 2, columns: 512, weighted: false),
                       QuantEncodeFixtures.q4_K_ref)
    }

    func testQ4KWeightedVariantMatchesReference() {
        XCTAssertEqual(encode(type: 12, rows: 2, columns: 512, weighted: true),
                       QuantEncodeFixtures.q4_K_weighted)
    }

    func testQ2KReferenceVariantMatchesReference() {
        XCTAssertEqual(encode(type: 10, rows: 2, columns: 512, weighted: false),
                       QuantEncodeFixtures.q2_K_ref)
    }

    func testQ2KWeightedVariantMatchesReference() {
        XCTAssertEqual(encode(type: 10, rows: 2, columns: 512, weighted: true),
                       QuantEncodeFixtures.q2_K_weighted)
    }

    func testIQ2XXSMatchesReference() {
        XCTAssertEqual(encode(type: 16, rows: 2, columns: 512, weighted: true),
                       QuantEncodeFixtures.iq2_xxs)
    }

    func testIQ2XXSRefusesMissingImatrix() {
        var out = [UInt8](repeating: 0, count: 1024)
        let written = Self.input.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBytes { dst in
                QuantEncode.quantizeChunk(type: 16, src: src.baseAddress!,
                                          dst: dst.baseAddress!, start: 0,
                                          rows: 1, columns: 512, imatrix: nil)
            }
        }
        XCTAssertEqual(written, 0)
    }

    func testTraits() {
        XCTAssertTrue([8, 10, 12, 15, 16].allSatisfy(QuantEncode.canQuantize))
        XCTAssertFalse(QuantEncode.canQuantize(1))   // f16 is not a target
        XCTAssertTrue(QuantEncode.requiresImatrix(16))
        XCTAssertFalse(QuantEncode.requiresImatrix(12))
        XCTAssertEqual(QuantEncode.rowSize(type: 16, columns: 512), 132)
        XCTAssertEqual(QuantEncode.rowSize(type: 12, columns: 512), 288)
    }
}
