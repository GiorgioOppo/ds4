import XCTest
@testable import DS4Core

final class BonsaiQuantizationTests: XCTestCase {
    func testPQ2AllCoefficientsAndBlockBoundaries() throws {
        var bytes: [UInt8] = []
        for scale: Float16 in [0.5, -2] {
            bytes += [UInt8(truncatingIfNeeded: scale.bitPattern), UInt8(truncatingIfNeeded: scale.bitPattern >> 8)]
            bytes += (0..<32).map { UInt8(truncatingIfNeeded: $0 * 37) }
        }
        let values = try BonsaiQuantization.decodeRow(bytes, type: 142, columns: 256)
        for i in 0..<256 {
            let j = i % 128, code = (Int(bytes[i / 128 * 34 + 2 + j / 4]) >> (2 * (j % 4))) & 3
            XCTAssertEqual(values[i], (i < 128 ? 0.5 : -2) * Float(code - 1))
        }
        XCTAssertEqual(BonsaiQuantization.rowBytes(type: 142, columns: 5120), 1360)
        XCTAssertNil(BonsaiQuantization.rowBytes(type: 142, columns: 129))
        XCTAssertThrowsError(try BonsaiQuantization.decodeRow(Array(bytes.dropLast()), type: 142, columns: 256))
    }

    func testPTQAllByteCodesIncludingWrappingAndTail() throws {
        // Independent iterative byte wrapping covers every stored byte pattern
        // and both five-trit bands plus the four-trit qh tail.
        for code in 0...255 {
            var block = [UInt8](repeating: UInt8(code), count: 28)
            block[26] = 0; block[27] = 0x38 // half(0.5)
            let actual = try BonsaiQuantization.decodeRow(block, type: 143, columns: 128)
            for index in 0..<128 {
                let power = index < 80 ? index / 16 : (index < 120 ? (index - 80) / 8 : (index - 120) / 2)
                var wrapped = UInt8(code)
                for _ in 0..<power { wrapped = wrapped &* 3 }
                let expected = Float((Int(wrapped) * 3 / 256) - 1) * 0.5
                XCTAssertEqual(actual[index], expected, "byte \(code), index \(index)")
            }
        }
        XCTAssertEqual(BonsaiQuantization.rowBytes(type: 143, columns: 5120), 1120)
        XCTAssertNil(BonsaiQuantization.rowBytes(type: 143, columns: 0))
    }

    func testSignedHadamardInverseAndValidation() throws {
        let values = (0..<2048).map { Float(($0 * 17) % 113 - 56) / 64 }
        let signs = (0..<2048).map { Int32($0 % 7 < 3 ? -1 : 1) }
        let rotated = try BonsaiQuantization.hadamard(values, signs: signs)
        let restored = try BonsaiQuantization.hadamard(rotated, signs: signs, inverse: true)
        for i in values.indices { XCTAssertEqual(restored[i], values[i], accuracy: 1e-6) }
        let norm = values.reduce(Float(0)) { $0 + $1 * $1 }
        let transformedNorm = rotated.reduce(Float(0)) { $0 + $1 * $1 }
        XCTAssertEqual(transformedNorm, norm, accuracy: 0.002)
        XCTAssertThrowsError(try BonsaiQuantization.hadamard(Array(values.dropLast()), signs: signs))
        XCTAssertThrowsError(try BonsaiQuantization.hadamard(values, signs: [Int32](repeating: 0, count: values.count)))
    }
}
