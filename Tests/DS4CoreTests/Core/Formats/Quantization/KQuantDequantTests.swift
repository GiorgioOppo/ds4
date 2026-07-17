import XCTest
@testable import DS4Core

/// Bit-equality tests for the K-quant DEQUANT references (Q2_K, Q5_K, Q6_K):
/// blocks are hand-assembled field by field with uniform bit patterns, so
/// every element's expected value is computable in closed form and any drift
/// in field offsets, plane order or bit extraction changes the compared
/// output. These formats have no local quantizer — the bytes are the fixture.
final class KQuantDequantTests: XCTestCase {
    // MARK: - Q2_K (84 B: scales[16] | qs[64] | d f16 | dmin f16)

    /// qs bytes are 0b11100100: plane p (bits 2p..2p+1) decodes to value p.
    /// scales[g] = (g+1) low nibble (scale), 2 high nibble (min).
    private func makeQ2KBlock(d: Float, dmin: Float) -> [UInt8] {
        var block = [UInt8]()
        for g in 0..<16 { block.append(UInt8(((g + 1) & 0xF) | (2 << 4))) }
        block.append(contentsOf: [UInt8](repeating: 0b1110_0100, count: 64))
        withUnsafeBytes(of: Half.bits(d).littleEndian) { block.append(contentsOf: $0) }
        withUnsafeBytes(of: Half.bits(dmin).littleEndian) { block.append(contentsOf: $0) }
        return block
    }

    func testQ2KDequantMatchesHandComputedPlanes() {
        let d: Float = 0.5, dmin: Float = 0.25
        var bytes = makeQ2KBlock(d: d, dmin: dmin)
        bytes += makeQ2KBlock(d: 1.0, dmin: 0)   // second super-block: stride proof

        var out = [Float](repeating: .nan, count: 512)
        bytes.withUnsafeBytes {
            Quantize.dequantQ2_K($0.baseAddress!, count: 512, into: &out)
        }

        // Element index = half + 32*plane + l; its 2-bit value IS the plane.
        for (half, plane, l) in [(0, 0, 0), (0, 1, 7), (0, 2, 31), (0, 3, 16),
                                 (128, 0, 15), (128, 2, 5), (128, 3, 31)] {
            let index = half + 32 * plane + l
            let group = index / 16
            let expected = d * Float((group + 1) & 0xF) * Float(plane) - dmin * 2
            XCTAssertEqual(out[index], expected,
                           "Q2_K element \(index) (plane \(plane)) diverges")
        }
        // Second super-block: d=1, dmin=0 → element (plane 2, l 0), group 4,
        // scale nibble 5, plane value 2.
        XCTAssertEqual(out[256 + 64], 1.0 * 5 * 2)
    }

    // MARK: - Q5_K (176 B: d | dmin | scales[12] | qh[32] | qs[128])

    func testQ5KDequantAddsHighBitPerGroup() {
        let d: Float = 0.5, dmin: Float = 0.25
        var block = [UInt8]()
        withUnsafeBytes(of: Half.bits(d).littleEndian) { block.append(contentsOf: $0) }
        withUnsafeBytes(of: Half.bits(dmin).littleEndian) { block.append(contentsOf: $0) }
        let scales: [UInt8] = [3, 7, 11, 15, 2, 4, 6, 8, 0x35, 0x7A, 0x1C, 0x59]
        block.append(contentsOf: scales)
        // qh bit g set ⇔ group g even → +16 only on even groups, every l.
        block.append(contentsOf: [UInt8](repeating: 0b0101_0101, count: 32))
        // low nibble 1, high nibble 2 → base q is 1 (even j) / 2 (odd j).
        block.append(contentsOf: [UInt8](repeating: 0x21, count: 128))

        var out = [Float](repeating: .nan, count: 256)
        block.withUnsafeBytes {
            Quantize.dequantQ5_K($0.baseAddress!, count: 256, into: &out)
        }

        for j in 0..<8 {
            let (sc, m) = scales.withUnsafeBufferPointer {
                Quantize.scaleMinK4(j, $0.baseAddress!)
            }
            let base = (j % 2 == 0) ? 1 : 2
            let q = Float(base + (j % 2 == 0 ? 16 : 0))
            let expected = d * Float(sc) * q - dmin * Float(m)
            for l in [0, 13, 31] {
                XCTAssertEqual(out[32 * j + l], expected,
                               "Q5_K group \(j) element \(l) diverges")
            }
        }
    }

    // MARK: - Q6_K (210 B: ql[128] | qh[64] | int8 scales[16] | d f16)

    func testQ6KDequantComposesSixBitsAndSignedScales() {
        let d: Float = 0.5
        var block = [UInt8](repeating: 0x51, count: 128)   // low 1, high 5
        // qh 0b10011100: quarter high bits 0, 3, 1, 2.
        block.append(contentsOf: [UInt8](repeating: 0b1001_1100, count: 64))
        for k in 0..<16 { block.append(UInt8(bitPattern: Int8(k - 8))) }
        withUnsafeBytes(of: Half.bits(d).littleEndian) { block.append(contentsOf: $0) }
        XCTAssertEqual(block.count, 210)

        var out = [Float](repeating: .nan, count: 256)
        block.withUnsafeBytes {
            Quantize.dequantQ6_K($0.baseAddress!, count: 256, into: &out)
        }

        // q per quarter: (1|0<<4)-32=-31, (1|3<<4)-32=17, (5|1<<4)-32=-11,
        // (5|2<<4)-32=5; scale index = half/16 + l/16 + {0,2,4,6}.
        let quarterQ: [Float] = [-31, 17, -11, 5]
        for half in [0, 128] {
            for quarter in 0..<4 {
                for l in [0, 15, 16, 31] {
                    let index = half + 32 * quarter + l
                    let scale = Float(half / 16 + l / 16 + quarter * 2 - 8)
                    XCTAssertEqual(out[index], d * scale * quarterQ[quarter],
                                   "Q6_K element \(index) diverges")
                }
            }
        }
    }

    // MARK: - Single-byte perturbations (pin per-byte quant indexing)

    // The uniform fixtures above pin field offsets, scales and emission order
    // but not the quant byte indexing (identical bytes hide index bugs).
    // Flipping ONE byte must change exactly the elements that byte encodes.

    func testQ2KSingleByteFlipChangesExactlyItsFourPlanes() {
        let d: Float = 0.5, dmin: Float = 0.25
        let uniform = makeQ2KBlock(d: d, dmin: dmin)
        var flipped = uniform
        // qs index 37 = second half (base 32) + l 5: elements 128+32p+5.
        flipped[16 + 37] = 0b0001_1011   // planes decode 3, 2, 1, 0

        var base = [Float](repeating: 0, count: 256)
        var out = [Float](repeating: 0, count: 256)
        uniform.withUnsafeBytes {
            Quantize.dequantQ2_K($0.baseAddress!, count: 256, into: &base)
        }
        flipped.withUnsafeBytes {
            Quantize.dequantQ2_K($0.baseAddress!, count: 256, into: &out)
        }

        let newPlaneValue: [Float] = [3, 2, 1, 0]
        for index in 0..<256 {
            if index >= 128 && index % 32 == 5 {
                let plane = (index - 128) / 32
                let scale = Float(((index / 16 + 1) & 0xF))
                XCTAssertEqual(out[index],
                               d * scale * newPlaneValue[plane] - dmin * 2,
                               "flipped Q2_K element \(index) wrong")
            } else {
                XCTAssertEqual(out[index], base[index],
                               "Q2_K element \(index) changed unexpectedly")
            }
        }
    }

    func testQ5KSingleByteFlipsHitOnlyTheirElements() {
        func makeBlock(qsByte41: UInt8, qhByte20: UInt8) -> [UInt8] {
            var block = [UInt8]()
            withUnsafeBytes(of: Half.bits(0.5).littleEndian) { block.append(contentsOf: $0) }
            withUnsafeBytes(of: Half.bits(0.25).littleEndian) { block.append(contentsOf: $0) }
            block.append(contentsOf: [3, 7, 11, 15, 2, 4, 6, 8, 0x35, 0x7A, 0x1C, 0x59])
            var qh = [UInt8](repeating: 0b0101_0101, count: 32)
            qh[20] = qhByte20
            block.append(contentsOf: qh)
            var qs = [UInt8](repeating: 0x21, count: 128)
            qs[41] = qsByte41
            block.append(contentsOf: qs)
            return block
        }

        var base = [Float](repeating: 0, count: 256)
        makeBlock(qsByte41: 0x21, qhByte20: 0b0101_0101).withUnsafeBytes {
            Quantize.dequantQ5_K($0.baseAddress!, count: 256, into: &base)
        }

        // qs[41] = chunk 1, l 9 → group 2 (low nibble) and group 3 (high).
        var out = [Float](repeating: 0, count: 256)
        makeBlock(qsByte41: 0x7C, qhByte20: 0b0101_0101).withUnsafeBytes {
            Quantize.dequantQ5_K($0.baseAddress!, count: 256, into: &out)
        }
        let scales: [UInt8] = [3, 7, 11, 15, 2, 4, 6, 8, 0x35, 0x7A, 0x1C, 0x59]
        let (sc2, m2) = scales.withUnsafeBufferPointer {
            Quantize.scaleMinK4(2, $0.baseAddress!)
        }
        let (sc3, m3) = scales.withUnsafeBufferPointer {
            Quantize.scaleMinK4(3, $0.baseAddress!)
        }
        for index in 0..<256 {
            switch index {
            case 32 * 2 + 9:   // low nibble 0xC, group 2 even → +16
                XCTAssertEqual(out[index],
                               0.5 * Float(sc2) * 28 - 0.25 * Float(m2))
            case 32 * 3 + 9:   // high nibble 0x7, group 3 odd → no bit
                XCTAssertEqual(out[index],
                               0.5 * Float(sc3) * 7 - 0.25 * Float(m3))
            default:
                XCTAssertEqual(out[index], base[index],
                               "Q5_K element \(index) changed unexpectedly")
            }
        }

        // qh[20] = 0 removes the +16 from every EVEN group at l 20 only.
        var noHigh = [Float](repeating: 0, count: 256)
        makeBlock(qsByte41: 0x21, qhByte20: 0).withUnsafeBytes {
            Quantize.dequantQ5_K($0.baseAddress!, count: 256, into: &noHigh)
        }
        for index in 0..<256 {
            let group = index / 32, l = index % 32
            if l == 20 && group % 2 == 0 {
                let (sc, m) = scales.withUnsafeBufferPointer {
                    Quantize.scaleMinK4(group, $0.baseAddress!)
                }
                XCTAssertEqual(noHigh[index],
                               0.5 * Float(sc) * 1 - 0.25 * Float(m),
                               "Q5_K high-bit removal wrong at \(index)")
            } else {
                XCTAssertEqual(noHigh[index], base[index],
                               "Q5_K element \(index) changed unexpectedly")
            }
        }
    }

    func testQ6KSingleByteFlipChangesItsLowAndHighQuarters() {
        func makeBlock(qlByte86: UInt8) -> [UInt8] {
            var block = [UInt8](repeating: 0x51, count: 128)
            block[86] = qlByte86           // second half, l 22
            block.append(contentsOf: [UInt8](repeating: 0b1001_1100, count: 64))
            for k in 0..<16 { block.append(UInt8(bitPattern: Int8(k - 8))) }
            withUnsafeBytes(of: Half.bits(0.5).littleEndian) { block.append(contentsOf: $0) }
            return block
        }

        var base = [Float](repeating: 0, count: 256)
        makeBlock(qlByte86: 0x51).withUnsafeBytes {
            Quantize.dequantQ6_K($0.baseAddress!, count: 256, into: &base)
        }
        var out = [Float](repeating: 0, count: 256)
        makeBlock(qlByte86: 0x2F).withUnsafeBytes {
            Quantize.dequantQ6_K($0.baseAddress!, count: 256, into: &out)
        }

        // ql[86] feeds quarter 1 (low nibble) and quarter 3 (high nibble) of
        // half 128 at l 22 (sub-block 1): qh stays 0b10011100.
        // q1 = (0xF | 0<<4) - 32 = -17, scale index 8+1+0 → 1
        // q3 = (0x2 | 1<<4) - 32 = -14, scale index 8+1+4 → 5
        for index in 0..<256 {
            switch index {
            case 128 + 22:
                XCTAssertEqual(out[index], 0.5 * 1 * -17)
            case 128 + 64 + 22:
                XCTAssertEqual(out[index], 0.5 * 5 * -14)
            default:
                XCTAssertEqual(out[index], base[index],
                               "Q6_K element \(index) changed unexpectedly")
            }
        }
    }

    func testQ6KSecondSuperblockUsesItsOwnFields() {
        // Block 1: all-zero ql/qh (q = -32), scale 1 everywhere, d = 1.
        var first = [UInt8](repeating: 0, count: 192)
        for _ in 0..<16 { first.append(UInt8(bitPattern: Int8(1))) }
        withUnsafeBytes(of: Half.bits(1.0).littleEndian) { first.append(contentsOf: $0) }
        // Block 2: identical but d = 0.25.
        var second = [UInt8](repeating: 0, count: 192)
        for _ in 0..<16 { second.append(UInt8(bitPattern: Int8(1))) }
        withUnsafeBytes(of: Half.bits(0.25).littleEndian) { second.append(contentsOf: $0) }

        var out = [Float](repeating: .nan, count: 512)
        (first + second).withUnsafeBytes {
            Quantize.dequantQ6_K($0.baseAddress!, count: 512, into: &out)
        }
        XCTAssertEqual(out[0], -32)
        XCTAssertEqual(out[255], -32)
        XCTAssertEqual(out[256], -8)
        XCTAssertEqual(out[511], -8)
    }
}
