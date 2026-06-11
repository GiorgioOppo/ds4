import XCTest
@testable import DS4Core

/// The software Float⇄binary16 fallback must match the hardware Float16
/// conversion bit-for-bit (round-to-nearest-even). On arm64 we can compare
/// directly against the hardware.
final class HalfTests: XCTestCase {

    func testEdgeCasesMatchHardware() throws {
        #if arch(arm64)
        let cases: [Float] = [
            0, -0.0, 1, -1, 0.5, 2, 65504,                  // exact values
            65519.96, 65520, 65537, -70000,                  // overflow boundary (RNE midpoint 65520)
            0x1p-14, 0x1.fp-15, 0x1p-24, 0x1p-25, 0x1.8p-25, // smallest normal / subnormals / ties
            5.960464477539063e-08, 1e-30,                    // min subnormal, underflow→0
            .infinity, -.infinity,
            3.14159265, -2.718281828, 1e4, 123.456,
            0.1, 0.2, 0.3, 1.0000001, 2049, 2047.5, 2048.5,  // mantissa-rounding ties
        ]
        for x in cases {
            XCTAssertEqual(Half.bitsSoftware(x), Float16(x).bitPattern,
                           "f32→f16 mismatch for \(x)")
        }
        // NaN: same class (quiet NaN), payload may differ.
        XCTAssertEqual(Half.bitsSoftware(.nan) & 0x7C00, 0x7C00)
        XCTAssertNotEqual(Half.bitsSoftware(.nan) & 0x03FF, 0)
        #else
        throw XCTSkip("hardware Float16 reference requires arm64")
        #endif
    }

    func testRandomMatchHardware() throws {
        #if arch(arm64)
        var rng = SystemRandomNumberGenerator()
        for _ in 0..<20_000 {
            let x = Float(bitPattern: UInt32.random(in: .min ... .max, using: &rng))
            if x.isNaN { continue }
            XCTAssertEqual(Half.bitsSoftware(x), Float16(x).bitPattern,
                           "f32→f16 mismatch for \(x) (bits 0x\(String(x.bitPattern, radix: 16)))")
        }
        // f16→f32: every one of the 65536 bit patterns must decode identically.
        for h in UInt16.min ... UInt16.max {
            let sw = Half.floatSoftware(h), hw = Float(Float16(bitPattern: h))
            if sw.isNaN && hw.isNaN { continue }
            XCTAssertEqual(sw.bitPattern, hw.bitPattern, "f16→f32 mismatch for 0x\(String(h, radix: 16))")
        }
        #else
        throw XCTSkip("hardware Float16 reference requires arm64")
        #endif
    }

    func testRoundTripExactValues() {
        // Every f16 value round-trips f16→f32→f16 exactly (no hardware needed).
        for h in UInt16.min ... UInt16.max {
            let f = Half.floatSoftware(h)
            if f.isNaN { continue }
            XCTAssertEqual(Half.bitsSoftware(f), h, "round-trip failed for 0x\(String(h, radix: 16))")
        }
    }
}
