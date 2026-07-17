import XCTest
@testable import DS4Metal

/// Phase 9 (utility): validates the real metal/cpy.metal kernels
/// (kernel_cpy_f32_f16 / kernel_cpy_f16_f32) via a F32 -> F16 -> F32 round-trip.
final class MetalCopyTests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        do { return try MetalRuntime() }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testCpyF32F16RoundTrip() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xC0FFEE
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let n = 1024
        var x = [Float](repeating: 0, count: n)
        for i in 0..<n { x[i] = rndF() }

        let half = try rt.cpyF32toF16(x)
        XCTAssertEqual(half.count, n)
        // Each element must equal the canonical Float16(Float) conversion.
        for i in 0..<n {
            XCTAssertEqual(half[i], Float16(x[i]).bitPattern, "f32->f16 mismatch at \(i)")
        }

        let back = try rt.cpyF16toF32(half)
        XCTAssertEqual(back.count, n)
        for i in 0..<n {
            XCTAssertEqual(back[i], Float(Float16(bitPattern: half[i])), "f16->f32 mismatch at \(i)")
        }
    }

    func testVectorizedContiguousCopiesMatchGenericIncludingScalarTails() throws {
        let rt = try makeRuntime()
        for n in [1, 3, 4, 5, 31, 32, 1025] {
            var values = [Float](repeating: 0, count: n)
            for i in values.indices {
                values[i] = Float((i * 7919 + n * 17) % 65521 - 32760) / 97.0
            }

            let genericHalf = try rt.cpyF32toF16(values)
            let vectorHalf = try rt.cpyF32toF16Vectorized(values)
            XCTAssertEqual(vectorHalf, genericHalf, "F32->F16 vector mismatch for n=\(n)")

            let genericFloat = try rt.cpyF16toF32(genericHalf)
            let vectorFloat = try rt.cpyF16toF32Vectorized(genericHalf)
            XCTAssertEqual(vectorFloat.map(\.bitPattern), genericFloat.map(\.bitPattern),
                           "F16->F32 vector mismatch for n=\(n)")
        }
    }

    func testVectorizedF16BitCopyPreservesEveryEncodingUsedByCaches() throws {
        let rt = try makeRuntime()
        let patterns: [UInt16] = [
            0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00,
            0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x7e01, 0x7fff,
        ]
        XCTAssertEqual(try rt.cpyF16BitsVectorized(patterns), patterns)
    }
}
