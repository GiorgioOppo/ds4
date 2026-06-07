import XCTest
@testable import DS4Metal

/// Phase 9 (utility): validates the real metal/cpy.metal kernels
/// (kernel_cpy_f32_f16 / kernel_cpy_f16_f32) via a F32 -> F16 -> F32 round-trip.
final class MetalCopyTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/cpy.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
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
}
