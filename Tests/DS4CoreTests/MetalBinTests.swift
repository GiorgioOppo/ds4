import XCTest
@testable import DS4Metal

/// Phase 9 (utility): validates the real metal/bin.metal kernel
/// (kernel_bin_fuse_f32_f32_f32) for add/sub/mul/div vs CPU.
final class MetalBinTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/bin.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testBinaryOps() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xB1B1
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let width = 513, rows = 7
        let n = width * rows
        var a = [Float](repeating: 0, count: n), b = [Float](repeating: 0, count: n)
        for i in 0..<n { a[i] = rndF(); b[i] = rndF() + 2.0 } // b offset to avoid div by ~0

        let add = try rt.binary(a, b, op: .add, width: width, rows: rows)
        let sub = try rt.binary(a, b, op: .sub, width: width, rows: rows)
        let mul = try rt.binary(a, b, op: .mul, width: width, rows: rows)
        let div = try rt.binary(a, b, op: .div, width: width, rows: rows)
        for i in 0..<n {
            XCTAssertEqual(add[i], a[i] + b[i], accuracy: max(abs(a[i]+b[i]),1)*1e-5, "add \(i)")
            XCTAssertEqual(sub[i], a[i] - b[i], accuracy: max(abs(a[i]-b[i]),1)*1e-5, "sub \(i)")
            XCTAssertEqual(mul[i], a[i] * b[i], accuracy: max(abs(a[i]*b[i]),1)*1e-5, "mul \(i)")
            XCTAssertEqual(div[i], a[i] / b[i], accuracy: max(abs(a[i]/b[i]),1)*1e-5, "div \(i)")
        }
    }
}
