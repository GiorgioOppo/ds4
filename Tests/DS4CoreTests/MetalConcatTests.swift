import XCTest
@testable import DS4Metal

/// Phase 9 (utility): validates the real metal/concat.metal kernel
/// (kernel_concat, dim 1) vs a CPU row concatenation.
final class MetalConcatTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/concat.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testConcatRowsDim1() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xCCAA
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let cols = 300, rows0 = 6, rows1 = 9
        var a = [Float](repeating: 0, count: rows0 * cols)
        var b = [Float](repeating: 0, count: rows1 * cols)
        for i in 0..<a.count { a[i] = rndF() }
        for i in 0..<b.count { b[i] = rndF() }

        let gpu = try rt.concatRows(a, rows0: rows0, b, rows1: rows1, cols: cols)
        XCTAssertEqual(gpu.count, (rows0 + rows1) * cols)

        for r in 0..<rows0 {
            for c in 0..<cols { XCTAssertEqual(gpu[r * cols + c], a[r * cols + c], "a row \(r) col \(c)") }
        }
        for r in 0..<rows1 {
            for c in 0..<cols { XCTAssertEqual(gpu[(rows0 + r) * cols + c], b[r * cols + c], "b row \(r) col \(c)") }
        }
    }
}
