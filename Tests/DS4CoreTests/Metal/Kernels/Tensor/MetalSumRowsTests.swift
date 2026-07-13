import XCTest
@testable import DS4Metal

/// Phase 9 (utility): validates the real metal/sum_rows.metal kernel
/// (kernel_sum_rows_f32_f32) for both sum and mean reductions vs CPU.
final class MetalSumRowsTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/sum_rows.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testSumAndMeanRows() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x5A5A
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let width = 300, rows = 17
        var x = [Float](repeating: 0, count: width * rows)
        for i in 0..<x.count { x[i] = rndF() }

        let gpuSum = try rt.sumRows(x, width: width, rows: rows, mean: false)
        let gpuMean = try rt.sumRows(x, width: width, rows: rows, mean: true)
        XCTAssertEqual(gpuSum.count, rows)

        for r in 0..<rows {
            var s: Float = 0
            for c in 0..<width { s += x[r * width + c] }
            XCTAssertEqual(gpuSum[r], s, accuracy: max(abs(s), 1) * 1e-4, "sum row \(r)")
            XCTAssertEqual(gpuMean[r], s / Float(width), accuracy: max(abs(s / Float(width)), 1) * 1e-4, "mean row \(r)")
        }
    }
}
