import XCTest
import Foundation
@testable import DS4Metal

/// Phase 9 (utility): validates the real metal/softmax.metal kernels
/// (kernel_soft_max_f32 and the float4 kernel_soft_max_f32_4) vs a CPU softmax.
final class MetalSoftmaxTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/softmax.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private func cpuSoftmax(_ x: [Float], width: Int, rows: Int) -> [Float] {
        var out = [Float](repeating: 0, count: x.count)
        for r in 0..<rows {
            let base = r * width
            var m = -Float.infinity
            for c in 0..<width { m = max(m, x[base + c]) }
            var s: Float = 0
            for c in 0..<width { let e = expf(x[base + c] - m); out[base + c] = e; s += e }
            for c in 0..<width { out[base + c] /= s }
        }
        return out
    }

    func testSoftmaxScalarAndVector() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x50F7
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        for width in [127, 256] { // 127 -> scalar kernel, 256 -> float4 kernel
            let rows = 9
            var x = [Float](repeating: 0, count: width * rows)
            for i in 0..<x.count { x[i] = rndF() }
            let gpu = try rt.softmax(x, width: width, rows: rows)
            let ref = cpuSoftmax(x, width: width, rows: rows)
            for i in 0..<x.count {
                XCTAssertEqual(gpu[i], ref[i], accuracy: 1e-5, "softmax width \(width) idx \(i)")
            }
            // rows sum to 1
            for r in 0..<rows {
                var s: Float = 0
                for c in 0..<width { s += gpu[r * width + c] }
                XCTAssertEqual(s, 1.0, accuracy: 1e-4, "softmax row \(r) sum width \(width)")
            }
        }
    }
}
