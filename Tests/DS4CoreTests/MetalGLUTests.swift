import XCTest
import Foundation
@testable import DS4Metal

/// Phase 9: validates the Swift SwiGLU dispatch (real metal/glu.metal kernel).
final class MetalGLUTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/glu.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testSwiGLUMatchesReference() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x9A9A
        func rnd() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) // ~[-8,8]
        }
        for (n, alpha, limit) in [(2048, Float(1), Float(0)), (2048, Float(1), Float(10)), (1024, Float(1.5), Float(7))] {
            var gate = [Float](repeating: 0, count: n), up = [Float](repeating: 0, count: n)
            for i in 0..<n { gate[i] = rnd(); up[i] = rnd() }
            let gpu = try rt.swiglu(gate: gate, up: up, alpha: alpha, limit: limit)
            var cpu = [Float](repeating: 0, count: n)
            for i in 0..<n {
                var x0 = gate[i], x1 = up[i]
                if limit > 1e-6 { x0 = min(x0, limit); x1 = min(max(x1, -limit), limit) }
                let silu = x0 / (1 + expf(-x0))
                cpu[i] = silu * x1 * alpha
            }
            var maxAbs: Float = 0
            for i in 0..<n { maxAbs = max(maxAbs, abs(gpu[i] - cpu[i])) }
            XCTAssertLessThan(maxAbs, 1e-4, "SwiGLU n=\(n) alpha=\(alpha) limit=\(limit) maxAbs=\(maxAbs)")
        }
    }
}
