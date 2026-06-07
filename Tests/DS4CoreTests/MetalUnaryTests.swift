import XCTest
import Foundation
@testable import DS4Metal

/// Phase 9 (utility): validates the real metal/unary.metal kernel
/// (kernel_unary_f32_f32) for silu/sigmoid/gelu/relu/clamp vs CPU references.
final class MetalUnaryTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/unary.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testUnaryActivations() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x1234
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 29) }

        let n = 777
        var x = [Float](repeating: 0, count: n)
        for i in 0..<n { x[i] = rndF() }

        let silu = try rt.unary(x, op: .silu, width: n)
        for i in 0..<n {
            let r = x[i] / (1 + expf(-x[i]))
            XCTAssertEqual(silu[i], r, accuracy: max(abs(r), 1) * 1e-4, "silu \(i)")
        }

        let sig = try rt.unary(x, op: .sigmoid, width: n)
        for i in 0..<n {
            let r = 1 / (1 + expf(-x[i]))
            XCTAssertEqual(sig[i], r, accuracy: 1e-4, "sigmoid \(i)")
        }

        let relu = try rt.unary(x, op: .relu, width: n)
        for i in 0..<n { XCTAssertEqual(relu[i], max(0, x[i]), accuracy: 1e-6, "relu \(i)") }

        let clamp = try rt.unary(x, op: .clamp, width: n, minV: -0.5, maxV: 0.5)
        for i in 0..<n { XCTAssertEqual(clamp[i], min(0.5, max(-0.5, x[i])), accuracy: 1e-6, "clamp \(i)") }

        let gelu = try rt.unary(x, op: .gelu, width: n)
        let coefA: Float = 0.044715, s2pi: Float = 0.79788456080286535587989211986876
        for i in 0..<n {
            let xv = x[i]
            let r = 0.5 * xv * (1 + tanhf(s2pi * xv * (1 + coefA * xv * xv)))
            XCTAssertEqual(gelu[i], r, accuracy: max(abs(r), 1) * 1e-4, "gelu \(i)")
        }
    }
}
