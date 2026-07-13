import XCTest
import Foundation
@testable import DS4Metal

/// Phase 9 / Stage A2: validates the real moe.metal fused FFN-tail kernels
/// (kernel_dsv4_moe_swiglu_weight, kernel_dsv4_moe_sum6_f32) vs CPU.
final class MetalMoEFusedTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/moe.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testSwiGLUWeight() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x5061
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let width = 333, rows = 6
        var gate = [Float](repeating: 0, count: rows * width)
        var up = [Float](repeating: 0, count: rows * width)
        var w = [Float](repeating: 0, count: rows)
        for i in 0..<gate.count { gate[i] = rndF() }
        for i in 0..<up.count { up[i] = rndF() }
        for i in 0..<rows { w[i] = rndF() }

        for clamp: Float in [0, 0.5] {
            let gpu = try rt.moeSwiGLUWeight(gate: gate, up: up, weights: w, width: width, rows: rows, clampValue: clamp)
            for r in 0..<rows {
                for i in 0..<width {
                    var g = gate[r*width+i], u = up[r*width+i]
                    if clamp > 1e-6 { g = min(g, clamp); u = min(clamp, max(-clamp, u)) }
                    let silu = g / (1 + expf(-g))
                    let ref = silu * u * w[r]
                    XCTAssertEqual(gpu[r*width+i], ref, accuracy: max(abs(ref),1)*1e-4, "swiglu_w clamp=\(clamp) r=\(r) i=\(i)")
                }
            }
        }
    }

    func testSum6() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x5062
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let width = 257, tokens = 4
        var experts = [Float](repeating: 0, count: tokens * 6 * width)
        for i in 0..<experts.count { experts[i] = rndF() }

        let gpu = try rt.moeSum6(experts: experts, width: width, tokens: tokens)
        XCTAssertEqual(gpu.count, tokens * width)
        for t in 0..<tokens {
            for c in 0..<width {
                var acc: Float = 0
                for e in 0..<6 { acc += experts[(t*6+e)*width + c] }
                XCTAssertEqual(gpu[t*width+c], acc, accuracy: max(abs(acc),1)*1e-5, "sum6 t=\(t) c=\(c)")
            }
        }
    }
}
