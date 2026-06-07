import XCTest
@testable import DS4Metal

/// Phase 9: validates the Swift RMSNorm dispatch (real metal/norm.metal kernels)
/// against a CPU reference.
final class MetalNormTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/norm.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private func refRMSNorm(_ x: [Float], rows: Int, n: Int, eps: Float, weight: [Float]?) -> [Float] {
        var out = [Float](repeating: 0, count: rows * n)
        for r in 0..<rows {
            var ss: Float = 0
            for i in 0..<n { let v = x[r*n+i]; ss += v * v }
            let scale = 1.0 / (ss / Float(n) + eps).squareRoot()
            for i in 0..<n {
                var v = x[r*n+i] * scale
                if let w = weight { v *= w[i] }
                out[r*n+i] = v
            }
        }
        return out
    }

    func testRMSNormMatchesReference() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xC0FFEE
        func rnd() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30)
        }
        let eps: Float = 1e-6
        for (n, rows) in [(4096, 1), (512, 8), (128, 33), (4, 100)] {
            var x = [Float](repeating: 0, count: rows * n)
            for i in 0..<x.count { x[i] = rnd() }
            var w = [Float](repeating: 0, count: n)
            for i in 0..<n { w[i] = rnd() }

            for weight in [Optional<[Float]>.none, .some(w)] {
                let gpu = try rt.rmsNorm(x, rows: rows, n: n, eps: eps, weight: weight)
                let cpu = refRMSNorm(x, rows: rows, n: n, eps: eps, weight: weight)
                var maxRel: Float = 0
                for i in 0..<rows*n {
                    maxRel = max(maxRel, abs(gpu[i] - cpu[i]) / max(abs(cpu[i]), 0.5))
                }
                XCTAssertLessThan(maxRel, 1e-3,
                                  "RMSNorm n=\(n) rows=\(rows) weighted=\(weight != nil) maxRel=\(maxRel)")
            }
        }
    }
}
