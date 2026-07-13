import XCTest
import Foundation
@testable import DS4Metal

/// Phase 9 / Stage A6: validates the real dsv4_hc.metal HC mixer split
/// (kernel_dsv4_hc_split_sinkhorn, HC=4) vs a CPU port of the exact arithmetic.
final class MetalHCSplitTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_hc.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private func sigmoid(_ z: Float) -> Float { 1 / (1 + expf(-z)) }
    private func twiceSigmoid(_ z: Float) -> Float { 2 / (1 + expf(-z)) }

    func testHCSplitSinkhorn() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x5C71
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let nRows = 4, iters = 3
        let eps: Float = 1e-3
        let scale: [Float] = [0.7, 1.1, 0.9]
        var base = [Float](repeating: 0, count: 24)
        for i in 0..<24 { base[i] = rndF() }
        var mix = [Float](repeating: 0, count: nRows * 24)
        for i in 0..<mix.count { mix[i] = rndF() }

        let gpu = try rt.hcSplitSinkhorn(mix: mix, scale: scale, base: base, nRows: nRows, sinkhornIters: iters, eps: eps)
        XCTAssertEqual(gpu.count, nRows * 24)

        for r in 0..<nRows {
            let mb = r * 24
            var out = [Float](repeating: 0, count: 24)
            for i in 0..<4 { out[i] = sigmoid(mix[mb+i] * scale[0] + base[i]) + eps }
            for i in 0..<4 { out[4+i] = twiceSigmoid(mix[mb+4+i] * scale[1] + base[4+i]) }
            // comb 4x4: rows
            var rrow = [[Float]](repeating: [Float](repeating: 0, count: 4), count: 4)
            for d in 0..<4 {
                for s in 0..<4 { rrow[d][s] = mix[mb+8+d*4+s] * scale[2] + base[8+d*4+s] }
                let m = rrow[d].max()!
                var sum: Float = 0
                for s in 0..<4 { rrow[d][s] = expf(rrow[d][s] - m); sum += rrow[d][s] }
                for s in 0..<4 { rrow[d][s] = rrow[d][s] / sum + eps }
            }
            // initial column normalize
            for s in 0..<4 {
                let cs = rrow[0][s] + rrow[1][s] + rrow[2][s] + rrow[3][s] + eps
                for d in 0..<4 { rrow[d][s] /= cs }
            }
            for _ in 1..<iters {
                for d in 0..<4 { let rs = rrow[d].reduce(0,+) + eps; for s in 0..<4 { rrow[d][s] /= rs } }
                for s in 0..<4 {
                    let cs = rrow[0][s] + rrow[1][s] + rrow[2][s] + rrow[3][s] + eps
                    for d in 0..<4 { rrow[d][s] /= cs }
                }
            }
            for d in 0..<4 { for s in 0..<4 { out[8+d*4+s] = rrow[d][s] } }

            for i in 0..<24 {
                XCTAssertEqual(gpu[mb+i], out[i], accuracy: max(abs(out[i]),1)*2e-4, "hc split r=\(r) i=\(i)")
            }
        }
    }
}
