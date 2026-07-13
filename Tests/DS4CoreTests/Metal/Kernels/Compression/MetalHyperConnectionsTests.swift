import XCTest
@testable import DS4Metal

/// Phase 9 (hyper-connections): validates the real metal/dsv4_hc.metal kernels
/// kernel_dsv4_hc_expand4 (per-layer HC mix) and kernel_dsv4_hc_weighted_sum
/// (final HC collapse) against CPU references.
final class MetalHyperConnectionsTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_hc.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testHCWeightedSum() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x4C01
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let nEmbd = 320, nHC = 4, nTokens = 5
        var x = [Float](repeating: 0, count: nTokens * nHC * nEmbd)
        var w = [Float](repeating: 0, count: nTokens * nHC)
        for i in 0..<x.count { x[i] = rndF() }
        for i in 0..<w.count { w[i] = rndF() }

        let gpu = try rt.hcWeightedSum(x: x, weights: w, nEmbd: nEmbd, nHC: nHC, nTokens: nTokens)
        XCTAssertEqual(gpu.count, nTokens * nEmbd)
        for t in 0..<nTokens {
            for d in 0..<nEmbd {
                var acc: Float = 0
                for h in 0..<nHC { acc += x[(t * nHC + h) * nEmbd + d] * w[t * nHC + h] }
                XCTAssertEqual(gpu[t * nEmbd + d], acc, accuracy: max(abs(acc), 1) * 1e-4, "ws t=\(t) d=\(d)")
            }
        }
    }

    func testHCExpand4() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x4C02
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let nEmbd = 256, nHC = 4, nTokens = 3
        var blockOut = [Float](repeating: 0, count: nTokens * nEmbd)
        var blockAdd = [Float](repeating: 0, count: nTokens * nEmbd)
        var residual = [Float](repeating: 0, count: nTokens * nHC * nEmbd)
        var post = [Float](repeating: 0, count: nTokens * nHC)
        var comb = [Float](repeating: 0, count: nTokens * nHC * nHC) // [t][j][k], k innermost
        for i in 0..<blockOut.count { blockOut[i] = rndF() }
        for i in 0..<blockAdd.count { blockAdd[i] = rndF() }
        for i in 0..<residual.count { residual[i] = rndF() }
        for i in 0..<post.count { post[i] = rndF() }
        for i in 0..<comb.count { comb[i] = rndF() }

        let gpu = try rt.hcExpand4(blockOut: blockOut, residual: residual, post: post, comb: comb,
                                   nEmbd: nEmbd, nTokens: nTokens, blockAdd: blockAdd)
        XCTAssertEqual(gpu.count, nTokens * nHC * nEmbd)

        // CPU reference matching the kernel index math.
        // comb access: comb[k*1 + j*nHC + t*nHC*nHC]  (k innermost)
        // residual:    residual[d + j*nEmbd + t*nHC*nEmbd]
        for t in 0..<nTokens {
            for d in 0..<nEmbd {
                let bv = blockOut[t * nEmbd + d] + blockAdd[t * nEmbd + d]
                var r = [Float](repeating: 0, count: nHC)
                for j in 0..<nHC { r[j] = residual[d + j * nEmbd + t * nHC * nEmbd] }
                for k in 0..<nHC {
                    var acc = bv * post[k + t * nHC]
                    for j in 0..<nHC { acc += comb[k + j * nHC + t * nHC * nHC] * r[j] }
                    let got = gpu[d + k * nEmbd + t * nHC * nEmbd]
                    XCTAssertEqual(got, acc, accuracy: max(abs(acc), 1) * 1e-4, "exp t=\(t) k=\(k) d=\(d)")
                }
            }
        }
    }
}
