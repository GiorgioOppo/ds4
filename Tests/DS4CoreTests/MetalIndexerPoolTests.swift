import XCTest
import Foundation
@testable import DS4Metal

/// Phase 9 / Stage A5: validates the real dsv4_misc.metal kernels
/// (indexer_weighted_sum, softmax_pool) vs CPU.
final class MetalIndexerPoolTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_misc.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testIndexerWeightedSum() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x1D71
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let nHead = 8, nTokens = 3, nComp = 40
        let scale: Float = 0.125
        var scores = [Float](repeating: 0, count: nHead * nTokens * nComp)
        var weights = [Float](repeating: 0, count: nTokens * nHead)
        for i in 0..<scores.count { scores[i] = rndF() }
        for i in 0..<weights.count { weights[i] = rndF() }

        let gpu = try rt.indexerWeightedSum(scores: scores, weights: weights, nHead: nHead, nTokens: nTokens, nComp: nComp, scale: scale)
        XCTAssertEqual(gpu.count, nTokens * nComp)
        for it in 0..<nTokens {
            for ic in 0..<nComp {
                var acc: Float = 0
                for ih in 0..<nHead {
                    let s = scores[ic + it*nComp + ih*nComp*nTokens]
                    let w = weights[ih + it*nHead]
                    acc += max(s, 0) * (w * scale)
                }
                XCTAssertEqual(gpu[it*nComp+ic], acc, accuracy: max(abs(acc),1)*1e-4, "iws it=\(it) ic=\(ic)")
            }
        }
    }

    func testSoftmaxPool() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x1D72
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let nRows = 5, width = 64
        var kv = [Float](repeating: 0, count: nRows * width)
        var score = [Float](repeating: 0, count: nRows)
        for i in 0..<kv.count { kv[i] = rndF() }
        for i in 0..<nRows { score[i] = rndF() }

        let gpu = try rt.softmaxPool(kv: kv, score: score, nRows: nRows, width: width)
        XCTAssertEqual(gpu.count, width)

        var m = -Float.infinity
        for r in 0..<nRows { m = max(m, score[r]) }
        var sum: Float = 0
        var w = [Float](repeating: 0, count: nRows)
        for r in 0..<nRows { w[r] = expf(score[r] - m); sum += w[r] }
        for d in 0..<width {
            var acc: Float = 0
            for r in 0..<nRows { acc += w[r] * kv[r*width+d] }
            let ref = acc / sum
            XCTAssertEqual(gpu[d], ref, accuracy: max(abs(ref),1)*1e-4, "pool d=\(d)")
        }
    }
}
