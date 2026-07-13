import XCTest
import Foundation
@testable import DS4Metal

/// Validates the indexer relevance score (kernel_dsv4_indexer_score_one_direct)
/// used by the >4096-token sparse path to rank compressed rows before top-k.
/// score[row] = scale * sum_head max(q[head]·k_row, 0) * weights[head].
final class MetalIndexerScoreTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"
    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_misc.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testIndexerScoreVsCPU() throws {
        let rt = try makeRuntime()
        let nHead = 64, headDim = 128, nComp = 600
        var seed: UInt64 = 0x1DEC
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }
        let q = (0..<(nHead*headDim)).map { _ in rnd() }
        let weights = (0..<nHead).map { _ in rnd() }
        let comp = (0..<(nComp*headDim)).map { _ in rnd() }
        let scale: Float = 1.0 / Float(headDim * nHead).squareRoot()

        let gpu = try rt.indexerScoreOne(q: q, weights: weights, indexComp: comp,
                                         nComp: nComp, nHead: nHead, headDim: headDim, scale: scale)
        XCTAssertEqual(gpu.count, nComp)

        var maxAbs: Float = 0, maxRef: Float = 0
        for row in 0..<nComp {
            var acc: Float = 0
            for h in 0..<nHead {
                var s: Float = 0
                for d in 0..<headDim { s += q[h*headDim+d] * comp[row*headDim+d] }
                acc += max(s, 0) * weights[h] * scale
            }
            maxAbs = max(maxAbs, abs(gpu[row] - acc)); maxRef = max(maxRef, abs(acc))
        }
        XCTAssertLessThan(maxAbs, max(1e-3, 1e-3 * maxRef), "indexer score diverges (maxAbs=\(maxAbs), maxRef=\(maxRef))")
    }
}
