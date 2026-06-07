import XCTest
@testable import DS4Metal

/// Phase 9 (utility): validates the real set_rows (KV scatter) and repeat (HC
/// expansion) kernels vs CPU references.
final class MetalScatterTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/set_rows.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testSetRowsScatter() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x5E70
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let width = 257, nRows = 5, dstRows = 16
        var src = [Float](repeating: 0, count: nRows * width)
        for i in 0..<src.count { src[i] = rndF() }
        let rowIdx: [Int32] = [10, 0, 7, 3, 15]

        let gpu = try rt.setRows(src: src, rowIndices: rowIdx, dstRows: dstRows, width: width)
        XCTAssertEqual(gpu.count, dstRows * width)

        var ref = [Float](repeating: 0, count: dstRows * width)
        for i in 0..<nRows {
            let d = Int(rowIdx[i])
            for c in 0..<width { ref[d * width + c] = src[i * width + c] }
        }
        for i in 0..<ref.count { XCTAssertEqual(gpu[i], ref[i], "set_rows idx \(i)") }
    }

    func testRepeatHCExpansion() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xC0DE
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let nTokens = 3, nEmbd = 512, nHC = 4
        var src = [Float](repeating: 0, count: nTokens * nEmbd)
        for i in 0..<src.count { src[i] = rndF() }

        let gpu = try rt.repeatHC(src: src, nTokens: nTokens, nEmbd: nEmbd, nHC: nHC)
        XCTAssertEqual(gpu.count, nTokens * nHC * nEmbd)
        for t in 0..<nTokens {
            for hc in 0..<nHC {
                for e in 0..<nEmbd {
                    XCTAssertEqual(gpu[(t * nHC + hc) * nEmbd + e], src[t * nEmbd + e],
                                   "repeat t=\(t) hc=\(hc) e=\(e)")
                }
            }
        }
    }
}
