import XCTest
import Foundation
@testable import DS4Metal

/// Stage C: confirms the encode-form ropeTail and kvFP8Store (composed in a
/// GraphContext command buffer) match the standalone Stage-A wrappers (already
/// validated vs the C engine). This proves the graph encode-forms are faithful.
final class GraphQKVTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_rope.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    func testRopeEncodeMatchesWrapper() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xC0DEB0
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 28) }

        let nTok = 2, nHead = 4, headDim = 128, nRot = 64
        var x = [Float](repeating: 0, count: nTok * nHead * headDim)
        for i in 0..<x.count { x[i] = rndF() }

        let ref = try rt.ropeTail(x, nTok: nTok, nHead: nHead, headDim: headDim, nRot: nRot,
                                  nCtxOrig: 4096, inverse: false, freqBase: 10000, freqScale: 1,
                                  extFactor: 0, attnFactor: 1, betaFast: 32, betaSlow: 1, pos0: 5, posStep: 1)

        let ctx = GraphContext(rt)
        let xt = try GPUTensor.floats(rt, x)
        try ctx.begin()
        try ctx.ropeTail(x: xt, nTok: nTok, nHead: nHead, headDim: headDim, nRot: nRot, nCtxOrig: 4096,
                         freqBase: 10000, freqScale: 1, extFactor: 0, attnFactor: 1, betaFast: 32, betaSlow: 1,
                         pos0: 5, posStep: 1)
        ctx.commit()
        let got = xt.floatArray(x.count)
        for i in 0..<x.count { XCTAssertEqual(got[i], ref[i], accuracy: 1e-5, "rope \(i)") }
    }

    func testKVFP8StoreEncodeMatchesWrapper() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0xC0DEB1
        func rndF() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 26) }

        let headDim = 512, nRot = 64, rawCap = 8, rawRow = 3
        var kv = [Float](repeating: 0, count: headDim)
        for i in 0..<headDim { kv[i] = rndF() }
        let rawCache = [Float](repeating: 0, count: rawCap * headDim)

        let (refKv, refRaw) = try rt.kvFP8Store(kv: kv, rawCache: rawCache, headDim: headDim, nRot: nRot, rawRow: rawRow, rawCap: rawCap)

        let ctx = GraphContext(rt)
        let kvt = try GPUTensor.floats(rt, kv)
        let rawt = try GPUTensor.floats(rt, rawCache)
        try ctx.begin()
        try ctx.kvFP8Store(kv: kvt, rawCache: rawt, headDim: headDim, nRot: nRot, rawRow: rawRow)
        ctx.commit()
        let gotKv = kvt.floatArray(headDim)
        let gotRaw = rawt.floatArray(rawCap * headDim)
        for i in 0..<headDim { XCTAssertEqual(gotKv[i], refKv[i], accuracy: 1e-5, "kv \(i)") }
        for i in 0..<(rawCap*headDim) { XCTAssertEqual(gotRaw[i], refRaw[i], accuracy: 1e-5, "raw \(i)") }
    }
}
