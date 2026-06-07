import XCTest
import Foundation
@testable import DS4Metal

/// Stage C: smoke test of the full decodeLayer wiring on a tiny synthetic config.
/// Confirms the composition executes end-to-end (all kernels dispatch, buffer
/// sizes are consistent) and produces finite output of the right shape. Numerical
/// fidelity is gated on the real model (>=64GB) per the plan.
final class GraphDecodeLayerTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_hc.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    private var seed: UInt64 = 0xDEC0DE
    private func rf() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }
    private func q8(_ rows: Int, _ cols: Int) -> [UInt8] {
        var out = [UInt8](); out.reserveCapacity(rows*(cols/32)*34)
        for _ in 0..<rows {
            var c = 0
            while c < cols {
                var amax: Float = 0; var blk = [Float](repeating: 0, count: 32)
                for i in 0..<32 { blk[i] = rf(); amax = max(amax, abs(blk[i])) }
                let d = amax/127.0
                withUnsafeBytes(of: Float16(d).bitPattern.littleEndian) { out.append($0[0]); out.append($0[1]) }
                for i in 0..<32 { out.append(UInt8(bitPattern: Int8(clamping: d != 0 ? Int((blk[i]/d).rounded()) : 0))) }
                c += 32
            }
        }
        return out
    }

    func testDecodeLayerSmoke() throws {
        let rt = try makeRuntime()
        var d = DSV4Dims(nEmbd: 512, nHC: 4, headDim: 512, nHead: 2, qRank: 256, qDim: 1024,
                         sharedFfn: 512, nExperts: 256, expertFfn: 256, k: 6, nRot: 64, vocab: 1000, nOutGroup: 2, nLoraO: 128)
        let hcDim = d.nHC * d.nEmbd
        let nKeys = 32, pos = 0

        func tf(_ n: Int) throws -> GPUTensor { var a = [Float](repeating: 0, count: n); for i in 0..<n { a[i] = rf() }; return try GPUTensor.floats(rt, a) }
        func tf16(_ n: Int) throws -> GPUTensor { var a = [UInt16](repeating: 0, count: n); for i in 0..<n { a[i] = Float16(rf()).bitPattern }; return try GPUTensor.bytes(rt, a.withUnsafeBytes { Array($0) }, elementCount: n) }
        func tq8(_ rows: Int, _ cols: Int) throws -> GPUTensor { let b = q8(rows, cols); return try GPUTensor.bytes(rt, b, elementCount: rows*cols) }
        func texp(_ experts: Int, _ rows: Int, _ inDim: Int) throws -> GPUTensor { try GPUTensor.zerosBytes(rt, byteLength: experts*rows*(inDim/256)*144) }

        let w = LayerWeights(
            hcAttnFn: try tf16(24 * hcDim), attnScale: try tf(3), attnBase: try tf(24), attnNorm: try tf(d.nEmbd),
            qA: try tq8(d.qRank, d.nEmbd), qANorm: try tf(d.qRank), qB: try tq8(d.qDim, d.qRank),
            kvW: try tq8(d.headDim, d.nEmbd), kvNorm: try tf(d.headDim), attnSinks: try tf(d.nHead), attnOutA: try tq8(d.nOutGroup*d.nLoraO, d.attnGroupDim), attnOut: try tq8(d.nEmbd, d.attnLowDim),
            hcFfnFn: try tf16(24 * hcDim), ffnScale: try tf(3), ffnBase: try tf(24), ffnNorm: try tf(d.nEmbd),
            sharedGate: try tq8(d.sharedFfn, d.nEmbd), sharedUp: try tq8(d.sharedFfn, d.nEmbd),
            sharedDown: try tq8(d.nEmbd, d.sharedFfn), routerW: try tq8(d.nExperts, d.nEmbd),
            expGate: try texp(d.nExperts, d.expertFfn, d.nEmbd), expUp: try texp(d.nExperts, d.expertFfn, d.nEmbd),
            expDown: try texp(d.nExperts, d.nEmbd, d.expertFfn))

        let s = try DecodeScratch(rt, d, maxKeys: nKeys)
        let rope = RopeParams(nCtxOrig: 4096, freqBase: 10000, freqScale: 1, extFactor: 0, attnFactor: 1, betaFast: 32, betaSlow: 1)
        let curHc = try tf(hcDim)
        let rawCache = try GPUTensor.zeros(rt, floatCount: nKeys * d.headDim)
        let outHc = try GPUTensor.zeros(rt, floatCount: hcDim)

        let ctx = GraphContext(rt)
        try ctx.begin()
        try ctx.decodeLayer(curHc: curHc, w: w, s: s, d: d, rope: rope, rawCache: rawCache,
                            nKeys: nKeys, pos: pos, outHc: outHc, rmsEps: 1e-5, hcEps: 1e-3)
        ctx.commit()

        let out = outHc.floatArray(hcDim)
        XCTAssertEqual(out.count, hcDim)
        var allFinite = true
        for v in out where !v.isFinite { allFinite = false; break }
        XCTAssertTrue(allFinite, "decode layer produced non-finite output")
    }
}
