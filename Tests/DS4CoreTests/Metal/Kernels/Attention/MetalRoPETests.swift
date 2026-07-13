import XCTest
import Foundation
@testable import DS4Metal

/// Phase 9: validates the Swift RoPE dispatch (real metal/dsv4_rope.metal kernel)
/// against a CPU reference replicating the same YaRN partial rotation.
final class MetalRoPETests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"

    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_rope.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    // CPU port of the kernel math (mode 0 / non-neox, pow path).
    private func ref(_ x: [Float], nTok: Int, nHead: Int, headDim: Int, nRot: Int, nCtxOrig: Int,
                     inverse: Bool, freqBase: Float, freqScale: Float, extFactor: Float,
                     attnFactor: Float, betaFast: Float, betaSlow: Float,
                     pos0: Int, posStep: Int) -> [Float] {
        func corrFactor(_ nr: Float) -> Float {
            Float(nRot) * logf(Float(nCtxOrig) / (nr * 2 * Float.pi)) / (2 * logf(freqBase))
        }
        let low = max(0, floorf(corrFactor(betaFast)))
        let high = min(Float(nRot) - 1, ceilf(corrFactor(betaSlow)))
        func ramp(_ i0: Int) -> Float {
            let y = (Float(i0 / 2) - low) / max(0.001, high - low)
            return 1 - min(1, max(0, y))
        }
        func yarn(_ thetaExtrap: Float, _ i0: Int) -> (Float, Float) {
            let thetaInterp = freqScale * thetaExtrap
            var theta = thetaInterp
            var mscale = attnFactor
            if extFactor != 0 {
                let mix = ramp(i0) * extFactor
                theta = thetaInterp * (1 - mix) + thetaExtrap * mix
                mscale *= 1 + 0.1 * logf(1 / freqScale)
            }
            return (cosf(theta) * mscale, sinf(theta) * mscale)
        }

        var out = x
        let nNope = headDim - nRot
        let invN = -1.0 / Float(nRot)
        for t in 0..<nTok {
            let thetaBase = Float(pos0 + t * posStep)
            for h in 0..<nHead {
                let base = (t * nHead + h) * headDim
                var r = 0
                while r < nRot {
                    let theta = thetaBase * powf(freqBase, invN * Float(r))
                    var (c, s) = yarn(theta, r)
                    if inverse { s = -s }
                    let j0 = base + nNope + r, j1 = j0 + 1
                    let x0 = x[j0], x1 = x[j1]
                    out[j0] = x0 * c - x1 * s
                    out[j1] = x0 * s + x1 * c
                    r += 2
                }
            }
        }
        return out
    }

    func testRoPEMatchesReference() throws {
        let rt = try makeRuntime()
        var seed: UInt64 = 0x5151
        func rnd() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30)
        }
        let headDim = 512, nRot = 64, nCtxOrig = 65536
        for cfg in [(extFactor: Float(0), inverse: false, freqScale: Float(1.0)),
                    (extFactor: Float(1), inverse: false, freqScale: Float(0.0625)),
                    (extFactor: Float(1), inverse: true,  freqScale: Float(0.0625))] {
            let nTok = 5, nHead = 2
            var x = [Float](repeating: 0, count: nTok * nHead * headDim)
            for i in 0..<x.count { x[i] = rnd() }

            let gpu = try rt.ropeTail(x, nTok: nTok, nHead: nHead, headDim: headDim, nRot: nRot,
                                      nCtxOrig: nCtxOrig, inverse: cfg.inverse,
                                      freqBase: 10000, freqScale: cfg.freqScale, extFactor: cfg.extFactor,
                                      attnFactor: 1.0, betaFast: 32, betaSlow: 1, pos0: 7, posStep: 1)
            let cpu = ref(x, nTok: nTok, nHead: nHead, headDim: headDim, nRot: nRot, nCtxOrig: nCtxOrig,
                          inverse: cfg.inverse, freqBase: 10000, freqScale: cfg.freqScale,
                          extFactor: cfg.extFactor, attnFactor: 1.0, betaFast: 32, betaSlow: 1,
                          pos0: 7, posStep: 1)
            var maxAbs: Float = 0
            for i in 0..<x.count { maxAbs = max(maxAbs, abs(gpu[i] - cpu[i])) }
            XCTAssertLessThan(maxAbs, 2e-4, "RoPE ext=\(cfg.extFactor) inv=\(cfg.inverse) maxAbs=\(maxAbs)")
        }
    }
}
