import XCTest
import Foundation
@testable import DS4Metal

/// Phase 9: validates the Swift RoPE dispatch (real metal/dsv4_rope.metal kernel)
/// against a CPU reference replicating the same YaRN partial rotation.
final class MetalRoPETests: XCTestCase {
    private func makeRuntime() throws -> MetalRuntime {
        // Exercise the same embedded kernels used by the demo and GUI, rather
        // than a developer-specific checkout that may contain older shaders.
        do { return try MetalRuntime() }
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

    /// Baseline, pair-only and affine-position kernels must agree bit-for-bit.
    /// Shapes cover single-token decode, multi-token prefill, inverse RoPE and
    /// both ordinary and YaRN-scaled parameters on the actual local GPU.
    func testOptimizedRoPEMatchesBaselineExactly() throws {
        let rt = try makeRuntime()
        struct Case {
            let nTok: Int
            let nHead: Int
            let headDim: Int
            let pos0: Int
            let posStep: Int
            let inverse: Bool
            let extended: Bool
        }
        let cases = [
            Case(nTok: 1, nHead: 64, headDim: 512, pos0: 2_047,
                 posStep: 1, inverse: false, extended: true),
            Case(nTok: 1, nHead: 1, headDim: 512, pos0: 65_533,
                 posStep: 1, inverse: true, extended: true),
            Case(nTok: 7, nHead: 4, headDim: 128, pos0: 37,
                 posStep: 3, inverse: false, extended: false),
            Case(nTok: 33, nHead: 7, headDim: 512, pos0: 2_047,
                 posStep: 1, inverse: true, extended: true),
        ]
        var seed: UInt64 = 0x726f70655f6162
        func nextFloat() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: seed >> 32)) / Float(Int32.max)
        }

        for (caseIndex, c) in cases.enumerated() {
            let nRot = 64
            var input = [Float](repeating: 0,
                                count: c.nTok * c.nHead * c.headDim)
            for i in input.indices {
                input[i] = nextFloat()
                // Ensure the untouched prefix preserves signed zero exactly.
                if i % 257 == 0 { input[i] = Float(bitPattern: 0x8000_0000) }
            }
            let freqBase: Float = c.extended ? 160_000 : 10_000
            let freqScale: Float = c.extended ? 1.0 / 16.0 : 1
            let extFactor: Float = c.extended ? 1 : 0
            let nCtxOrig = c.extended ? 65_536 : 0
            let attnFactor: Float = c.extended
                ? 1 / (1 + 0.1 * logf(1 / freqScale))
                : 1

            func run(_ mode: RoPEKernelMode) throws -> [Float] {
                try rt.ropeTail(
                    input, nTok: c.nTok, nHead: c.nHead,
                    headDim: c.headDim, nRot: nRot,
                    nCtxOrig: nCtxOrig, inverse: c.inverse,
                    freqBase: freqBase, freqScale: freqScale,
                    extFactor: extFactor, attnFactor: attnFactor,
                    betaFast: 32, betaSlow: 1, pos0: c.pos0,
                    posStep: c.posStep, kernelMode: mode)
            }

            let baseline = try run(.baseline)
            let pair = try run(.pair)
            let affine = try run(.affine)
            for i in input.indices {
                XCTAssertEqual(pair[i].bitPattern, baseline[i].bitPattern,
                               "pair mismatch case=\(caseIndex) index=\(i)")
                XCTAssertEqual(affine[i].bitPattern, baseline[i].bitPattern,
                               "affine mismatch case=\(caseIndex) index=\(i)")
            }
        }
    }

    func testOptimizedRoPEFallsBackForLaneIncompatibleShape() {
        XCTAssertEqual(MetalRuntime.resolveRoPEKernelMode(
            .pair, nTok: 4, headDim: 130, nRot: 64), .baseline)
        XCTAssertEqual(MetalRuntime.resolveRoPEKernelMode(
            .affine, nTok: 4, headDim: 130, nRot: 64), .baseline)
    }
}
