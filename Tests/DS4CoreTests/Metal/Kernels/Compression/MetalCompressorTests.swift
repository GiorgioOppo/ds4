import XCTest
@testable import DS4Metal

/// NSA compressor pool validation: GPU softmax-pool (real kernel_dsv4_softmax_pool,
/// per-dimension) vs a faithful CPU port of compressor_pool_decode_state (ds4.c:8423),
/// for both ratio-128 (single lane) and ratio-4 (two lane) layouts.
final class MetalCompressorTests: XCTestCase {
    static let metalDir = "/Users/oppog/Downloads/ds4-main/DS4-gui/metal"
    private func makeRuntime() throws -> MetalRuntime {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.metalDir + "/dsv4_misc.metal"),
                          "vendored metal kernels not present")
        do { return try MetalRuntime(metalDir: Self.metalDir) }
        catch { throw XCTSkip("Metal unavailable: \(error)") }
    }

    // CPU reference — compressor_pool_decode_state (ds4.c:8423).
    private func poolRef(stateKv: [Float], stateScore: [Float], headDim: Int, ratio: Int) -> [Float] {
        let coff = ratio == 4 ? 2 : 1
        let width = coff * headDim
        var out = [Float](repeating: 0, count: headDim)
        for j in 0..<headDim {
            var maxS = -Float.greatestFiniteMagnitude
            if ratio == 4 {
                for r in 0..<ratio {
                    maxS = max(maxS, stateScore[r * width + j])
                    maxS = max(maxS, stateScore[(ratio + r) * width + headDim + j])
                }
            } else {
                for r in 0..<ratio { maxS = max(maxS, stateScore[r * width + j]) }
            }
            var denom: Float = 0, sum: Float = 0
            if ratio == 4 {
                for r in 0..<ratio {
                    let wp = expf(stateScore[r * width + j] - maxS)
                    let wc = expf(stateScore[(ratio + r) * width + headDim + j] - maxS)
                    denom += wp + wc
                    sum += wp * stateKv[r * width + j]
                    sum += wc * stateKv[(ratio + r) * width + headDim + j]
                }
            } else {
                for r in 0..<ratio {
                    let w = expf(stateScore[r * width + j] - maxS)
                    denom += w
                    sum += w * stateKv[r * width + j]
                }
            }
            out[j] = denom > 0 ? sum / denom : 0
        }
        return out
    }

    private func run(ratio: Int, headDim: Int) throws {
        let rt = try makeRuntime()
        let coff = ratio == 4 ? 2 : 1
        let width = coff * headDim
        let rows = coff * ratio
        var seed: UInt64 = 0x9e37 &+ UInt64(ratio)
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }
        var stateKv = [Float](repeating: 0, count: rows * width)
        var stateScore = [Float](repeating: 0, count: rows * width)
        for i in 0..<(rows * width) { stateKv[i] = rnd(); stateScore[i] = rnd() * 3 }

        let gpu = try rt.compressorPool(stateKv: stateKv, stateScore: stateScore, headDim: headDim, ratio: ratio)
        let ref = poolRef(stateKv: stateKv, stateScore: stateScore, headDim: headDim, ratio: ratio)
        var maxAbs: Float = 0
        for j in 0..<headDim { maxAbs = max(maxAbs, abs(gpu[j] - ref[j])) }
        XCTAssertLessThan(maxAbs, 2e-4, "compressor pool ratio=\(ratio) diverges (maxAbs=\(maxAbs))")
    }

    func testCompressorPoolRatio128() throws { try run(ratio: 128, headDim: 512) }
    func testCompressorPoolRatio4() throws { try run(ratio: 4, headDim: 512) }
    func testCompressorPoolIndexerRatio4() throws { try run(ratio: 4, headDim: 128) }

    // ---- Full recurrent update (store + APE -> pool -> rmsnorm) over N tokens ----
    // Validates the GPU chain (compressorStoreOne + compressorPool + rmsNorm) with
    // recurrent state + ratio-4 shift + emit timing, vs an independent CPU oracle.
    // rope(comp_pos) + fp8 are single-row post-ops on already-validated kernels and
    // are validated end-to-end vs C at wiring time.

    private func cpuRmsNormWeighted(_ x: [Float], _ w: [Float], eps: Float) -> [Float] {
        var ss: Float = 0; for v in x { ss += v * v }
        let r = 1.0 / (ss / Float(x.count) + eps).squareRoot()
        return (0..<x.count).map { x[$0] * r * w[$0] }
    }
    // ratio-4 state shift: prev lane <- cur lane, then cur lane <- (old cur).
    private func shiftRatio4(_ s: inout [Float], ratio: Int, width: Int) {
        for r in 0..<ratio { for j in 0..<width { s[r * width + j] = s[(ratio + r) * width + j] } }
        for r in 0..<ratio { for j in 0..<width { s[(ratio + r) * width + j] = s[r * width + j] } }
    }

    private func runUpdate(ratio: Int, headDim: Int, nTokens: Int) throws {
        let rt = try makeRuntime()
        let coff = ratio == 4 ? 2 : 1
        let width = coff * headDim
        let rows = coff * ratio
        let eps: Float = 1e-6
        var seed: UInt64 = 0xC0FFEE &+ UInt64(ratio &* 131 &+ headDim)
        func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30) }

        let ape = (0..<(ratio * width)).map { _ in rnd() * 0.5 }
        let normW = (0..<headDim).map { _ in 0.5 + abs(rnd()) }

        let NEG: Float = -1e30
        var gKv = [Float](repeating: 0, count: rows * width), gSc = [Float](repeating: NEG, count: rows * width)
        var cKv = [Float](repeating: 0, count: rows * width), cSc = [Float](repeating: NEG, count: rows * width)
        var maxAbs: Float = 0; var emits = 0

        for pos in 0..<nTokens {
            let kvCur = (0..<width).map { _ in rnd() }
            let scCur = (0..<width).map { _ in rnd() * 2 }
            // GPU store.
            (gKv, gSc) = try rt.compressorStoreOne(kv: kvCur, score: scCur, ape: ape,
                                                   stateKv: gKv, stateScore: gSc, width: width, ratio: ratio, pos: pos)
            // CPU store (faithful to kernel_dsv4_compressor_store_one).
            let posMod = pos % ratio
            let dstRow = ratio == 4 ? ratio + posMod : posMod
            for j in 0..<width {
                cKv[dstRow * width + j] = kvCur[j]
                cSc[dstRow * width + j] = scCur[j] + ape[posMod * width + j]
            }
            guard (pos + 1) % ratio == 0 else { continue }
            emits += 1
            let gPool = try rt.compressorPool(stateKv: gKv, stateScore: gSc, headDim: headDim, ratio: ratio)
            let gOut = try rt.rmsNorm(gPool, rows: 1, n: headDim, eps: eps, weight: normW)
            let cPool = poolRef(stateKv: cKv, stateScore: cSc, headDim: headDim, ratio: ratio)
            let cOut = cpuRmsNormWeighted(cPool, normW, eps: eps)
            for j in 0..<headDim { maxAbs = max(maxAbs, abs(gOut[j] - cOut[j])) }
            if ratio == 4 { shiftRatio4(&gKv, ratio: ratio, width: width); shiftRatio4(&cKv, ratio: ratio, width: width)
                            shiftRatio4(&gSc, ratio: ratio, width: width); shiftRatio4(&cSc, ratio: ratio, width: width) }
        }
        XCTAssertGreaterThan(emits, 0, "no emit occurred")
        XCTAssertLessThan(maxAbs, 5e-4, "compressor update ratio=\(ratio) diverges over \(nTokens) tokens (maxAbs=\(maxAbs), emits=\(emits))")
    }

    func testCompressorUpdateRatio4() throws { try runUpdate(ratio: 4, headDim: 512, nTokens: 13) }
    func testCompressorUpdateRatio128() throws { try runUpdate(ratio: 128, headDim: 512, nTokens: 130) }
}
