import XCTest
import Metal
@testable import DeepSeekKit

final class CompressorTests: XCTestCase {

    /// Prefill, no overlap (ratio = 128 path in V4). Compares Metal forward
    /// against the pure-Swift reference. Tests skip the FP4/FP8 quant step
    /// because the CPU reference also skips it (it would only add precision
    /// noise without changing the structural match).
    func testPrefillNoOverlapMatchesCPU() throws {
        let B = 1
        let dim = 8
        let headDim = 8
        let ropeHeadDim = 4
        let ratio = 4
        let S = 8                  // 2 compressed tokens
        let normEps: Float = 1e-6

        // Build a config sized for the test.
        var cfg = ModelConfig()
        cfg.dim = dim
        cfg.headDim = headDim
        cfg.ropeHeadDim = ropeHeadDim
        cfg.normEps = normEps
        cfg.maxSeqLen = 64
        cfg.maxBatchSize = B

        // Random inputs.
        let x = randomArray(B * S * dim, seed: 1, scale: 0.5)
        let wkv = randomArray(headDim * dim, seed: 2, scale: 0.2)        // coff=1 for ratio != 4
        let wgate = randomArray(headDim * dim, seed: 3, scale: 0.2)
        let ape = randomArray(ratio * headDim, seed: 4, scale: 0.1)
        let normW = (0..<headDim).map { _ in Float(1.0) }                 // identity-ish norm

        // RoPE freqs (no YaRN: originalSeqLen = 0).
        let freqs = YaRN.precomputeFreqsCis(dim: ropeHeadDim, seqlen: cfg.maxSeqLen,
                                             originalSeqLen: 0, base: cfg.ropeTheta,
                                             factor: cfg.ropeFactor,
                                             betaFast: cfg.betaFast, betaSlow: cfg.betaSlow)

        // ---- GPU side ----
        let xT = x.withUnsafeBytes { Tensor.from(bytes: $0, shape: [B, S, dim], dtype: .f32) }
        let wkvT = wkv.withUnsafeBytes { Tensor.from(bytes: $0, shape: [headDim, dim], dtype: .f32) }
        let wgateT = wgate.withUnsafeBytes { Tensor.from(bytes: $0, shape: [headDim, dim], dtype: .f32) }
        let apeT = ape.withUnsafeBytes { Tensor.from(bytes: $0, shape: [ratio, headDim], dtype: .f32) }
        let normWT = normW.withUnsafeBytes { Tensor.from(bytes: $0, shape: [headDim], dtype: .f32) }
        let freqsT = freqs.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [cfg.maxSeqLen, ropeHeadDim / 2, 2], dtype: .f32)
        }

        let kvState = Tensor.empty(shape: [B, ratio, headDim], dtype: .f32)
        let scoreState = Tensor.empty(shape: [B, ratio, headDim], dtype: .f32)
        let wkvLin = Linear(inFeatures: dim, outFeatures: headDim, weight: wkvT, scale: nil)
        let wgateLin = Linear(inFeatures: dim, outFeatures: headDim, weight: wgateT, scale: nil)
        let normRMS = RMSNorm(weight: normWT, eps: normEps)
        let comp = Compressor(config: cfg, compressRatio: ratio, headDim: headDim, rotate: false,
                              ape: apeT, wkv: wkvLin, wgate: wgateLin, norm: normRMS,
                              kvState: kvState, scoreState: scoreState)
        comp.rope = RoPE(ropeHeadDim: ropeHeadDim, freqs: freqsT)

        let cmd = Device.shared.queue.makeCommandBuffer()!
        guard let out = comp(xT, startPos: 0, in: cmd) else {
            XCTFail("expected non-nil compressed output"); return
        }
        cmd.commit(); cmd.waitUntilCompleted()
        let gpu = out.toFloatArray()

        // ---- CPU reference ----
        let cpu = Compressor.referenceCPU(
            x: x, wkv: wkv, wgate: wgate, ape: ape, normWeight: normW, normEps: normEps,
            ropeFreqs: freqs, B: B, S: S, dim: dim, headDim: headDim,
            ropeHeadDim: ropeHeadDim, ratio: ratio, overlap: false)

        XCTAssertEqual(gpu.count, cpu.count)
        for i in 0..<gpu.count {
            XCTAssertEqual(gpu[i], cpu[i], accuracy: 1e-3, "i=\(i)")
        }
        _ = x; _ = wkv; _ = wgate; _ = ape; _ = normW; _ = freqs
    }

    private func randomArray(_ count: Int, seed: UInt64, scale: Float = 1) -> [Float] {
        var state = seed | 1
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let frac = Float(Double(state >> 11) / Double(1 << 53))
            out[i] = (frac - 0.5) * 2 * scale
        }
        return out
    }
}
