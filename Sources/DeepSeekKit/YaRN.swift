import Foundation

/// YaRN frequency scaling for long-context RoPE. Mirrors `precompute_freqs_cis`
/// from `Original/DeepSeek-V4-Pro/inference/model.py` lines 199–229.
///
/// Returns a flat array of length `2 * seqlen * (dim/2)` interpreted as
/// `[seqlen][dim/2]` of (cos, sin) pairs. This is what the Metal kernel
/// expects so it can do a row × cos/sin lookup without recomputing
/// trigonometry per token.
public enum YaRN {
    public static func precomputeFreqsCis(dim: Int,
                                          seqlen: Int,
                                          originalSeqLen: Int,
                                          base: Float,
                                          factor: Float,
                                          betaFast: Int,
                                          betaSlow: Int) -> [Float] {
        let half = dim / 2
        var freqs = [Float](repeating: 0, count: half)
        for i in 0..<half {
            freqs[i] = 1.0 / pow(base, Float(2 * i) / Float(dim))
        }
        if originalSeqLen > 0 {
            let (low, high) = correctionRange(lowRot: Float(betaFast),
                                              highRot: Float(betaSlow),
                                              dim: dim, base: base,
                                              maxSeqLen: originalSeqLen)
            for i in 0..<half {
                let smooth = 1 - rampFactor(min: low, max: high, idx: i)
                freqs[i] = freqs[i] / factor * (1 - smooth) + freqs[i] * smooth
            }
        }
        var out = [Float](repeating: 0, count: 2 * seqlen * half)
        for t in 0..<seqlen {
            for i in 0..<half {
                let angle = Float(t) * freqs[i]
                out[2 * (t * half + i) + 0] = cos(angle)
                out[2 * (t * half + i) + 1] = sin(angle)
            }
        }
        return out
    }

    private static func correctionDim(numRotations: Float, dim: Int, base: Float, maxSeqLen: Int) -> Float {
        return Float(dim) * log(Float(maxSeqLen) / (numRotations * 2 * .pi)) / (2 * log(base))
    }

    private static func correctionRange(lowRot: Float, highRot: Float, dim: Int, base: Float, maxSeqLen: Int) -> (Float, Float) {
        let lo = floor(correctionDim(numRotations: lowRot, dim: dim, base: base, maxSeqLen: maxSeqLen))
        let hi = ceil(correctionDim(numRotations: highRot, dim: dim, base: base, maxSeqLen: maxSeqLen))
        return (max(lo, 0), min(hi, Float(dim - 1)))
    }

    private static func rampFactor(min lo: Float, max hi: Float, idx: Int) -> Float {
        var hi2 = hi
        if lo == hi2 { hi2 += 0.001 }
        let v = (Float(idx) - lo) / (hi2 - lo)
        return max(0, min(1, v))
    }
}
