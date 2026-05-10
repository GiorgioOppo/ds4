import Foundation
import Metal

/// RoPE with YaRN precomputed cos/sin tables. Applies rotation to the last
/// `ropeHeadDim` of each head, leaving the leading `nopeHeadDim` unchanged —
/// matches `apply_rotary_emb(q[..., -rd:], freqs_cis)` in the reference.
public final class RoPE {
    public let ropeHeadDim: Int
    public let freqs: Tensor                    // [seqlen, ropeHeadDim/2, 2] f32
    private let pipeline: MTLComputePipelineState

    public init(ropeHeadDim: Int, freqs: Tensor) {
        self.ropeHeadDim = ropeHeadDim
        self.freqs = freqs
        self.pipeline = Device.shared.makePipeline("rope_apply_f32")
    }

    public static func makeFreqs(config: ModelConfig, useYarn: Bool) -> Tensor {
        let originalSeqLen = useYarn ? config.originalSeqLen : 0
        let theta = useYarn ? config.compressRopeTheta : config.ropeTheta
        let arr = YaRN.precomputeFreqsCis(dim: config.ropeHeadDim,
                                          seqlen: config.maxSeqLen,
                                          originalSeqLen: originalSeqLen,
                                          base: theta,
                                          factor: config.ropeFactor,
                                          betaFast: config.betaFast,
                                          betaSlow: config.betaSlow)
        return arr.withUnsafeBytes { raw in
            Tensor.from(bytes: raw,
                        shape: [config.maxSeqLen, config.ropeHeadDim / 2, 2],
                        dtype: .f32)
        }
    }

    /// In-place rotation. `x`: [tokens, heads, headDim] f32.
    public func apply(_ x: Tensor, startPos: Int, inverse: Bool, in cmd: MTLCommandBuffer) {
        precondition(x.dtype == .f32 && x.shape.count == 3)
        let T = x.shape[0], H = x.shape[1], D = x.shape[2]
        precondition(D >= ropeHeadDim, "head_dim < rope_head_dim")

        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(freqs.buffer, offset: freqs.offset, index: 1)
        var dims = SIMD4<UInt32>(UInt32(T), UInt32(H), UInt32(D), UInt32(ropeHeadDim))
        enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 2)
        var misc = SIMD2<UInt32>(UInt32(startPos), inverse ? 1 : 0)
        enc.setBytes(&misc, length: MemoryLayout.size(ofValue: misc), index: 3)
        let half = ropeHeadDim / 2
        let grid = MTLSize(width: T, height: H, depth: half)
        let tg = MTLSize(width: min(T, 4), height: min(H, 4), depth: min(half, 16))
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}
