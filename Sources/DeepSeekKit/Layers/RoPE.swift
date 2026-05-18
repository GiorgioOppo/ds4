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

        // ---- Threadgroup sizing tuned per occupancy ----
        // Il vecchio piano `min(T,4)×min(H,4)×min(half,16) = 256`
        // capava il threadgroup a 256 thread e con la `z` non
        // multipla di 32 sprecava lane SIMD. Riconfiguriamo
        // leggendo i limiti hardware del pipeline (su Apple GPU
        // simdWidth = 32, maxTotal = 1024).
        //
        // Layout: il thread (gid.z) indirizza il pair index `i`, e
        // ogni thread accede a `x[base + 2*i]` / `x[base + 2*i+1]`.
        // Gli accessi sono contigui in memoria quando `z` varia →
        // mettiamo simdWidth thread nella dimensione `z` per
        // ottenere coalescenza.
        let simdWidth = pipeline.threadExecutionWidth
        let maxTotal = pipeline.maxTotalThreadsPerThreadgroup
        let tgZ = max(1, min(half, simdWidth))
        // Target ~256 thread per blocco prima di considerare `x`
        // (T-dim) per non strozzare la occupancy a regime memory-bound.
        let tgY = max(1, min(H, max(1, 256 / max(1, tgZ))))
        // Riempi con tgX fino al maxTotal del pipeline.
        let remaining = max(1, maxTotal / max(1, tgY * tgZ))
        let tgX = max(1, min(T, remaining))

        let grid = MTLSize(width: T, height: H, depth: half)
        let tg = MTLSize(width: tgX, height: tgY, depth: tgZ)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}
