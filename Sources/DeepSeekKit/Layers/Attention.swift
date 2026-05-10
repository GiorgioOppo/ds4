import Foundation
import Metal

/// Multi-head Latent Attention (MLA) with sliding window + per-layer KV
/// compression. Direct port of the `Attention` module in
/// `Original/DeepSeek-V4-Pro/inference/model.py` lines 436–543.
///
/// Layer-mode is selected by `compressRatio`:
///   - 0:   pure sliding-window attention (window_size = 128), no compression
///   - 4:   sliding window + Compressor (overlap) + Indexer (sparse top-k)
///   - 128: sliding window + Compressor (no overlap), no Indexer
///
/// Q path: low-rank wq_a (dim → q_lora_rank) → q_norm → wq_b → rsqrt(var)
/// K/V path: shared wkv (dim → head_dim), kv_norm
/// O path: grouped low-rank wo_a (head_dim → o_lora_rank, per group) → wo_b
public final class MLA {
    public let layerId: Int
    public let nHeads: Int
    public let headDim: Int
    public let ropeHeadDim: Int
    public let nopeHeadDim: Int
    public let nGroups: Int
    public let oLoraRank: Int
    public let qLoraRank: Int
    public let windowSize: Int
    public let compressRatio: Int
    public let eps: Float
    public let softmaxScale: Float

    public let wqA: Linear
    public let qNorm: RMSNorm
    public let wqB: Linear
    public let wkv: Linear
    public let kvNorm: RMSNorm
    public let woA: Linear
    public let woB: Linear
    public let attnSink: Tensor       // [n_heads] f32

    public let rope: RoPE
    public let compressor: Compressor?
    public let indexer: Indexer?

    /// Sliding window KV ring buffer + compressed KV tail.
    /// Layout: kv_cache[:, :window] = window ring; kv_cache[:, window:] = compressed.
    public var kvCache: Tensor

    public init(config: ModelConfig, layerId: Int,
                wqA: Linear, qNorm: RMSNorm, wqB: Linear,
                wkv: Linear, kvNorm: RMSNorm,
                woA: Linear, woB: Linear,
                attnSink: Tensor,
                rope: RoPE,
                compressor: Compressor?, indexer: Indexer?,
                kvCache: Tensor) {
        self.layerId = layerId
        self.nHeads = config.nHeads
        self.headDim = config.headDim
        self.ropeHeadDim = config.ropeHeadDim
        self.nopeHeadDim = config.headDim - config.ropeHeadDim
        self.nGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.qLoraRank = config.qLoraRank
        self.windowSize = config.windowSize
        self.compressRatio = config.compressRatios[layerId]
        self.eps = config.normEps
        self.softmaxScale = pow(Float(config.headDim), -0.5)
        self.wqA = wqA; self.qNorm = qNorm; self.wqB = wqB
        self.wkv = wkv; self.kvNorm = kvNorm
        self.woA = woA; self.woB = woB
        self.attnSink = attnSink
        self.rope = rope
        self.compressor = compressor
        self.indexer = indexer
        self.kvCache = kvCache
    }

    /// One forward pass — prefill if startPos == 0, decode otherwise.
    ///
    /// NOT IMPLEMENTED. The reference at model.py:484 weaves together:
    ///   - Q low-rank with intermediate RMSNorm and rsqrt(var) re-norm
    ///   - act_quant on the non-rope dims of KV (FP8 QAT)
    ///   - sliding-window topk-idx generation (`get_window_topk_idxs`)
    ///   - compressed-tokens topk-idx (`get_compress_topk_idxs` or Indexer)
    ///   - sparse_attn(q, kv, attn_sink, topk_idxs, scale)
    ///   - inverse RoPE on the rope dims of the output
    ///   - grouped low-rank output via einsum("bsgd,grd->bsgr", o, wo_a)
    ///   - wo_b
    public func callAsFunction(_ x: Tensor, startPos: Int, in cmd: MTLCommandBuffer) -> Tensor {
        fatalError("MLA.forward not implemented — porting target: model.py:484")
    }
}
