import Foundation
import Metal

/// Indexer: top-k learned KV-position selector for sparse attention.
/// Mirrors `Indexer` in `Original/DeepSeek-V4-Pro/inference/model.py` lines
/// 380–433. Used by layers whose `compress_ratio == 4` to pick which
/// compressed positions the main attention should attend to.
///
/// Pipeline (decode):
///   1. q  = wq_b(qr)                  — [b, s, h_idx, d_idx]
///   2. apply RoPE on the rope tail
///   3. Hadamard rotate(q), then FP4-quant in-place (QAT noise)
///   4. compressor(x) produces compressed KV (also FP4 + Hadamard)
///   5. weights = weights_proj(x) * (softmax_scale * h_idx^-0.5)
///   6. score   = einsum("bshd,btd->bsht", q, kv) → relu → weighted sum over heads
///   7. topk_idxs = score.topk(index_topk)
///
/// All numerically routed through FP4 — quality matters because these indices
/// drive the sparse attention.
public final class Indexer {
    public let dim: Int
    public let nHeads: Int
    public let headDim: Int
    public let ropeHeadDim: Int
    public let qLoraRank: Int
    public let indexTopk: Int
    public let compressRatio: Int
    public let softmaxScale: Float

    public let wqB: Linear              // [n_heads * head_dim, q_lora_rank]
    public let weightsProj: Linear      // [n_heads, dim] (BF16 in checkpoint)
    public let compressor: Compressor   // its own compressor with rotate=true

    public var kvCache: Tensor          // [maxBatch, maxSeqLen/ratio, head_dim]
    public var rope: RoPE?

    public init(config: ModelConfig, compressRatio: Int,
                wqB: Linear, weightsProj: Linear, compressor: Compressor,
                kvCache: Tensor) {
        self.dim = config.dim
        self.nHeads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.ropeHeadDim = config.ropeHeadDim
        self.qLoraRank = config.qLoraRank
        self.indexTopk = config.indexTopk
        self.compressRatio = compressRatio
        self.softmaxScale = pow(Float(config.indexHeadDim), -0.5)
        self.wqB = wqB
        self.weightsProj = weightsProj
        self.compressor = compressor
        self.kvCache = kvCache
    }

    /// Returns the int32 topk indices tensor of shape [b, s, indexTopk].
    /// NOT IMPLEMENTED. See model.py:402 for the reference pipeline.
    public func callAsFunction(_ x: Tensor, qr: Tensor, startPos: Int, offset: Int,
                               in cmd: MTLCommandBuffer) -> Tensor {
        fatalError("Indexer.forward not implemented — porting target: model.py:402")
    }
}
