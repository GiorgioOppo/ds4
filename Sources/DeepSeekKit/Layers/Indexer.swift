import Foundation
import Metal

/// Indexer: top-k learned KV-position selector for sparse attention.
/// Mirrors `Indexer` in `Reference/inference/model.py` lines 380–433.
///
/// Used by layers whose `compress_ratio == 4` to pick which compressed KV
/// positions the main attention should attend to. Implementation matches
/// the prefill path; the decode path reuses it with `seqlen == 1`.
public final class Indexer {
    public let dim: Int
    public let nHeads: Int
    public let headDim: Int
    public let ropeHeadDim: Int
    public let qLoraRank: Int
    public let indexTopk: Int
    public let compressRatio: Int
    public let softmaxScale: Float

    public let wqB: Linear
    public let weightsProj: Linear
    public let compressor: Compressor
    /// `[maxBatch, maxSeqLen/ratio, head_dim]`. Optional so `releaseCache()`
    /// can free the underlying `MTLBuffer`; lazy re-alloc on next forward.
    public private(set) var kvCache: Tensor?
    public var rope: RoPE?

    private let kvCacheShape: [Int]
    private let kvCacheDType: DType

    private let pScoreReduce: MTLComputePipelineState
    private let pTopKPostproc: MTLComputePipelineState

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
        self.kvCacheShape = kvCache.shape
        self.kvCacheDType = kvCache.dtype
        self.pScoreReduce = Device.shared.makePipeline("indexer_score_reduce_f32")
        self.pTopKPostproc = Device.shared.makePipeline("indexer_topk_postprocess_i32")
    }

    private func ensureKVCache() -> Tensor {
        if let c = kvCache { return c }
        let t = Tensor.empty(shape: kvCacheShape, dtype: kvCacheDType)
        kvCache = t
        return t
    }

    /// Drop the indexer's kvCache buffer. ARC frees the underlying
    /// `MTLBuffer`; the next forward will re-allocate.
    public func releaseCache() {
        kvCache = nil
        compressor.releaseState()
    }

    /// Re-allocate the indexer's KV cache from a snapshot blob.
    /// The indexer's `compressor.kvCache` is a *reference* to the
    /// same Tensor (Indexer.swift wires `compressor.kvCache = self.kvCache`
    /// on first forward), so we set both pointers here — the
    /// compressor's alias would otherwise stay pointing at the
    /// pre-restore buffer and the next forward would read stale
    /// data.
    public func restoreKVCacheBytes(shape: [Int], dtype: DType, bytes: Data) {
        precondition(shape == kvCacheShape,
                      "Indexer restore: shape mismatch (got \(shape), expected \(kvCacheShape))")
        precondition(dtype == kvCacheDType,
                      "Indexer restore: dtype mismatch")
        let t = Tensor.empty(shape: shape, dtype: dtype)
        t.writeBytes(bytes)
        self.kvCache = t
        self.compressor.kvCache = t
    }

    /// Returns `[B, S, K]` Int32 indices (with -1 padding for invalid slots).
    public func callAsFunction(_ x: Tensor, qr: Tensor, startPos: Int, offset: Int,
                                in cmd: MTLCommandBuffer) -> Tensor {
        precondition(x.dtype == .f32 && x.shape.count == 3)
        let B = x.shape[0], S = x.shape[1]
        let endPos = startPos + S
        let ratio = compressRatio
        let kvCache = ensureKVCache()

        // Wire our internal Compressor to share kv_cache + rope on first
        // call. Mirrors model.py:408-410 (`self.compressor.kv_cache = self.kv_cache;
        // self.compressor.freqs_cis = self.freqs_cis`). Without this the
        // compressor traps in its decode path at the first should_emit
        // boundary, because its `rope` is nil.
        if compressor.kvCache == nil {
            compressor.kvCache = kvCache
            compressor.rope = rope
        }

        // 1. q = wq_b(qr) → [B, S, n_heads, head_dim]
        let qFlat = wqB(qr.reshape([B * S, qLoraRank]), in: cmd)
        let q = qFlat.reshape([B, S, nHeads, headDim])

        // 2. RoPE on rope tail (interpret as [B*S, n_heads, head_dim]).
        guard let rope = rope else { fatalError("Indexer.rope not set") }
        rope.apply(q.reshape([B * S, nHeads, headDim]),
                   startPos: startPos, inverse: false, in: cmd)

        // 3. Hadamard rotation on the head dim (per-head independently).
        Hadamard.apply(q, in: cmd)

        // 4. FP4 quant in place (QAT noise).
        _ = ActQuant(format: .fp4).quant(q.reshape([B * S * nHeads, headDim]),
                                          inplace: true, in: cmd)

        // 5. Run the Compressor — it writes into self.kvCache (shared with
        //    us via the wiring above), so by the time this call returns
        //    the einsum below reads the up-to-date compressed positions.
        //    No explicit blit is needed any more.
        _ = compressor(x, startPos: startPos, in: cmd)

        // No compressed positions available yet — return an empty top-k.
        // MLA's caller path treats compK==0 as "use window indices only".
        let T = endPos / ratio
        if T == 0 {
            return Tensor.empty(shape: [B, S, 0], dtype: .i32)
        }

        // 6. weights = weights_proj(x) * (softmax_scale * n_heads^-0.5)
        let weightsFlat = weightsProj(x.reshape([B * S, dim]), in: cmd)
        let weightsScale = softmaxScale * pow(Float(nHeads), -0.5)
        let weights = Elementwise.scale(weightsFlat, by: weightsScale, in: cmd)

        // 7. score = einsum("bshd,btd->bsht", q, kv_cache[:B, :T]).
        // Slice kv_cache to first T rows by reshaping the same buffer.
        let kvSlice = Tensor(shape: [B, T, headDim], dtype: .f32,
                              buffer: kvCache.buffer, offset: kvCache.offset)
        let score = Einsum.bshdBtd(q: q, kv: kvSlice, in: cmd)   // [B, S, n_heads, T]

        // 8. y[B, S, T] = Σ_h relu(score) * weights, with prefill causal mask.
        let y = Tensor.empty(shape: [B, S, T], dtype: .f32)
        do {
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pScoreReduce)
            enc.setBuffer(score.buffer, offset: 0, index: 0)
            enc.setBuffer(weights.buffer, offset: 0, index: 1)
            enc.setBuffer(y.buffer, offset: 0, index: 2)
            var dims = SIMD4<UInt32>(UInt32(B), UInt32(S), UInt32(nHeads), UInt32(T))
            enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 3)
            var misc = SIMD2<UInt32>(UInt32(ratio), startPos == 0 ? 1 : 0)
            enc.setBytes(&misc, length: MemoryLayout.size(ofValue: misc), index: 4)
            let grid = MTLSize(width: T, height: S, depth: B)
            enc.dispatchThreads(grid,
                                threadsPerThreadgroup: pScoreReduce.tunedThreadgroup(forGrid: grid))
            enc.endEncoding()
        }

        // 9. Top-K along the last axis.
        let yFlat = y.reshape([B * S, T])
        let k = min(indexTopk, T)
        let topk = TopK.apply(yFlat, k: k, in: cmd)
        let topkIdxs = topk.indices.reshape([B, S, k])

        // 10. Mask invalid + add offset.
        do {
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pTopKPostproc)
            enc.setBuffer(topkIdxs.buffer, offset: 0, index: 0)
            var dims = SIMD3<UInt32>(UInt32(B), UInt32(S), UInt32(k))
            enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 1)
            var misc = SIMD3<UInt32>(UInt32(ratio), UInt32(offset), startPos == 0 ? 1 : 0)
            enc.setBytes(&misc, length: MemoryLayout.size(ofValue: misc), index: 2)
            let grid = MTLSize(width: k, height: S, depth: B)
            enc.dispatchThreads(grid,
                                threadsPerThreadgroup: pTopKPostproc.tunedThreadgroup(forGrid: grid))
            enc.endEncoding()
        }

        return topkIdxs
    }
}
