import Foundation
import Metal

/// Multi-head Latent Attention (MLA) with sliding window + per-layer KV
/// compression. Direct port of the `Attention` module in
/// `Reference/inference/model.py` lines 436–543.
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
    /// Optional so `releaseCache()` can drop the underlying `MTLBuffer` and
    /// return its pages to the system; the next forward re-allocates lazily.
    public private(set) var kvCache: Tensor?

    private let kvCacheShape: [Int]
    private let kvCacheDType: DType

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
        self.kvCacheShape = kvCache.shape
        self.kvCacheDType = kvCache.dtype
    }

    private func ensureKVCache() -> Tensor {
        if let c = kvCache { return c }
        let t = Tensor.empty(shape: kvCacheShape, dtype: kvCacheDType)
        kvCache = t
        return t
    }

    /// Drop the KV cache buffer (and the compressor's alias to it). ARC
    /// frees the underlying `MTLBuffer`, returning unified-memory pages to
    /// the system. Safe to call between prompts; the next `callAsFunction`
    /// will re-allocate a fresh zero-initialized buffer.
    public func releaseCache() {
        kvCache = nil
        compressor?.kvCache = nil
    }

    /// MLA forward. Handles all three cases:
    ///   - prefill startPos == 0, seqlen <= window_size: write whole kv
    ///   - prefill startPos == 0, seqlen > window_size: write last window
    ///     rows with cutoff/wrap (model.py:521-523)
    ///   - decode startPos > 0, seqlen == 1: ring-buffer single-row write
    ///
    /// act_quant of the non-rope KV dims is skipped (QAT noise; structural
    /// forward correctness doesn't depend on it). Tier 3 will add it back.
    public func callAsFunction(_ x: Tensor, startPos: Int, in cmd: MTLCommandBuffer) -> Tensor {
        precondition(x.dtype == .f32 && x.shape.count == 3)
        let B = x.shape[0], S = x.shape[1]
        let isDecode = startPos > 0
        if isDecode {
            precondition(S == 1, "MLA decode expects seqlen == 1")
        }

        let kvCache = ensureKVCache()

        // Wire compressor's KV cache slice on first call (model.py:490-494).
        // After releaseCache() comp.kvCache is also nil, so this re-aliases
        // against the freshly-allocated kvCache.
        if let comp = compressor, comp.kvCache == nil {
            // The compressor writes into the trailing slice of kv_cache.
            // We model this by sharing the buffer via a Tensor with offset.
            let win = windowSize
            let bytesPerRow = headDim * MemoryLayout<Float>.size
            let cacheRowsPerBatch = kvCache.shape[1]
            let compRows = cacheRowsPerBatch - win
            comp.kvCache = Tensor(shape: [B, compRows, headDim], dtype: .f32,
                                  buffer: kvCache.buffer,
                                  offset: kvCache.offset + win * bytesPerRow)
            comp.rope = rope
            indexer?.rope = rope
        }

        // ---------- Q path ----------
        // qr = q_norm(wq_a(x))
        let xFlat = x.reshape([B * S, x.shape[2]])
        let qrFlat = qNorm(wqA(xFlat, in: cmd), in: cmd)            // [B*S, q_lora_rank]
        let qr = qrFlat.reshape([B, S, qLoraRank])

        // q = wq_b(qr).unflatten(-1, (n_heads, head_dim))
        var q = wqB(qrFlat, in: cmd)                                 // [B*S, n_heads*head_dim]
        q = q.reshape([B * S, nHeads, headDim])

        // q *= rsqrt(mean(q^2) + eps) over head_dim (re-norm per head)
        let rsqrtP = Device.shared.makePipeline("rsqrt_mean_square_f32")
        let bcastP = Device.shared.makePipeline("broadcast_row_mul_f32")
        let qFlat = q.reshape([B * S * nHeads, headDim])
        let qRsqrt = Tensor.empty(shape: [B * S * nHeads], dtype: .f32)
        do {
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(rsqrtP)
            enc.setBuffer(qFlat.buffer, offset: qFlat.offset, index: 0)
            enc.setBuffer(qRsqrt.buffer, offset: 0, index: 1)
            var d = UInt32(headDim); var e = eps
            enc.setBytes(&d, length: 4, index: 2)
            enc.setBytes(&e, length: 4, index: 3)
            enc.setThreadgroupMemoryLength(256 * MemoryLayout<Float>.size, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: B * S * nHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        do {
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(bcastP)
            enc.setBuffer(qFlat.buffer, offset: qFlat.offset, index: 0)
            enc.setBuffer(qRsqrt.buffer, offset: 0, index: 1)
            var dims = SIMD2<UInt32>(UInt32(B * S * nHeads), UInt32(headDim))
            enc.setBytes(&dims, length: MemoryLayout.size(ofValue: dims), index: 2)
            enc.dispatchThreads(MTLSize(width: headDim, height: B * S * nHeads, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
        }

        // RoPE on rope tail of q (treated as [B*S, n_heads, head_dim]).
        rope.apply(q.reshape([B * S, nHeads, headDim]),
                   startPos: startPos, inverse: false, in: cmd)

        // ---------- KV path ----------
        // kv = kv_norm(wkv(x))  → [B, S, head_dim]
        let kvFlat = kvNorm(wkv(xFlat, in: cmd), in: cmd)
        // RoPE on rope tail (interpret as [B*S, 1, head_dim]).
        rope.apply(kvFlat.reshape([B * S, 1, headDim]),
                   startPos: startPos, inverse: false, in: cmd)
        let kv = kvFlat.reshape([B, S, headDim])

        // ---------- Topk indices ----------
        // Window indices use the actual startPos so the ring-buffer wrap
        // is handled correctly during decode. For prefill startPos == 0
        // and AttentionIndices.slidingWindow returns the prefill table.
        let winIdxs = AttentionIndices.slidingWindow(windowSize: windowSize,
                                                       batch: B, seqlen: S, startPos: startPos)
        var topkArr = winIdxs
        var K = windowSize

        // Compressed offset: prefill compressed tokens are appended to `kv`
        // at index `kv.size(1)` == S. Decode reads from kvCache where the
        // compressed slice starts at `windowSize` (after the window ring).
        let compOffset = isDecode ? windowSize : S

        var compK = 0
        if compressRatio > 0 {
            if let idx = indexer {
                // Indexer returns [B, S, K_idx] i32 on GPU. Read back to
                // host so we can concat with winIdxs uniformly.
                let topkT = idx(x, qr: qr, startPos: startPos, offset: compOffset, in: cmd)
                cmd.commit(); cmd.waitUntilCompleted()
                compK = topkT.shape[2]
                let p = topkT.buffer.contents().bindMemory(to: Int32.self,
                                                            capacity: B * S * compK)
                let compIdxs = Array(UnsafeBufferPointer(start: p, count: B * S * compK))
                topkArr = mergeTopk(window: winIdxs, compress: compIdxs,
                                     B: B, S: S, kWin: K, kComp: compK)
                K += compK
            } else {
                let (compIdxs, kc) = AttentionIndices.compressed(ratio: compressRatio,
                                                                  batch: B, seqlen: S,
                                                                  startPos: startPos,
                                                                  offset: compOffset)
                compK = kc
                topkArr = mergeTopk(window: winIdxs, compress: compIdxs,
                                     B: B, S: S, kWin: K, kComp: compK)
                K += kc
            }
        }
        let topkT = topkArr.withUnsafeBytes {
            Tensor.from(bytes: $0, shape: [B, S, K], dtype: .i32)
        }

        // ---------- KV cache write ----------
        let bytesPerRow = headDim * MemoryLayout<Float>.size
        let cacheRows = kvCache.shape[1]
        do {
            let blit = cmd.makeBlitCommandEncoder()!
            if isDecode {
                // Single-row ring-buffer write: kv_cache[:B, startPos % win] = kv[:, 0]
                let slot = startPos % windowSize
                for b in 0..<B {
                    let srcOff = b * bytesPerRow
                    let dstOff = kvCache.offset + (b * cacheRows + slot) * bytesPerRow
                    blit.copy(from: kv.buffer, sourceOffset: srcOff,
                              to: kvCache.buffer, destinationOffset: dstOff,
                              size: bytesPerRow)
                }
            } else if S <= windowSize {
                // Full S rows, contiguous, starting at slot 0.
                for b in 0..<B {
                    let srcOff = b * S * bytesPerRow
                    let dstOff = kvCache.offset + b * cacheRows * bytesPerRow
                    blit.copy(from: kv.buffer, sourceOffset: srcOff,
                              to: kvCache.buffer, destinationOffset: dstOff,
                              size: S * bytesPerRow)
                }
            } else {
                // Prefill seqlen > window: keep only the last `windowSize` kv
                // rows, with cutoff/wrap so the ring buffer ends at
                // (S - 1) % win. Mirrors model.py:521-523.
                let cutoff = S % windowSize
                let firstHalf = windowSize - cutoff   // rows that go into [cutoff, windowSize)
                let kvLastWinStart = S - windowSize
                for b in 0..<B {
                    // First chunk: kv[:, S-win .. S-cutoff] → kvCache[:, cutoff..win]
                    if firstHalf > 0 {
                        let srcOff = (b * S + kvLastWinStart) * bytesPerRow
                        let dstOff = kvCache.offset
                            + (b * cacheRows + cutoff) * bytesPerRow
                        blit.copy(from: kv.buffer, sourceOffset: srcOff,
                                  to: kvCache.buffer, destinationOffset: dstOff,
                                  size: firstHalf * bytesPerRow)
                    }
                    // Second chunk: kv[:, S-cutoff .. S] → kvCache[:, 0..cutoff]
                    if cutoff > 0 {
                        let srcOff = (b * S + (S - cutoff)) * bytesPerRow
                        let dstOff = kvCache.offset + b * cacheRows * bytesPerRow
                        blit.copy(from: kv.buffer, sourceOffset: srcOff,
                                  to: kvCache.buffer, destinationOffset: dstOff,
                                  size: cutoff * bytesPerRow)
                    }
                }
            }
            blit.endEncoding()
        }

        // ---------- Build the kv tensor passed to sparse_attn ----------
        // Prefill: kvFull = kv ++ compressor_output (along S axis), so the
        //   topk indices [0, S) map into kv and [S, S+compS) into the comp.
        // Decode: kvFull = kvCache[:B] (the whole [windowSize + compRows]
        //   buffer the compressor itself writes into).
        let kvFull: Tensor
        if isDecode {
            // Run compressor for side-effect: it updates state and may write
            // a new compressed token into the trailing slice of kvCache.
            if let comp = compressor {
                _ = comp(x, startPos: startPos, in: cmd)
            }
            kvFull = Tensor(shape: [B, cacheRows, headDim], dtype: .f32,
                             buffer: kvCache.buffer, offset: kvCache.offset)
        } else if let comp = compressor, let cOut = comp(x, startPos: startPos, in: cmd) {
            let compS = cOut.shape[1]
            let total = S + compS
            kvFull = Tensor.empty(shape: [B, total, headDim], dtype: .f32)
            let blit = cmd.makeBlitCommandEncoder()!
            for b in 0..<B {
                let srcKvOff = b * S * bytesPerRow
                let dstKvOff = b * total * bytesPerRow
                blit.copy(from: kv.buffer, sourceOffset: srcKvOff,
                          to: kvFull.buffer, destinationOffset: dstKvOff,
                          size: S * bytesPerRow)
                let srcCOff = b * compS * bytesPerRow
                let dstCOff = (b * total + S) * bytesPerRow
                blit.copy(from: cOut.buffer, sourceOffset: srcCOff,
                          to: kvFull.buffer, destinationOffset: dstCOff,
                          size: compS * bytesPerRow)
            }
            blit.endEncoding()
        } else {
            kvFull = kv
        }

        // ---------- Sparse attention ----------
        let qPerToken = q.reshape([B, S, nHeads, headDim])
        let o = SparseAttention.apply(q: qPerToken, kv: kvFull, sink: attnSink,
                                       topkIdxs: topkT, scale: softmaxScale, in: cmd)

        // ---------- Inverse RoPE ----------
        rope.apply(o.reshape([B * S, nHeads, headDim]),
                   startPos: startPos, inverse: true, in: cmd)

        // ---------- Grouped output: einsum("bsgd,grd->bsgr"), then wo_b ----------
        let perGroupD = nHeads * headDim / nGroups
        let oView = o.reshape([B, S, nGroups, perGroupD])
        let woAR = woA.weight.reshape([nGroups, oLoraRank, perGroupD])
        let oR = Einsum.bsgdGrd(o: oView, woA: woAR, in: cmd)        // [B, S, nGroups, oLoraRank]
        let oFlat = oR.reshape([B * S, nGroups * oLoraRank])
        return woB(oFlat, in: cmd).reshape([B, S, dim])
    }

    private var dim: Int { wkv.inFeatures }

    private func mergeTopk(window: [Int32], compress: [Int32],
                           B: Int, S: Int, kWin: Int, kComp: Int) -> [Int32] {
        let total = kWin + kComp
        var out = [Int32](repeating: 0, count: B * S * total)
        for b in 0..<B {
            for s in 0..<S {
                for k in 0..<kWin {
                    out[(b * S + s) * total + k] = window[(b * S + s) * kWin + k]
                }
                for k in 0..<kComp {
                    out[(b * S + s) * total + kWin + k] = compress[(b * S + s) * kComp + k]
                }
            }
        }
        return out
    }
}
