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
        // V4 reference (`Reference/inference/model.py:395`):
        //   self.softmax_scale = self.head_dim ** -0.5
        // — plain inverse-sqrt with NO YaRN mscale correction (unlike
        // V3, which adds `mscale = 0.1 * f * log(rope_factor) + 1` and
        // multiplies by `mscale**2`). The earlier attempt to import
        // V3's formula here made attention slightly sharper than what
        // the V4 weights were trained for, and outputs got worse.
        // Keep it simple and match the reference.
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

    /// Drop the KV cache buffer (and all state of the attached compressor).
    /// ARC frees the underlying `MTLBuffer`s, returning unified-memory pages
    /// to the system. The next `callAsFunction` will re-allocate everything
    /// lazily and re-wire the compressor's alias against the fresh buffer.
    ///
    /// Must be called only between forward passes — not thread-safe.
    /// Use between unrelated prompts; if a decode resumes after a release
    /// it must start from `startPos == 0` (post-prefill or a fresh
    /// sequence) since the compressor's rolling `kvState`/`scoreState` is
    /// zeroed on realloc.
    public func releaseCache() {
        kvCache = nil
        // Drop the compressor's alias into the kv cache buffer *and* its
        // own rolling state so the whole MLA-side state is freed in one
        // call. Symmetrical with `Indexer.releaseCache`.
        compressor?.releaseState()
    }

    /// Rewind dello stato KV a una position arbitraria. Propaga al
    /// `Compressor` (che gestisce kvState/scoreState rolling) e
    /// LASCIA la `kvCache` principale invariata — i suoi entry
    /// alle position oltre `pos` saranno sovrascritti dal forward
    /// successivo (entry alle position 0..pos restano valide, sono
    /// il vero prefix preservato).
    ///
    /// `pos` deve essere multiplo del `compressRatio` (se il
    /// compressor è presente) per garantire rewind safe a window
    /// boundary. Per i layer puramente sliding-window (no
    /// compressor), qualsiasi `pos` è OK.
    ///
    /// - Returns: `true` se il rewind è riuscito (state coerente
    ///   con position `pos`), `false` se non possibile (caller
    ///   deve fare `releaseCache()` + cold-prefill).
    @discardableResult
    public func rewindKVTo(pos: Int) -> Bool {
        guard pos >= 0 else { return false }
        if let comp = compressor {
            return comp.rewindStateTo(pos: pos)
        }
        // No compressor → puramente sliding window. Niente state
        // rolling da resettare; il ring buffer si auto-overwrite
        // al prossimo forward.
        return true
    }

    /// Re-allocate the main KV cache from a snapshot blob. Used by
    /// the KV-cache restore path (Step B2 onward) so a previously-
    /// computed cache state can be swapped back into a freshly
    /// loaded model. The compressor's alias slice is intentionally
    /// left nil — `callAsFunction` re-builds it on the next forward
    /// (it checks `comp.kvCache == nil` and re-aliases against the
    /// freshly-set kvCache buffer).
    public func restoreKVCacheBytes(shape: [Int], dtype: DType, bytes: Data) {
        precondition(shape == kvCacheShape,
                      "MLA restore: shape mismatch (got \(shape), expected \(kvCacheShape))")
        precondition(dtype == kvCacheDType,
                      "MLA restore: dtype mismatch")
        let t = Tensor.empty(shape: shape, dtype: dtype)
        t.writeBytes(bytes)
        self.kvCache = t
    }

    /// MLA forward. Handles all three cases:
    ///   - prefill startPos == 0, seqlen <= window_size: write whole kv
    ///   - prefill startPos == 0, seqlen > window_size: write last window
    ///     rows with cutoff/wrap (model.py:521-523)
    ///   - decode startPos > 0, seqlen == 1: ring-buffer single-row write
    ///
    /// FP8 QAT noise is applied in-place to the non-rope dims of KV after
    /// RoPE, matching model.py:506 (`act_quant(kv[..., :-rd], 64, ..., True)`).
    /// Without this the residual stream amplifies uncontrolled past layer 5.
    ///
    /// `cmd` is `inout`: when MLA has to flush the queue to read indexer
    /// topk to host (compress_ratio == 4), the original command buffer is
    /// committed and replaced with a fresh one. Caller's subsequent work
    /// must go onto the swapped buffer.
    public func callAsFunction(_ x: Tensor, startPos: Int, in cmd: inout MTLCommandBuffer) -> Tensor {
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
        // Drain the pipe and trace at layer 0 only — we need the layer
        // boundary intermediates resolved on host to localize the first
        // operation that introduces NaN/Inf. Streaming inference also
        // commits per-layer, so the extra sync only fires under the
        // diagnostic flag.
        if TraceFlags.normTrace && (layerId == 0 || layerId == 5 || layerId == 6) {
            cmd.commit(); cmd.waitUntilCompleted()
            traceTensorStats("mla[\(layerId)] after wqA+qNorm", qrFlat)
            cmd = Device.shared.queue.makeCommandBuffer()!
        }
        let qr = qrFlat.reshape([B, S, qLoraRank])

        // q = wq_b(qr).unflatten(-1, (n_heads, head_dim))
        var q = wqB(qrFlat, in: cmd)                                 // [B*S, n_heads*head_dim]
        if TraceFlags.normTrace && (layerId == 0 || layerId == 5 || layerId == 6) {
            cmd.commit(); cmd.waitUntilCompleted()
            traceTensorStats("mla[\(layerId)] after wqB", q)
            cmd = Device.shared.queue.makeCommandBuffer()!
        }
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
            let grid = MTLSize(width: headDim, height: B * S * nHeads, depth: 1)
            enc.dispatchThreads(grid,
                                threadsPerThreadgroup: bcastP.tunedThreadgroup(forGrid: grid))
            enc.endEncoding()
        }

        // RoPE on rope tail of q (treated as [B*S, n_heads, head_dim]).
        rope.apply(q.reshape([B * S, nHeads, headDim]),
                   startPos: startPos, inverse: false, in: cmd)

        // ---------- KV path ----------
        // kv = kv_norm(wkv(x))  → [B, S, head_dim]
        let kvFlat = kvNorm(wkv(xFlat, in: cmd), in: cmd)
        if TraceFlags.normTrace && (layerId == 0 || layerId == 5 || layerId == 6) {
            cmd.commit(); cmd.waitUntilCompleted()
            traceTensorStats("mla[\(layerId)] after wkv+kvNorm", kvFlat)
            cmd = Device.shared.queue.makeCommandBuffer()!
        }
        // RoPE on rope tail (interpret as [B*S, 1, head_dim]).
        rope.apply(kvFlat.reshape([B * S, 1, headDim]),
                   startPos: startPos, inverse: false, in: cmd)
        // FP8-simulate the non-rope dims of KV to match training-time QAT
        // noise. Mirrors model.py:506
        //     act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        // The model was trained with KV nope dims round-tripped through
        // FP8 (block 64, clipped to ±448). Without this step the KV
        // values exceed FP8 range in deeper layers, attention scores
        // become outsized, and the residual stream amplifies across
        // layers (observed: layer 0 L2=75 → layer 42 L2=615k).
        ActQuant.partialInplaceQuant(
            kvFlat.reshape([B * S, headDim]),
            colStart: 0, colEnd: headDim - ropeHeadDim,
            blockSize: Quant.actBlockSizeFP8KVNope, in: cmd)
        if TraceFlags.normTrace && (layerId == 0 || layerId == 5 || layerId == 6) {
            cmd.commit(); cmd.waitUntilCompleted()
            traceTensorStats("mla[\(layerId)] kv after fp8-QAT", kvFlat)
            cmd = Device.shared.queue.makeCommandBuffer()!
        }
        let kv = kvFlat.reshape([B, S, headDim])

        // ---------- Topk indices ----------
        // All built directly on GPU into a single [B, S, kWin+kComp] i32
        // tensor: window slice via `attn_window_indices_i32`, compressed
        // slice via either `attn_copy_compressed_i32` (indexer present)
        // or `attn_compressed_indices_i32` (ratio==128).
        //
        // Window indices use the actual startPos so the ring-buffer wrap
        // is handled correctly during decode. For prefill startPos == 0
        // the kernel produces the prefill table.
        let kWin = windowSize
        // Compressed offset: prefill compressed tokens are appended to `kv`
        // at index `kv.size(1)` == S. Decode reads from kvCache where the
        // compressed slice starts at `windowSize` (after the window ring).
        let compOffset = isDecode ? windowSize : S

        var kComp = 0
        if compressRatio > 0 {
            if let idx = indexer {
                let endPos = startPos + S
                let T = endPos / compressRatio
                kComp = min(idx.indexTopk, T)
            } else {
                kComp = isDecode
                    ? (startPos + 1) / compressRatio
                    : (S / compressRatio)
            }
        }
        let K = kWin + kComp

        let topkT = Tensor.empty(shape: [B, S, K], dtype: .i32)
        AttnIndicesGPU.window(into: topkT, B: B, S: S, K: K,
                              kWin: kWin, startPos: startPos, in: cmd)

        if compressRatio > 0 {
            if let idx = indexer {
                // Indexer returns [B, S, kComp] i32 on GPU; copy into the
                // right half of topkT. No commit, no host readback.
                let comp = idx(x, qr: qr, startPos: startPos,
                                offset: compOffset, in: cmd)
                AttnIndicesGPU.copyCompressed(into: topkT, from: comp,
                                               B: B, S: S, K: K,
                                               kWin: kWin, kComp: kComp,
                                               in: cmd)
            } else {
                AttnIndicesGPU.compressedDeterministic(into: topkT,
                                                        B: B, S: S, K: K,
                                                        kWin: kWin, kComp: kComp,
                                                        ratio: compressRatio,
                                                        offset: compOffset,
                                                        startPos: startPos,
                                                        in: cmd)
            }
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
        if TraceFlags.normTrace && (layerId == 0 || layerId == 5 || layerId == 6) {
            cmd.commit(); cmd.waitUntilCompleted()
            traceTensorStats("mla[\(layerId)] after sparse_attn", o)
            cmd = Device.shared.queue.makeCommandBuffer()!
        }

        // ---------- Inverse RoPE ----------
        rope.apply(o.reshape([B * S, nHeads, headDim]),
                   startPos: startPos, inverse: true, in: cmd)

        // ---------- Grouped output: einsum("bsgd,grd->bsgr"), then wo_b ----------
        let perGroupD = nHeads * headDim / nGroups
        let oView = o.reshape([B, S, nGroups, perGroupD])
        let woAR = woA.weight.reshape([nGroups, oLoraRank, perGroupD])
        // When wo_a is FP8 on disk (DeepSeek-V4-HF native, expert_dtype=fp4
        // / attn fmt=e4m3), we keep the FP8 + UE8M0 scale all the way to
        // the einsum kernel which dequantizes inline. For INT-quantized
        // / BF16-fused converted models woA.scale is nil and the kernel
        // takes the BF16 or f32 path automatically.
        let oR = Einsum.bsgdGrd(o: oView, woA: woAR,
                                  woAScale: woA.scale,
                                  in: cmd)                            // [B, S, nGroups, oLoraRank]
        if TraceFlags.normTrace && (layerId == 0 || layerId == 5 || layerId == 6) {
            cmd.commit(); cmd.waitUntilCompleted()
            traceTensorStats("mla[\(layerId)] after einsum wo_a", oR)
            cmd = Device.shared.queue.makeCommandBuffer()!
        }
        let oFlat = oR.reshape([B * S, nGroups * oLoraRank])
        let result = woB(oFlat, in: cmd).reshape([B, S, dim])
        if TraceFlags.normTrace && (layerId == 0 || layerId == 5 || layerId == 6) {
            cmd.commit(); cmd.waitUntilCompleted()
            traceTensorStats("mla[\(layerId)] after wo_b", result)
            cmd = Device.shared.queue.makeCommandBuffer()!
        }
        return result
    }

    private var dim: Int { wkv.inFeatures }
}
