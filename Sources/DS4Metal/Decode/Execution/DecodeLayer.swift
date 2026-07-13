import Foundation
import Metal

// Stage C: full per-token decode layer, composing the validated graph encode-ops
// in the exact order traced from metal_graph_encode_decode_layer (ds4.c). This is
// structural wiring; every sub-op is individually validated vs CPU. End-to-end
// NUMERICAL validation requires the real model (>=64GB) — here a smoke test only
// confirms it executes with finite output and consistent shapes.
//
// Layout assumptions (DSV4 Flash): HC=4 streams of nEmbd; MLA latent headDim=512
// used as both K and V; q_dim = nHead*headDim; mixHc = 2*HC + HC*HC = 24.
// The compression / sparse-indexer path (ratio!=0) is omitted in this first
// faithful version (dense attention over the raw cache); flash requires
// nKeys % 32 == 0 (pad path deferred).

extension GraphContext {
    /// Q4_K resident dense projection. The fallback is intentionally retained
    /// so field runs can compare output/token parity without rebuilding.
    private func denseQ4(_ weight: GPUTensor, _ x: GPUTensor, _ out: GPUTensor,
                         inDim: Int, outDim: Int, id0: GPUTensor,
                         d: DSV4Dims) throws {
        if d.denseQ4Kernel {
            try matmulQ4_K(weight: weight, x: x, out: out,
                           inDim: inDim, outDim: outDim)
        } else {
            try moeMatvecID(.q4_K, experts: weight, ids: id0, activation: x, out: out,
                            k: 1, inDim: inDim, outDim: outDim, perExpertAct: false)
        }
    }

    /// Exact decode-time NSA indexer top-K and f16 attention-mask construction,
    /// encoded after `indexerScoresEnc` in the SAME command buffer. This removes
    /// the historical GPU→CPU score readback and the second command-buffer wait.
    /// The Metal kernel mirrors IndexerSelect's heap comparator, so this changes
    /// scheduling only, not the selected compressed-row set.
    func indexerTopKMask(scores: GPUTensor, mask: GPUTensor,
                         nRaw: Int, nComp: Int, nScores: Int, topK: Int) throws {
        precondition(nRaw >= 0 && nComp >= 0 && nScores >= 0)
        precondition(nScores <= nComp && topK > 0)
        let keep = min(topK, nScores)
        var args = [UInt8](repeating: 0, count: 16)
        func u32(_ off: Int, _ value: Int) {
            var v = UInt32(value).littleEndian
            withUnsafeBytes(of: &v) { args.replaceSubrange(off..<(off + 4), with: $0) }
        }
        u32(0, nRaw); u32(4, nComp); u32(8, nScores); u32(12, topK)
        let pso = try rt.pipeline("kernel_dsv4_indexer_topk_mask_one")
        let nth = max(32, keep)
        guard nth <= pso.maxTotalThreadsPerThreadgroup else {
            throw MetalError.unsupported(
                "indexer GPU top-k richiede \(nth) thread, limite \(pso.maxTotalThreadsPerThreadgroup)")
        }
        let e = encoder
        e.setComputePipelineState(pso)
        args.withUnsafeBytes { e.setBytes($0.baseAddress!, length: args.count, index: 0) }
        e.setBuffer(scores.buffer, offset: scores.byteOffset, index: 1)
        e.setBuffer(mask.buffer, offset: mask.byteOffset, index: 2)
        e.setThreadgroupMemoryLength(max(1, keep) * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: nth, height: 1, depth: 1))
    }

    /// One HC-reduce: flat=rmsNorm(curHc, hcDim); mix=matmulF32(mixerFn, flat);
    /// split=sinkhorn(mix, scale, base); embd=weightedSum(curHc, pre); cur=rmsNorm(embd, norm).
    private func hcReduce(curHc: GPUTensor, mixerFn: GPUTensor, scale: GPUTensor, base: GPUTensor,
                          norm: GPUTensor, s: DecodeScratch, d: DSV4Dims,
                          rmsEps: Float, hcEps: Float) throws {
        let hcDim = d.nHC * d.nEmbd
        try rmsNorm(curHc, weight: nil, out: s.flat, rows: 1, n: hcDim, eps: rmsEps)
        try matmulF16(weight: mixerFn, x: s.flat, out: s.mix, inDim: hcDim, outDim: 24) // hc_attn_fn/hc_ffn_fn are F16
        if d.fusedHC && d.nHC == 4 {
            // C decode release path: split + collapse + norm in one dispatch.
            // Writes the same three outputs (s.split / s.embd / s.cur) as the
            // unfused triple below — downstream consumers see no difference.
            try hcSplitWeightedSumNorm4(mix: s.mix, scale: scale, base: base, x: curHc,
                                        split: s.split, embd: s.embd, normWeight: norm,
                                        normOut: s.cur, nEmbd: d.nEmbd, nRows: 1,
                                        sinkhornIters: DSV4Shape.nHCSinkhornIter,
                                        eps: hcEps, normEps: rmsEps)
        } else {
            try hcSplitSinkhorn(mix: s.mix, scale: scale, base: base, out: s.split, nRows: 1,
                                sinkhornIters: DSV4Shape.nHCSinkhornIter, eps: hcEps)
            try hcWeightedSum(x: curHc, weights: s.split, out: s.embd, nEmbd: d.nEmbd, nHC: d.nHC, nTokens: 1)
            try rmsNorm(s.embd, weight: norm, out: s.cur, rows: 1, n: d.nEmbd, eps: rmsEps)
        }
    }

    /// Full decode layer (resident experts, one command buffer). `curHc`
    /// (nHC*nEmbd) in; result in `outHc`. rawCache holds nKeys latent rows.
    public func decodeLayer(curHc: GPUTensor, w: LayerWeights, s: DecodeScratch, d: DSV4Dims,
                            rope: RopeParams, rawCache: GPUTensor, nKeys: Int, pos: Int, token: Int = -1,
                            outHc: GPUTensor, rmsEps: Float, hcEps: Float, comp: CompressorState? = nil) throws {
        try decodeRoute(curHc: curHc, w: w, s: s, d: d, rope: rope, rawCache: rawCache,
                        nKeys: nKeys, pos: pos, token: token, rmsEps: rmsEps, hcEps: hcEps, comp: comp)
        try decodeExperts(w: w, s: s, d: d, gateExp: w.expGate, upExp: w.expUp, downExp: w.expDown,
                          ids: s.selected, outHc: outHc)
    }

    /// Phase 1: HC-reduce pre-attn -> Q/KV -> attention -> attn-out + hcExpand4
    /// -> HC-reduce pre-FFN -> router selection + weights. Leaves s.cur (FFN
    /// input), s.afterAttn (residual), s.split (pre-FFN HC split), s.selected
    /// (the 6 expert ids), s.rw (route weights). Used both by the resident path
    /// and the expert-cache path (which commits here, reads s.selected, gathers
    /// the 6 experts, then runs decodeExperts).
    public func decodeRoute(curHc: GPUTensor, w: LayerWeights, s: DecodeScratch, d: DSV4Dims,
                            rope: RopeParams, rawCache: GPUTensor, nKeys: Int, pos: Int, token: Int = -1,
                            rmsEps: Float, hcEps: Float, comp: CompressorState? = nil) throws {
        let nComp = try decodeRoutePre(curHc: curHc, w: w, s: s, d: d, rope: rope, rawCache: rawCache,
                                       pos: pos, rmsEps: rmsEps, hcEps: hcEps, comp: comp, idx: nil,
                                       indexerScoring: false)
        try decodeRouteAttn(curHc: curHc, w: w, s: s, d: d, rope: rope, rawCache: rawCache, nKeys: nKeys, pos: pos,
                            token: token, rmsEps: rmsEps, hcEps: hcEps, nComp: nComp, comp: comp)
    }

    /// Phase 1a (pre-attention): HC-reduce, attention+indexer compressor updates,
    /// Q/KV paths, fp8 raw store, and — when `indexerScoring` — the indexer
    /// relevance scores for this token (s.idxScores[0..nIdxComp)). Split from the
    /// attention so the caller can commit, run the CPU top-K and write the
    /// compressed-row mask before encoding the attention (C: indexer_allowed_
    /// decode_one + the dense top-k mask path). Returns n_comp visible to this token.
    public func decodeRoutePre(curHc: GPUTensor, w: LayerWeights, s: DecodeScratch, d: DSV4Dims,
                               rope: RopeParams, rawCache: GPUTensor, pos: Int,
                               rmsEps: Float, hcEps: Float, comp: CompressorState?, idx: CompressorState?,
                               indexerScoring: Bool) throws -> Int {
        // 1) HC-reduce pre-attn  (s.cur = attn_norm)
        try hcReduce(curHc: curHc, mixerFn: w.hcAttnFn, scale: w.attnScale, base: w.attnBase,
                     norm: w.attnNorm, s: s, d: d, rmsEps: rmsEps, hcEps: hcEps)
        // 1.5) NSA attention compressor (compressed layers only): update recurrent state
        // from attn_norm and, every `ratio` tokens, emit a pooled compressed KV row.
        var nComp = 0
        if let comp = comp {
            nComp = try runCompressor(attnNorm: s.cur, w: w, comp: comp, rope: rope,
                                      pos: pos, rmsEps: rmsEps, nRot: d.nRot)
        }
        // 1.6) NSA INDEXER compressor (ratio-4 layers): same recurrence on its own
        // weights (head_dim 128), finalized with Hadamard+FP4 instead of fp8.
        var nIdxComp = 0
        if let idx = idx, let ikv = w.idxKv, let igate = w.idxGate,
           let iape = w.idxApe, let inorm = w.idxNorm {
            nIdxComp = try runCompressor(x: s.cur, kv: ikv, gate: igate, ape: iape, normW: inorm,
                                         comp: idx, rope: rope, pos: pos, rmsEps: rmsEps,
                                         nRot: d.nRot, finalize: .indexerQat,
                                         q8Weights: w.idxCompQ8)
        }
        try phase("comp")                             // DS4_PROFILE_ROUTE boundary (hc-pre + compressor)
        // 2) Q path: q_a -> norm -> q_b -> head-norm -> rope
        encoder.pushDebugGroup("q-proj")              // Instruments: name the GPU phase
        if w.qAQ4 {
            // DS4_QKV_Q4: q_a requantized to Q4_K — same single-"expert"
            // dispatch as q_b below.
            try denseQ4(w.qA, s.cur, s.qr, inDim: d.nEmbd, outDim: d.qRank,
                        id0: s.id0, d: d)
        } else {
            try matmulQ8_0(weight: w.qA, x: s.cur, out: s.qr, inDim: d.nEmbd, outDim: d.qRank)
        }
        try rmsNorm(s.qr, weight: w.qANorm, out: s.qrNorm, rows: 1, n: d.qRank, eps: rmsEps)
        if w.qBQ4 {
            // DS4_DENSE_Q4: q_b requantized to Q4_K — dense matvec through the
            // validated MoE id-kernel with a single "expert" (k=1, id 0).
            try denseQ4(w.qB, s.qrNorm, s.q, inDim: d.qRank, outDim: d.qDim,
                        id0: s.id0, d: d)
        } else {
            try matmulQ8_0(weight: w.qB, x: s.qrNorm, out: s.q, inDim: d.qRank, outDim: d.qDim)
        }
        try rmsNorm(s.q, weight: nil, out: s.q, rows: d.nHead, n: d.headDim, eps: rmsEps) // head norm, in-place
        try ropeTail(x: s.q, nTok: 1, nHead: d.nHead, headDim: d.headDim, nRot: d.nRot, nCtxOrig: rope.nCtxOrig,
                     freqBase: rope.freqBase, freqScale: rope.freqScale, extFactor: rope.extFactor,
                     attnFactor: rope.attnFactor, betaFast: rope.betaFast, betaSlow: rope.betaSlow, pos0: pos, posStep: 1)
        encoder.popDebugGroup()                       // q-proj
        try phase("q")                                // DS4_PROFILE_ROUTE boundary (q proj)
        // 3) KV path: kv -> norm -> rope -> fp8 store into rawCache[pos]
        encoder.pushDebugGroup("kv-proj")
        if w.kvQ4 {
            // DS4_QKV_Q4: kv requantized to Q4_K (see q_a above).
            try denseQ4(w.kvW, s.cur, s.kvRaw, inDim: d.nEmbd, outDim: d.headDim,
                        id0: s.id0, d: d)
        } else {
            try matmulQ8_0(weight: w.kvW, x: s.cur, out: s.kvRaw, inDim: d.nEmbd, outDim: d.headDim)
        }
        try rmsNorm(s.kvRaw, weight: w.kvNorm, out: s.kv, rows: 1, n: d.headDim, eps: rmsEps)
        try ropeTail(x: s.kv, nTok: 1, nHead: 1, headDim: d.headDim, nRot: d.nRot, nCtxOrig: rope.nCtxOrig,
                     freqBase: rope.freqBase, freqScale: rope.freqScale, extFactor: rope.extFactor,
                     attnFactor: rope.attnFactor, betaFast: rope.betaFast, betaSlow: rope.betaSlow, pos0: pos, posStep: 1)
        try kvFP8Store(kv: s.kv, rawCache: rawCache, headDim: d.headDim, nRot: d.nRot,
                       rawRow: pos % (rawCache.count / d.headDim))   // ring slot (= pos with the full cache)
        encoder.popDebugGroup()                       // kv-proj
        // 3.5) Indexer scoring (only when the comp rows exceed the top-K): the
        // indexer query comes from qr_norm via indexer.attn_q_b (+rope+Hadamard),
        // the head weights from attn_norm via indexer.proj. C: indexer_allowed_decode_one.
        if indexerScoring, let idx = idx, nIdxComp > 0,
           let iqb = w.idxQB, let iproj = w.idxProj {
            let ih = d.nIndexerHeadDim, inH = d.nIndexerHead
            if w.idxQBF16 {
                try matmulF16(weight: iqb, x: s.qrNorm, out: s.idxQ, inDim: d.qRank, outDim: inH * ih)
            } else {
                try matmulQ8_0(weight: iqb, x: s.qrNorm, out: s.idxQ, inDim: d.qRank, outDim: inH * ih)
            }
            try ropeTail(x: s.idxQ, nTok: 1, nHead: inH, headDim: ih, nRot: d.nRot, nCtxOrig: rope.nCtxOrig,
                         freqBase: rope.freqBase, freqScale: rope.freqScale, extFactor: rope.extFactor,
                         attnFactor: rope.attnFactor, betaFast: rope.betaFast, betaSlow: rope.betaSlow,
                         pos0: pos, posStep: 1)
            try indexerHadamardFp4Enc(s.idxQ, rows: inH, rowStrideBytes: ih * 4)
            try matmulF16(weight: iproj, x: s.cur, out: s.idxW, inDim: d.nEmbd, outDim: inH)
            let scale = 1.0 / Float(ih * inH).squareRoot()
            try indexerScoresEnc(q: s.idxQ, weights: s.idxW, indexComp: idx.cache,
                                 scores: s.idxScores, nComp: nIdxComp, nHead: inH, headDim: ih, scale: scale)
        }
        return nComp
    }

    /// Phase 1b: attention (raw SWA window + compressed rows, with s.mask possibly
    /// carrying the indexer top-K selection) -> attn out (residual: curHc) ->
    /// pre-FFN HC -> router.
    public func decodeRouteAttn(curHc: GPUTensor, w: LayerWeights, s: DecodeScratch, d: DSV4Dims,
                                rope: RopeParams, rawCache: GPUTensor, nKeys: Int, pos: Int, token: Int = -1,
                                rmsEps: Float, hcEps: Float, nComp: Int, comp: CompressorState?) throws {
        // 4) attention over the raw SWA WINDOW + comp.cache[0..nComp] -> heads.
        //    Only the last nSWA raw rows are visible (C slides its raw cache at
        //    that cap); attending ALL raw rows would be out-of-distribution for
        //    the model and degrades long generations. Rows carry their absolute
        //    RoPE position, so dropping the oldest is exactly the C semantics.
        let rawLo = max(0, nKeys - d.nSWA)
        encoder.pushDebugGroup("attention")
        try flashAttnCore(q: s.q, kvF32: rawCache, kvF16: s.kvF16, mask: s.mask, sinks: w.attnSinks,
                          pad: s.pad, tmp: s.tmp, heads: s.heads, nHead: d.nHead, nKeys: nKeys - rawLo,
                          rawStartRow: rawLo, hasSinks: true,
                          comp: comp?.cache, nComp: nComp)
        encoder.popDebugGroup()                       // attention
        try phase("attn")                             // DS4_PROFILE_ROUTE boundary (flash-attn)
        encoder.pushDebugGroup("attn-out")
        // 5) post-attn heads RoPE (inverse) + faithful low-rank output projection:
        //    attn_low = attnOutLowQ8(output_a, heads); blockOut = matmulQ8(output_b, attn_low);
        //    hcExpand4(blockOut, curHc, post=split[4:8], comb=split[8:24]) = afterAttn.
        try ropeTail(x: s.heads, nTok: 1, nHead: d.nHead, headDim: d.headDim, nRot: d.nRot, nCtxOrig: rope.nCtxOrig,
                     freqBase: rope.freqBase, freqScale: rope.freqScale, extFactor: rope.extFactor,
                     attnFactor: rope.attnFactor, betaFast: rope.betaFast, betaSlow: rope.betaSlow,
                     pos0: pos, posStep: 1, inverse: true)
        if w.attnOutAQ4 {
            // DS4_DENSE_Q4: the grouped low-rank projection IS a MoE matvec
            // with all nOutGroup "experts" selected — group g's weight slab
            // maps heads[g·groupDim…] to low[g·rank…] (perExpertAct row g).
            if d.denseQ4Kernel {
                try attnOutLowQ4(outputA: w.attnOutA, heads: s.heads, low: s.attnLow,
                                 nGroups: d.nOutGroup, groupDim: d.attnGroupDim,
                                 rank: d.nLoraO)
            } else {
                try moeMatvecID(.q4_K, experts: w.attnOutA, ids: s.idsGroup, activation: s.heads,
                                out: s.attnLow, k: d.nOutGroup, inDim: d.attnGroupDim,
                                outDim: d.nLoraO, perExpertAct: true)
            }
        } else {
            try attnOutLowQ8(outputA: w.attnOutA, heads: s.heads, low: s.attnLow,
                             nGroups: d.nOutGroup, groupDim: d.attnGroupDim, rank: d.nLoraO)
        }
        if w.attnOutQ4 {
            // DS4_DENSE_Q4: output_b requantized to Q4_K (see q_b above).
            try denseQ4(w.attnOut, s.attnLow, s.blockOut,
                        inDim: d.attnLowDim, outDim: d.nEmbd, id0: s.id0, d: d)
        } else {
            try matmulQ8_0(weight: w.attnOut, x: s.attnLow, out: s.blockOut, inDim: d.attnLowDim, outDim: d.nEmbd)
        }
        try hcExpand4(blockOut: s.blockOut, residual: curHc, post: s.split, comb: s.split,
                      blockAdd: nil, out: s.afterAttn, nEmbd: d.nEmbd, nTokens: 1,
                      postByteOffset: 4 * 4, combByteOffset: 8 * 4)
        encoder.popDebugGroup()                       // attn-out
        try phase("out-proj")                         // output_a/b + HC expand
        encoder.pushDebugGroup("hc-ffn")
        // 6) HC-reduce pre-FFN (on afterAttn)
        try hcReduce(curHc: s.afterAttn, mixerFn: w.hcFfnFn, scale: w.ffnScale, base: w.ffnBase,
                     norm: w.ffnNorm, s: s, d: d, rmsEps: rmsEps, hcEps: hcEps)
        encoder.popDebugGroup()                       // hc-ffn
        try phase("hc-ffn")                           // pre-FFN HC reduce
        encoder.pushDebugGroup("router")
        // 7) router: logits -> softplus -> sqrt -> top-6 -> weights.
        //    ffn_gate_inp is Q8_0 in the Q4_K model but F16 in the IQ2_XXS model.
        if d.routerF16 {
            try matmulF16(weight: w.routerW, x: s.cur, out: s.logits, inDim: d.nEmbd, outDim: d.nExperts)
        } else {
            try matmulQ8_0(weight: w.routerW, x: s.cur, out: s.logits, inDim: d.nEmbd, outDim: d.nExperts)
        }
        if d.fusedRouterProbs {
            try routerProbabilities(logits: s.logits, probabilities: s.probs,
                                    width: d.nExperts)
        } else {
            try unary(s.logits, op: .softplus, out: s.sp, width: d.nExperts)
            try unary(s.sp, op: .sqrt, out: s.probs, width: d.nExperts)
        }
        // Selection: hash table on the first n_hash_layer layers (row = token id),
        // biased top-6 elsewhere; weights ALWAYS normalize the unbiased probs of
        // the selected experts (ds4.c layer_routed_moe_one + router kernel args).
        if w.tid2eid != nil && token < 0 {
            throw MetalError.unsupported("hash-routed layer dispatched without a token id")
        }
        try routerFinalizeTop6(probs: s.probs, selected: s.selected, bias: w.expBias,
                               hashTable: w.tid2eid, hashRows: w.tid2eidRows, token: token,
                               weights: d.fusedRouterFinalize ? s.rw : nil)
        if !d.fusedRouterFinalize {
            try routerWeights(probs: s.probs, selected: s.selected, weights: s.rw)
        }
        encoder.popDebugGroup()                       // router
    }

    /// Phase 2: shared FFN + routed MoE (over `gateExp`/`upExp`/`downExp` indexed
    /// by `ids`) + residual HC expand -> outHc. Resident path: pass w.expGate/Up/
    /// Down + s.selected. Expert-cache path: pass the packed 6 experts + an ids
    /// 0..<6 buffer (after gathering only the selected experts from the GGUF).
    /// `activeK` (≤ d.k) experts actually computed; defaults to d.k. With activeK<d.k
    /// the caller must have packed activeK experts and zeroed `s.down6` rows
    /// activeK..d.k-1 (so the fixed `moeSum6` adds zeros for the unused slots).
    /// `cur`/`afterAttn`/`split` override the scratch FFN inputs (used by the
    /// batched prefill, which saves them per token during the route phase).
    public func decodeExperts(w: LayerWeights, s: DecodeScratch, d: DSV4Dims,
                              gateExp: GPUTensor, upExp: GPUTensor, downExp: GPUTensor,
                              ids: GPUTensor, outHc: GPUTensor, activeK: Int = -1,
                              cur: GPUTensor? = nil, afterAttn: GPUTensor? = nil,
                              split: GPUTensor? = nil, rw: GPUTensor? = nil,
                              expertStride: Int? = nil) throws {
        try decodeSharedFFN(w: w, s: s, d: d, cur: cur)
        try decodeRoutedExperts(w: w, s: s, d: d, gateExp: gateExp, upExp: upExp,
                                downExp: downExp, ids: ids, outHc: outHc, activeK: activeK,
                                cur: cur, afterAttn: afterAttn, split: split, rw: rw,
                                expertStride: expertStride)
    }

    /// The shared-expert FFN half of decodeExperts (gate/up -> swiglu -> down into
    /// s.sharedOut). It does NOT depend on the routing selection, so the streaming
    /// decode path commits it asynchronously and the GPU runs it WHILE the CPU
    /// gathers the routed experts from the SSD (I/O/compute overlap). Same
    /// dispatches in the same order as the fused sequence — identical numerics.
    public func decodeSharedFFN(w: LayerWeights, s: DecodeScratch, d: DSV4Dims,
                                cur: GPUTensor? = nil) throws {
        let x = cur ?? s.cur
        // shared FFN: gate/up -> swiglu -> down. Each projection dispatches on
        // its own quant: Q4_K (DS4_SHARED_Q4 resident requant) runs through the
        // validated MoE id-kernel with a single "expert" (k=1, id 0), exactly
        // like the DS4_DENSE_Q4 attention projections.
        if w.sharedGateQ4 {
            try denseQ4(w.sharedGate, x, s.sgate, inDim: d.nEmbd, outDim: d.sharedFfn,
                        id0: s.id0, d: d)
        } else {
            try matmulQ8_0(weight: w.sharedGate, x: x, out: s.sgate, inDim: d.nEmbd, outDim: d.sharedFfn)
        }
        if w.sharedUpQ4 {
            try denseQ4(w.sharedUp, x, s.sup, inDim: d.nEmbd, outDim: d.sharedFfn,
                        id0: s.id0, d: d)
        } else {
            try matmulQ8_0(weight: w.sharedUp, x: x, out: s.sup, inDim: d.nEmbd, outDim: d.sharedFfn)
        }
        try swiglu(gate: s.sgate, up: s.sup, out: s.smid, n: d.sharedFfn, limit: d.swigluClamp)
        if w.sharedDownQ4 {
            try denseQ4(w.sharedDown, s.smid, s.sharedOut,
                        inDim: d.sharedFfn, outDim: d.nEmbd, id0: s.id0, d: d)
        } else {
            try matmulQ8_0(weight: w.sharedDown, x: s.smid, out: s.sharedOut, inDim: d.sharedFfn, outDim: d.nEmbd)
        }
    }

    /// The routed-MoE half of decodeExperts: matvec over the provided experts, add
    /// of s.sharedOut (which MUST already be encoded — decodeSharedFFN, either on
    /// this command buffer or on one already committed) and the HC expand -> outHc.
    /// `expertStride`: byte fra un esperto e il successivo nei buffer gate/up/
    /// down — nil = packing stretto; il pool INTERLEAVED della slot-cache passa
    /// la dimensione del record gate|up|down (le tre viste condividono un solo
    /// buffer, un miss è UNA pread dal bundle).
    public func decodeRoutedExperts(w: LayerWeights, s: DecodeScratch, d: DSV4Dims,
                                    gateExp: GPUTensor, upExp: GPUTensor, downExp: GPUTensor,
                                    ids: GPUTensor, outHc: GPUTensor, activeK: Int = -1,
                                    cur: GPUTensor? = nil, afterAttn: GPUTensor? = nil,
                                    split: GPUTensor? = nil, rw: GPUTensor? = nil,
                                    expertStride: Int? = nil) throws {
        let kk = activeK < 0 ? d.k : max(1, min(activeK, d.k))
        let x = cur ?? s.cur
        let resid = afterAttn ?? s.afterAttn
        let sp = split ?? s.split
        // Route weights: per-token buffer in the batched prefill (many tokens
        // share ONE command buffer, so the shared s.rw can't be rewritten
        // between them); s.rw everywhere else.
        let weights = rw ?? s.rw
        // routed MoE over the provided experts, dispatched on the PER-LAYER quant
        // (w.*Quant) — so a mixed-precision GGUF's boosted layer uses the right
        // kernel. Fused C-release path (pair_swiglu + down_sum6, 2 dispatches) when
        // the quant scheme has the kernels; otherwise the validated 5-dispatch path.
        let pairFused = d.fusedMoE && w.gateQuant == w.upQuant
            && (w.gateQuant == .iq2_xxs || w.gateQuant == .q4_K)
        if pairFused {
            try moePairSwiGLU(w.gateQuant, gateExp: gateExp, upExp: upExp, ids: ids,
                              activation: x, weights: weights, gateScratch: s.gate6,
                              upScratch: s.up6, mid: s.mid6,
                              k: kk, inDim: d.nEmbd, outDim: d.expertFfn, clamp: d.swigluClamp,
                              expertStride: expertStride)
        } else {
            try moeMatvecID(w.gateQuant, experts: gateExp, ids: ids, activation: x, out: s.gate6, k: kk, inDim: d.nEmbd, outDim: d.expertFfn, perExpertAct: false, expertStride: expertStride)
            try moeMatvecID(w.upQuant, experts: upExp, ids: ids, activation: x, out: s.up6, k: kk, inDim: d.nEmbd, outDim: d.expertFfn, perExpertAct: false, expertStride: expertStride)
            try moeSwiGLUWeight(gate: s.gate6, up: s.up6, weights: weights, mid: s.mid6, width: d.expertFfn, rows: kk, clampValue: d.swigluClamp)
        }
        // down_sum6 hardcodes 6 expert slots: usable only at full k.
        let sumFused = d.fusedMoE && kk == 6
            && (w.downQuant == .q2_K || w.downQuant == .q4_K)
        if sumFused {
            try moeDownSum6(w.downQuant, experts: downExp, ids: ids, mid: s.mid6,
                            out: s.routed, inDim: d.expertFfn, outDim: d.nEmbd,
                            expertStride: expertStride)
        } else {
            try moeMatvecID(w.downQuant, experts: downExp, ids: ids, activation: s.mid6, out: s.down6, k: kk, inDim: d.expertFfn, outDim: d.nEmbd, perExpertAct: true, expertStride: expertStride)
            try moeSum6(experts: s.down6, out: s.routed, width: d.nEmbd)
        }
        try add(s.sharedOut, s.routed, out: s.ffnOut, width: d.nEmbd)
        // HC expand post-FFN (post=split[4:8], comb=split[8:24]) -> outHc
        try hcExpand4(blockOut: s.ffnOut, residual: resid, post: sp, comb: sp,
                      blockAdd: nil, out: outHc, nEmbd: d.nEmbd, nTokens: 1,
                      postByteOffset: 4 * 4, combByteOffset: 8 * 4)
    }

    /// Expert parallelism (worker shard VERTICALE, Fase B): SOLO la somma
    /// pesata degli esperti forniti — la stessa sequenza di dispatch di
    /// decodeRoutedExperts fino a s.routed, SENZA shared-add né HC expand
    /// (vivono sul coordinatore). I quant arrivano espliciti (lo shard non
    /// carica LayerWeights). Input: s.cur (attivazione), s.rw (pesi di route,
    /// k valori). ATTENZIONE: quando il percorso non fuso è attivo (k < 6),
    /// moe_sum6 somma SEMPRE 6 righe — il chiamante DEVE aver azzerato le
    /// righe k..<6 di s.down6 prima di encodare.
    public func decodeExpertPartial(s: DecodeScratch, d: DSV4Dims,
                                    gateQuant: MoEQuant, upQuant: MoEQuant, downQuant: MoEQuant,
                                    gateExp: GPUTensor, upExp: GPUTensor, downExp: GPUTensor,
                                    ids: GPUTensor, k: Int, out: GPUTensor,
                                    expertStride: Int? = nil) throws {
        let kk = max(1, min(k, d.k))
        let pairFused = d.fusedMoE && gateQuant == upQuant
            && (gateQuant == .iq2_xxs || gateQuant == .q4_K)
        if pairFused {
            try moePairSwiGLU(gateQuant, gateExp: gateExp, upExp: upExp, ids: ids,
                              activation: s.cur, weights: s.rw, gateScratch: s.gate6,
                              upScratch: s.up6, mid: s.mid6,
                              k: kk, inDim: d.nEmbd, outDim: d.expertFfn, clamp: d.swigluClamp,
                              expertStride: expertStride)
        } else {
            try moeMatvecID(gateQuant, experts: gateExp, ids: ids, activation: s.cur, out: s.gate6,
                            k: kk, inDim: d.nEmbd, outDim: d.expertFfn, perExpertAct: false,
                            expertStride: expertStride)
            try moeMatvecID(upQuant, experts: upExp, ids: ids, activation: s.cur, out: s.up6,
                            k: kk, inDim: d.nEmbd, outDim: d.expertFfn, perExpertAct: false,
                            expertStride: expertStride)
            try moeSwiGLUWeight(gate: s.gate6, up: s.up6, weights: s.rw, mid: s.mid6,
                                width: d.expertFfn, rows: kk, clampValue: d.swigluClamp)
        }
        let sumFused = d.fusedMoE && kk == 6 && (downQuant == .q2_K || downQuant == .q4_K)
        if sumFused {
            try moeDownSum6(downQuant, experts: downExp, ids: ids, mid: s.mid6,
                            out: out, inDim: d.expertFfn, outDim: d.nEmbd,
                            expertStride: expertStride)
        } else {
            try moeMatvecID(downQuant, experts: downExp, ids: ids, activation: s.mid6, out: s.down6,
                            k: kk, inDim: d.expertFfn, outDim: d.nEmbd, perExpertAct: true,
                            expertStride: expertStride)
            try moeSum6(experts: s.down6, out: out, width: d.nEmbd)
        }
    }

    /// Coda del layer per l'expert parallelism (coordinatore VERTICALE, Fase C):
    /// la somma pesata degli esperti routed arriva DALLA RETE (già ridotta tra
    /// i worker) in `partial` — restano l'add della FFN condivisa (s.sharedOut,
    /// già encodata) e l'HC expand, identici al percorso locale.
    public func decodeRemoteTail(s: DecodeScratch, d: DSV4Dims, partial: GPUTensor,
                                 outHc: GPUTensor) throws {
        try add(s.sharedOut, partial, out: s.ffnOut, width: d.nEmbd)
        try hcExpand4(blockOut: s.ffnOut, residual: s.afterAttn, post: s.split, comb: s.split,
                      blockAdd: nil, out: outHc, nEmbd: d.nEmbd, nTokens: 1,
                      postByteOffset: 4 * 4, combByteOffset: 8 * 4)
    }

    /// Tail of the routed FFN for the batched-MM prefill (DS4_PREFILL_MM): the
    /// k weighted down rows are already in s.down6 (blitted from the group's
    /// matmul output) — sum6 + shared add + HC expand, the same dispatches the
    /// matvec path ends with. s.sharedOut must already be encoded.
    public func decodeRoutedTail(s: DecodeScratch, d: DSV4Dims, outHc: GPUTensor,
                                 afterAttn: GPUTensor, split: GPUTensor) throws {
        try moeSum6(experts: s.down6, out: s.routed, width: d.nEmbd)
        try add(s.sharedOut, s.routed, out: s.ffnOut, width: d.nEmbd)
        try hcExpand4(blockOut: s.ffnOut, residual: afterAttn, post: split, comb: split,
                      blockAdd: nil, out: outHc, nEmbd: d.nEmbd, nTokens: 1,
                      postByteOffset: 4 * 4, combByteOffset: 8 * 4)
    }
}

