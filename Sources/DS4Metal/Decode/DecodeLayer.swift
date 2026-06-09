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

public struct DSV4Dims {
    public var nEmbd: Int, nHC: Int, headDim: Int, nHead: Int
    public var qRank: Int, qDim: Int, sharedFfn: Int
    public var nExperts: Int, expertFfn: Int, k: Int, nRot: Int, vocab: Int
    public var nOutGroup: Int, nLoraO: Int   // low-rank attention output (n_out_group, n_lora_o)
    public var swigluClamp: Float            // DS4_SWIGLU_CLAMP_EXP (gate/up clamp limit, Flash=10)
    // Routed-expert quant formats + router precision (set from the GGUF; default to
    // the Q4_K-experts + Q8 router scheme). The IQ2_XXS 2-bit model uses iq2_xxs
    // gate/up, q2_K down, f16 router.
    public var gateQuant: MoEQuant = .q4_K
    public var upQuant: MoEQuant = .q4_K
    public var downQuant: MoEQuant = .q4_K
    public var routerF16: Bool = false
    /// Experts actually computed per token (≤ k). Lowering this below k=6 reduces
    /// expert I/O (fewer experts gathered from the mmap) and compute, at a quality
    /// cost — the model was trained with k=6. Honored by the streaming/gather path.
    public var activeExperts: Int = 6
    /// Use the fused MoE kernels (pair_swiglu + down_sum6, the C engine's release
    /// path: 2 dispatches instead of 5) when the quant scheme supports them.
    /// DS4_FUSED_MOE=0 disables for A/B comparison.
    public var fusedMoE: Bool = ProcessInfo.processInfo.environment["DS4_FUSED_MOE"] != "0"
    /// Per-group slice of the attention heads (qDim / nOutGroup).
    public var attnGroupDim: Int { qDim / nOutGroup }
    /// Low-rank attention output dim (nOutGroup * nLoraO).
    public var attnLowDim: Int { nOutGroup * nLoraO }
    public init(nEmbd: Int, nHC: Int, headDim: Int, nHead: Int, qRank: Int, qDim: Int,
                sharedFfn: Int, nExperts: Int, expertFfn: Int, k: Int, nRot: Int, vocab: Int,
                nOutGroup: Int = 8, nLoraO: Int = 1024, swigluClamp: Float = 10.0) {
        self.nEmbd = nEmbd; self.nHC = nHC; self.headDim = headDim; self.nHead = nHead
        self.qRank = qRank; self.qDim = qDim; self.sharedFfn = sharedFfn
        self.nExperts = nExperts; self.expertFfn = expertFfn; self.k = k; self.nRot = nRot; self.vocab = vocab
        self.nOutGroup = nOutGroup; self.nLoraO = nLoraO; self.swigluClamp = swigluClamp
    }
}

public struct RopeParams {
    public var nCtxOrig: Int, freqBase: Float, freqScale: Float, extFactor: Float
    public var attnFactor: Float, betaFast: Float, betaSlow: Float
    public init(nCtxOrig: Int, freqBase: Float, freqScale: Float, extFactor: Float,
                attnFactor: Float, betaFast: Float, betaSlow: Float) {
        self.nCtxOrig = nCtxOrig; self.freqBase = freqBase; self.freqScale = freqScale
        self.extFactor = extFactor; self.attnFactor = attnFactor; self.betaFast = betaFast; self.betaSlow = betaSlow
    }
}

/// All resident per-layer weights as GPUTensors. attn/ffn mixer scale/base are
/// 3-float / 24-float tensors; norms are nEmbd (or qRank/headDim) F32; q_a/q_b/kv/
/// attn_out/shared/router are Q8_0 bytes; experts are Q4_K bytes.
public struct LayerWeights {
    public var hcAttnFn: GPUTensor        // F32 [24 x hcDim]
    public var attnScale: GPUTensor       // F32 [3]
    public var attnBase: GPUTensor        // F32 [24]
    public var attnNorm: GPUTensor        // F32 [nEmbd]
    public var qA: GPUTensor              // Q8 [qRank x nEmbd]
    public var qANorm: GPUTensor          // F32 [qRank]
    public var qB: GPUTensor              // Q8 [qDim x qRank]
    public var kvW: GPUTensor             // Q8 [headDim x nEmbd]
    public var kvNorm: GPUTensor          // F32 [headDim]
    public var attnSinks: GPUTensor       // F32 [nHead] — per-head attention sink logit (softmax denom)
    public var attnOutA: GPUTensor        // Q8 output_a [nOutGroup x nLoraO x groupDim] (grouped low-rank)
    public var attnOut: GPUTensor         // Q8 output_b [nEmbd x lowDim]
    public var hcFfnFn: GPUTensor         // F32 [24 x hcDim]
    public var ffnScale: GPUTensor        // F32 [3]
    public var ffnBase: GPUTensor         // F32 [24]
    public var ffnNorm: GPUTensor         // F32 [nEmbd]
    public var sharedGate: GPUTensor, sharedUp: GPUTensor, sharedDown: GPUTensor  // Q8
    public var routerW: GPUTensor         // Q8 [nExperts x nEmbd]
    public var expGate: GPUTensor, expUp: GPUTensor, expDown: GPUTensor           // Q4_K experts
    // NSA attention compressor (present only on compressed layers, ratio!=0; nil on 0,1).
    // F16 projections from attn_norm to coff*headDim (coff=2 for ratio-4, 1 for ratio-128).
    public var compKv: GPUTensor?         // F16 attn_compressor_kv   [nEmbd x coff*headDim]
    public var compGate: GPUTensor?       // F16 attn_compressor_gate [nEmbd x coff*headDim]
    public var compApe: GPUTensor?        // attn_compressor_ape      [coff*headDim x ratio] (absolute pos emb)
    public var compNorm: GPUTensor?       // F32 attn_compressor_norm [headDim]
    public init(hcAttnFn: GPUTensor, attnScale: GPUTensor, attnBase: GPUTensor, attnNorm: GPUTensor,
                qA: GPUTensor, qANorm: GPUTensor, qB: GPUTensor, kvW: GPUTensor, kvNorm: GPUTensor,
                attnSinks: GPUTensor,
                attnOutA: GPUTensor, attnOut: GPUTensor, hcFfnFn: GPUTensor, ffnScale: GPUTensor, ffnBase: GPUTensor, ffnNorm: GPUTensor,
                sharedGate: GPUTensor, sharedUp: GPUTensor, sharedDown: GPUTensor, routerW: GPUTensor,
                expGate: GPUTensor, expUp: GPUTensor, expDown: GPUTensor,
                compKv: GPUTensor? = nil, compGate: GPUTensor? = nil,
                compApe: GPUTensor? = nil, compNorm: GPUTensor? = nil) {
        self.compKv = compKv; self.compGate = compGate; self.compApe = compApe; self.compNorm = compNorm
        self.hcAttnFn = hcAttnFn; self.attnScale = attnScale; self.attnBase = attnBase; self.attnNorm = attnNorm
        self.qA = qA; self.qANorm = qANorm; self.qB = qB; self.kvW = kvW; self.kvNorm = kvNorm
        self.attnSinks = attnSinks
        self.attnOutA = attnOutA; self.attnOut = attnOut; self.hcFfnFn = hcFfnFn; self.ffnScale = ffnScale; self.ffnBase = ffnBase; self.ffnNorm = ffnNorm
        self.sharedGate = sharedGate; self.sharedUp = sharedUp; self.sharedDown = sharedDown; self.routerW = routerW
        self.expGate = expGate; self.expUp = expUp; self.expDown = expDown
    }
}

/// Reusable scratch GPUTensors for one decode layer (allocate once, reuse across layers).
public final class DecodeScratch {
    let flat, mix, split, embd, cur: GPUTensor
    let qr, qrNorm, q, kvRaw, kv: GPUTensor
    let kvF16, mask, sinks, pad, tmp, heads, blockOut: GPUTensor
    let attnLow: GPUTensor   // low-rank attention output (nOutGroup * nLoraO)
    let afterAttn: GPUTensor
    let logits, sp, probs, selected, rw: GPUTensor
    let gate6, up6, mid6, down6, routed: GPUTensor
    let sgate, sup, smid, sdown, sharedOut, ffnOut: GPUTensor

    public init(_ rt: MetalRuntime, _ d: DSV4Dims, maxKeys: Int) throws {
        let hcDim = d.nHC * d.nEmbd
        flat = try .zeros(rt, floatCount: hcDim); mix = try .zeros(rt, floatCount: 24)
        split = try .zeros(rt, floatCount: 24); embd = try .zeros(rt, floatCount: d.nEmbd)
        cur = try .zeros(rt, floatCount: d.nEmbd)
        qr = try .zeros(rt, floatCount: d.qRank); qrNorm = try .zeros(rt, floatCount: d.qRank)
        q = try .zeros(rt, floatCount: d.qDim)
        kvRaw = try .zeros(rt, floatCount: d.headDim); kv = try .zeros(rt, floatCount: d.headDim)
        let sb = GraphContext.flashScratchBytes(nHead: d.nHead, nKeys: maxKeys)
        kvF16 = try .zerosBytes(rt, byteLength: sb.kvF16); mask = try .zerosBytes(rt, byteLength: sb.mask)
        sinks = try .zerosBytes(rt, byteLength: sb.sinks); pad = try .zerosBytes(rt, byteLength: sb.pad)
        tmp = try .zerosBytes(rt, byteLength: sb.tmp); heads = try .zeros(rt, floatCount: d.nHead * d.headDim)
        attnLow = try .zeros(rt, floatCount: d.attnLowDim)
        blockOut = try .zeros(rt, floatCount: d.nEmbd); afterAttn = try .zeros(rt, floatCount: d.nHC * d.nEmbd)
        logits = try .zeros(rt, floatCount: d.nExperts); sp = try .zeros(rt, floatCount: d.nExperts)
        probs = try .zeros(rt, floatCount: d.nExperts); selected = try .zerosBytes(rt, byteLength: d.k * 4)
        rw = try .zeros(rt, floatCount: d.k)
        gate6 = try .zeros(rt, floatCount: d.k * d.expertFfn); up6 = try .zeros(rt, floatCount: d.k * d.expertFfn)
        mid6 = try .zeros(rt, floatCount: d.k * d.expertFfn); down6 = try .zeros(rt, floatCount: d.k * d.nEmbd)
        routed = try .zeros(rt, floatCount: d.nEmbd)
        sgate = try .zeros(rt, floatCount: d.sharedFfn); sup = try .zeros(rt, floatCount: d.sharedFfn)
        smid = try .zeros(rt, floatCount: d.sharedFfn); sdown = try .zeros(rt, floatCount: d.nEmbd)
        sharedOut = try .zeros(rt, floatCount: d.nEmbd); ffnOut = try .zeros(rt, floatCount: d.nEmbd)
    }
}

extension GraphContext {
    /// One HC-reduce: flat=rmsNorm(curHc, hcDim); mix=matmulF32(mixerFn, flat);
    /// split=sinkhorn(mix, scale, base); embd=weightedSum(curHc, pre); cur=rmsNorm(embd, norm).
    private func hcReduce(curHc: GPUTensor, mixerFn: GPUTensor, scale: GPUTensor, base: GPUTensor,
                          norm: GPUTensor, s: DecodeScratch, d: DSV4Dims, eps: Float) throws {
        let hcDim = d.nHC * d.nEmbd
        try rmsNorm(curHc, weight: nil, out: s.flat, rows: 1, n: hcDim, eps: eps)
        try matmulF16(weight: mixerFn, x: s.flat, out: s.mix, inDim: hcDim, outDim: 24) // hc_attn_fn/hc_ffn_fn are F16
        try hcSplitSinkhorn(mix: s.mix, scale: scale, base: base, out: s.split, nRows: 1,
                            sinkhornIters: DSV4Shape.nHCSinkhornIter, eps: eps)
        try hcWeightedSum(x: curHc, weights: s.split, out: s.embd, nEmbd: d.nEmbd, nHC: d.nHC, nTokens: 1)
        try rmsNorm(s.embd, weight: norm, out: s.cur, rows: 1, n: d.nEmbd, eps: eps)
    }

    /// Full decode layer (resident experts, one command buffer). `curHc`
    /// (nHC*nEmbd) in; result in `outHc`. rawCache holds nKeys latent rows.
    public func decodeLayer(curHc: GPUTensor, w: LayerWeights, s: DecodeScratch, d: DSV4Dims,
                            rope: RopeParams, rawCache: GPUTensor, nKeys: Int, pos: Int,
                            outHc: GPUTensor, rmsEps: Float, hcEps: Float, comp: CompressorState? = nil) throws {
        try decodeRoute(curHc: curHc, w: w, s: s, d: d, rope: rope, rawCache: rawCache,
                        nKeys: nKeys, pos: pos, rmsEps: rmsEps, hcEps: hcEps, comp: comp)
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
                            rope: RopeParams, rawCache: GPUTensor, nKeys: Int, pos: Int,
                            rmsEps: Float, hcEps: Float, comp: CompressorState? = nil) throws {
        // 1) HC-reduce pre-attn  (s.cur = attn_norm)
        try hcReduce(curHc: curHc, mixerFn: w.hcAttnFn, scale: w.attnScale, base: w.attnBase,
                     norm: w.attnNorm, s: s, d: d, eps: rmsEps)
        // 1.5) NSA attention compressor (compressed layers only): update recurrent state
        // from attn_norm and, every `ratio` tokens, emit a pooled compressed KV row.
        var nComp = 0
        if let comp = comp {
            nComp = try runCompressor(attnNorm: s.cur, w: w, comp: comp, rope: rope,
                                      pos: pos, rmsEps: rmsEps, nRot: d.nRot)
        }
        // 2) Q path: q_a -> norm -> q_b -> head-norm -> rope
        try matmulQ8_0(weight: w.qA, x: s.cur, out: s.qr, inDim: d.nEmbd, outDim: d.qRank)
        try rmsNorm(s.qr, weight: w.qANorm, out: s.qrNorm, rows: 1, n: d.qRank, eps: rmsEps)
        try matmulQ8_0(weight: w.qB, x: s.qrNorm, out: s.q, inDim: d.qRank, outDim: d.qDim)
        try rmsNorm(s.q, weight: nil, out: s.q, rows: d.nHead, n: d.headDim, eps: rmsEps) // head norm, in-place
        try ropeTail(x: s.q, nTok: 1, nHead: d.nHead, headDim: d.headDim, nRot: d.nRot, nCtxOrig: rope.nCtxOrig,
                     freqBase: rope.freqBase, freqScale: rope.freqScale, extFactor: rope.extFactor,
                     attnFactor: rope.attnFactor, betaFast: rope.betaFast, betaSlow: rope.betaSlow, pos0: pos, posStep: 1)
        // 3) KV path: kv -> norm -> rope -> fp8 store into rawCache[pos]
        try matmulQ8_0(weight: w.kvW, x: s.cur, out: s.kvRaw, inDim: d.nEmbd, outDim: d.headDim)
        try rmsNorm(s.kvRaw, weight: w.kvNorm, out: s.kv, rows: 1, n: d.headDim, eps: rmsEps)
        try ropeTail(x: s.kv, nTok: 1, nHead: 1, headDim: d.headDim, nRot: d.nRot, nCtxOrig: rope.nCtxOrig,
                     freqBase: rope.freqBase, freqScale: rope.freqScale, extFactor: rope.extFactor,
                     attnFactor: rope.attnFactor, betaFast: rope.betaFast, betaSlow: rope.betaSlow, pos0: pos, posStep: 1)
        try kvFP8Store(kv: s.kv, rawCache: rawCache, headDim: d.headDim, nRot: d.nRot, rawRow: pos)
        // 4) attention over rawCache[0..nKeys] + comp.cache[0..nComp] -> heads
        try flashAttnCore(q: s.q, kvF32: rawCache, kvF16: s.kvF16, mask: s.mask, sinks: w.attnSinks,
                          pad: s.pad, tmp: s.tmp, heads: s.heads, nHead: d.nHead, nKeys: nKeys, hasSinks: true,
                          comp: comp?.cache, nComp: nComp)
        // 5) post-attn heads RoPE (inverse) + faithful low-rank output projection:
        //    attn_low = attnOutLowQ8(output_a, heads); blockOut = matmulQ8(output_b, attn_low);
        //    hcExpand4(blockOut, curHc, post=split[4:8], comb=split[8:24]) = afterAttn.
        try ropeTail(x: s.heads, nTok: 1, nHead: d.nHead, headDim: d.headDim, nRot: d.nRot, nCtxOrig: rope.nCtxOrig,
                     freqBase: rope.freqBase, freqScale: rope.freqScale, extFactor: rope.extFactor,
                     attnFactor: rope.attnFactor, betaFast: rope.betaFast, betaSlow: rope.betaSlow,
                     pos0: pos, posStep: 1, inverse: true)
        try attnOutLowQ8(outputA: w.attnOutA, heads: s.heads, low: s.attnLow,
                         nGroups: d.nOutGroup, groupDim: d.attnGroupDim, rank: d.nLoraO)
        try matmulQ8_0(weight: w.attnOut, x: s.attnLow, out: s.blockOut, inDim: d.attnLowDim, outDim: d.nEmbd)
        try hcExpand4(blockOut: s.blockOut, residual: curHc, post: s.split, comb: s.split,
                      blockAdd: nil, out: s.afterAttn, nEmbd: d.nEmbd, nTokens: 1,
                      postByteOffset: 4 * 4, combByteOffset: 8 * 4)
        // 6) HC-reduce pre-FFN (on afterAttn)
        try hcReduce(curHc: s.afterAttn, mixerFn: w.hcFfnFn, scale: w.ffnScale, base: w.ffnBase,
                     norm: w.ffnNorm, s: s, d: d, eps: rmsEps)
        // 7) router: logits -> softplus -> sqrt -> top-6 -> weights.
        //    ffn_gate_inp is Q8_0 in the Q4_K model but F16 in the IQ2_XXS model.
        if d.routerF16 {
            try matmulF16(weight: w.routerW, x: s.cur, out: s.logits, inDim: d.nEmbd, outDim: d.nExperts)
        } else {
            try matmulQ8_0(weight: w.routerW, x: s.cur, out: s.logits, inDim: d.nEmbd, outDim: d.nExperts)
        }
        try unary(s.logits, op: .softplus, out: s.sp, width: d.nExperts)
        try unary(s.sp, op: .sqrt, out: s.probs, width: d.nExperts)
        try routerFinalizeTop6(probs: s.probs, selected: s.selected)
        try routerWeights(probs: s.probs, selected: s.selected, weights: s.rw)
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
                              split: GPUTensor? = nil) throws {
        let kk = activeK < 0 ? d.k : max(1, min(activeK, d.k))
        let x = cur ?? s.cur
        let resid = afterAttn ?? s.afterAttn
        let sp = split ?? s.split
        // shared FFN: gate/up -> swiglu -> down
        try matmulQ8_0(weight: w.sharedGate, x: x, out: s.sgate, inDim: d.nEmbd, outDim: d.sharedFfn)
        try matmulQ8_0(weight: w.sharedUp, x: x, out: s.sup, inDim: d.nEmbd, outDim: d.sharedFfn)
        try swiglu(gate: s.sgate, up: s.sup, out: s.smid, n: d.sharedFfn, limit: d.swigluClamp)
        try matmulQ8_0(weight: w.sharedDown, x: s.smid, out: s.sharedOut, inDim: d.sharedFfn, outDim: d.nEmbd)
        // routed MoE over the provided experts (per-tensor quant: gate/up + down).
        // Fused C-release path (pair_swiglu + down_sum6, 2 dispatches) when the
        // quant scheme has the kernels; otherwise the validated 5-dispatch path.
        let pairFused = d.fusedMoE && d.gateQuant == d.upQuant
            && (d.gateQuant == .iq2_xxs || d.gateQuant == .q4_K)
        if pairFused {
            try moePairSwiGLU(d.gateQuant, gateExp: gateExp, upExp: upExp, ids: ids,
                              activation: x, weights: s.rw, gateScratch: s.gate6,
                              upScratch: s.up6, mid: s.mid6,
                              k: kk, inDim: d.nEmbd, outDim: d.expertFfn, clamp: d.swigluClamp)
        } else {
            try moeMatvecID(d.gateQuant, experts: gateExp, ids: ids, activation: x, out: s.gate6, k: kk, inDim: d.nEmbd, outDim: d.expertFfn, perExpertAct: false)
            try moeMatvecID(d.upQuant, experts: upExp, ids: ids, activation: x, out: s.up6, k: kk, inDim: d.nEmbd, outDim: d.expertFfn, perExpertAct: false)
            try moeSwiGLUWeight(gate: s.gate6, up: s.up6, weights: s.rw, mid: s.mid6, width: d.expertFfn, rows: kk, clampValue: d.swigluClamp)
        }
        // down_sum6 hardcodes 6 expert slots: usable only at full k.
        let sumFused = d.fusedMoE && kk == 6
            && (d.downQuant == .q2_K || d.downQuant == .q4_K)
        if sumFused {
            try moeDownSum6(d.downQuant, experts: downExp, ids: ids, mid: s.mid6,
                            out: s.routed, inDim: d.expertFfn, outDim: d.nEmbd)
        } else {
            try moeMatvecID(d.downQuant, experts: downExp, ids: ids, activation: s.mid6, out: s.down6, k: kk, inDim: d.expertFfn, outDim: d.nEmbd, perExpertAct: true)
            try moeSum6(experts: s.down6, out: s.routed, width: d.nEmbd)
        }
        try add(s.sharedOut, s.routed, out: s.ffnOut, width: d.nEmbd)
        // HC expand post-FFN (post=split[4:8], comb=split[8:24]) -> outHc
        try hcExpand4(blockOut: s.ffnOut, residual: resid, post: sp, comb: sp,
                      blockAdd: nil, out: outHc, nEmbd: d.nEmbd, nTokens: 1,
                      postByteOffset: 4 * 4, combByteOffset: 8 * 4)
    }
}
