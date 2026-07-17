import Foundation

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
    // DS4_COMP_Q8: the compressor kv/gate projections requantized F16 → Q8_0 at
    // load (resident, DenseStreamer+CompressorQ8); the dispatch reads THIS flag.
    public var compQ8 = false
    // NSA indexer (DSA, ratio-4 layers only): its own compressor (head_dim=128)
    // plus the per-token scoring projections. All optional (nil elsewhere).
    public var idxQB: GPUTensor?          // indexer.attn_q_b  [qRank x 64*128] (F16 or Q8)
    public var idxQBF16 = true            // idxQB precision (the C accepts both)
    public var idxProj: GPUTensor?        // F16 indexer.proj  [nEmbd x 64]
    public var idxKv: GPUTensor?          // F16 indexer_compressor_kv   [nEmbd x coff*128]
    public var idxGate: GPUTensor?        // F16 indexer_compressor_gate [nEmbd x coff*128]
    public var idxApe: GPUTensor?         // F16 indexer_compressor_ape  [coff*128 x ratio]
    public var idxNorm: GPUTensor?        // F32 indexer_compressor_norm [128]
    // DS4_COMP_Q8 twin flag for the indexer compressor kv/gate projections.
    public var idxCompQ8 = false
    // Hash routing + selection bias (ds4.c layer_hash_selected_experts /
    // layer_topk_selected_experts_from_probs). tid2eid is the REQUIRED token-id →
    // expert table of the first n_hash_layer (3) layers: I32 [6 x n_vocab], row =
    // token id. expBias (exp_probs_b.bias, F32 [nExperts]) shifts the probs for
    // top-k SELECTION only — the route weights stay normalized on unbiased probs.
    public var tid2eid: GPUTensor?        // I32 [6 x nVocab]
    public var tid2eidRows = 0            // n_vocab (kernel clamps token to rows-1)
    public var expBias: GPUTensor?        // F32 [nExperts]
    // Routed-expert quant PER LAYER. Mixed-precision GGUFs upcast some layers
    // (e.g. a few to Q4_K over an IQ2_XXS/Q2_K base via --tensor-type); the decode
    // kernels dispatch on THESE, not on the model-global DSV4Dims quant. Detected
    // from the actual tensor types in GGUFWeights.layer; default = the common Q4_K.
    public var gateQuant: MoEQuant = .q4_K
    public var upQuant: MoEQuant = .q4_K
    public var downQuant: MoEQuant = .q4_K
    // DS4_DENSE_Q4: the giant q_b / output_a / output_b projections were
    // requantized Q8_0 → Q4_K at load (resident); the dispatch reads THESE flags.
    public var qBQ4 = false
    public var attnOutQ4 = false
    public var attnOutAQ4 = false
    // DS4_SHARED_Q4 (requires DS4_DENSE_Q4): the shared-expert FFN projections
    // requantized Q8_0 → Q4_K at load (resident) — their Q8 slabs leave the
    // per-token dense stream entirely. Per-projection flags like the attn trio.
    public var sharedGateQ4 = false
    public var sharedUpQ4 = false
    public var sharedDownQ4 = false
    // DS4_QKV_Q4 (requires DS4_DENSE_Q4): the remaining mid-size attention
    // projections (q_a, kv — ~16 MB/layer of Q8 still streamed after the big
    // trio went resident) requantized Q8_0 → Q4_K at load (resident). Lossy
    // like the others; A/B before adopting as default.
    public var qAQ4 = false
    public var kvQ4 = false
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
