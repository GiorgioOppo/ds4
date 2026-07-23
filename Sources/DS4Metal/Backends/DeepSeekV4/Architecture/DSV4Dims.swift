import Foundation
import Metal

public struct DSV4Dims: Sendable {
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
    /// Use the fused HC-reduce tail (split+collapse+RMSNorm in one dispatch,
    /// kernel_dsv4_hc_split_weighted_sum_norm4 — the C decode release path)
    /// instead of the three separate dispatches. Runs twice per layer, so this
    /// saves ~170 dispatches/token. Same math; only the RMSNorm reduction
    /// order differs (±1 ulp class). DS4_FUSED_HC=0 restores the unfused path.
    public var fusedHC: Bool = ProcessInfo.processInfo.environment["DS4_FUSED_HC"] != "0"
    /// Dedicated single-matrix Q4_K kernel for resident dense projections.
    /// `=0` restores the historical k=1 MoE-id wrapper for A/B parity.
    public var denseQ4Kernel: Bool = ProcessInfo.processInfo.environment["DS4_DENSE_Q4_KERNEL"] != "0"
    /// One vector kernel for router softplus+sqrt instead of two scalar passes.
    /// `=0` restores the historical two-dispatch graph.
    public var fusedRouterProbs: Bool = ProcessInfo.processInfo.environment["DS4_FUSED_ROUTER_PROBS"] != "0"
    /// DS4_INDEXED_ATTN=1 (opt-in): quando l'indexer DSA è attivo, attention
    /// SOLO sulle topK righe compresse selezionate (kernel indicizzato rb16)
    /// invece di maschera + flash sull'intero span — costo per token
    /// O(nSWA + topK) costante col contesto. `assente/0` = percorso storico.
    public var indexedAttn: Bool = ProcessInfo.processInfo.environment["DS4_INDEXED_ATTN"] == "1"
    /// Fuse top-6 selection and its bit-identical six-term weight normalization.
    /// `=0` restores the historical second router-weight dispatch.
    public var fusedRouterFinalize: Bool = ProcessInfo.processInfo.environment["DS4_FUSED_ROUTER_FINALIZE"] != "0"
    /// Raw-KV sliding window (C: DS4_N_SWA, GGUF `attention.sliding_window` = 128).
    /// Attention sees only the LAST nSWA raw rows; older context is visible only
    /// through the NSA-compressed rows — matching the trained NSA semantics (the C
    /// kv_cache_push_raw literally slides its raw buffer at this cap).
    public var nSWA: Int = 128
    /// NSA indexer (DSA): on ratio-4 layers, when the compressed rows exceed
    /// `indexerTopK` the attention sees only the top-K most relevant ones for the
    /// current query (C: DS4_N_INDEXER_TOP_K=512 on Flash, 64 heads × 128 dim).
    public var nIndexerHead: Int = 64
    public var nIndexerHeadDim: Int = 128
    public var indexerTopK: Int = 512
    /// Training-time router scale. Flash uses 1.5; Pro uses 2.5.
    public var expertWeightScale: Float = 1.5
    /// Initial layers whose expert id is provided by the hash router tensor.
    public var nHashLayers: Int = 3
    /// Sinkhorn normalization iterations used by the hyper-connection mixer.
    public var sinkhornIterations: Int = 20
    /// Per-group slice of the attention heads (qDim / nOutGroup).
    public var attnGroupDim: Int { qDim / nOutGroup }
    /// Low-rank attention output dim (nOutGroup * nLoraO).
    public var attnLowDim: Int { nOutGroup * nLoraO }
    public init(nEmbd: Int, nHC: Int, headDim: Int, nHead: Int, qRank: Int, qDim: Int,
                sharedFfn: Int, nExperts: Int, expertFfn: Int, k: Int, nRot: Int, vocab: Int,
                nOutGroup: Int = 8, nLoraO: Int = 1024, swigluClamp: Float = 10.0,
                expertWeightScale: Float = 1.5, nHashLayers: Int = 3,
                sinkhornIterations: Int = 20) {
        self.nEmbd = nEmbd; self.nHC = nHC; self.headDim = headDim; self.nHead = nHead
        self.qRank = qRank; self.qDim = qDim; self.sharedFfn = sharedFfn
        self.nExperts = nExperts; self.expertFfn = expertFfn; self.k = k; self.nRot = nRot; self.vocab = vocab
        self.nOutGroup = nOutGroup; self.nLoraO = nLoraO; self.swigluClamp = swigluClamp
        self.expertWeightScale = expertWeightScale; self.nHashLayers = nHashLayers
        self.sinkhornIterations = sinkhornIterations
    }
}
