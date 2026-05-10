import Foundation

/// Mirrors the `ModelArgs` dataclass from `Original/DeepSeek-V4-Pro/inference/model.py`.
/// Field names use the Python snake_case via CodingKeys so config.json maps directly.
public struct ModelConfig: Codable, Sendable {
    public var maxBatchSize: Int = 4
    public var maxSeqLen: Int = 4096
    public var dtype: String = "fp8"            // "bf16" | "fp8"
    public var scaleFmt: String? = "ue8m0"      // nil | "ue8m0"
    public var expertDtype: String? = nil       // nil | "fp4"
    public var scaleDtype: String = "fp8"       // "fp32" | "fp8"

    public var vocabSize: Int = 129280
    public var dim: Int = 4096
    public var moeInterDim: Int = 4096
    public var nLayers: Int = 7
    public var nHashLayers: Int = 0
    public var nMtpLayers: Int = 1
    public var nHeads: Int = 64

    // MoE
    public var nRoutedExperts: Int = 8
    public var nSharedExperts: Int = 1
    public var nActivatedExperts: Int = 2
    public var scoreFunc: String = "sqrtsoftplus"   // "softmax" | "sigmoid" | "sqrtsoftplus"
    public var routeScale: Float = 1.0
    public var swigluLimit: Float = 0.0

    // MLA + sliding window + per-layer compression
    public var qLoraRank: Int = 1024
    public var headDim: Int = 512
    public var ropeHeadDim: Int = 64
    public var normEps: Float = 1e-6
    public var oGroups: Int = 8
    public var oLoraRank: Int = 1024
    public var windowSize: Int = 128
    /// One ratio per layer. 0 = pure sliding window. 4 = compressed with Indexer.
    /// 128 = heavy compression, no Indexer.
    public var compressRatios: [Int] = [0, 0, 4, 128, 4, 128, 4, 0]

    // YaRN RoPE
    public var compressRopeTheta: Float = 40000.0
    public var originalSeqLen: Int = 0
    public var ropeTheta: Float = 10000.0
    public var ropeFactor: Float = 40
    public var betaFast: Int = 32
    public var betaSlow: Int = 1

    // Indexer
    public var indexNHeads: Int = 64
    public var indexHeadDim: Int = 128
    public var indexTopk: Int = 512

    // Hyper-Connections
    public var hcMult: Int = 4
    public var hcSinkhornIters: Int = 20
    public var hcEps: Float = 1e-6

    enum CodingKeys: String, CodingKey {
        case maxBatchSize = "max_batch_size"
        case maxSeqLen = "max_seq_len"
        case dtype, scaleFmt = "scale_fmt", expertDtype = "expert_dtype", scaleDtype = "scale_dtype"
        case vocabSize = "vocab_size"
        case dim
        case moeInterDim = "moe_inter_dim"
        case nLayers = "n_layers"
        case nHashLayers = "n_hash_layers"
        case nMtpLayers = "n_mtp_layers"
        case nHeads = "n_heads"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case nActivatedExperts = "n_activated_experts"
        case scoreFunc = "score_func"
        case routeScale = "route_scale"
        case swigluLimit = "swiglu_limit"
        case qLoraRank = "q_lora_rank"
        case headDim = "head_dim"
        case ropeHeadDim = "rope_head_dim"
        case normEps = "norm_eps"
        case oGroups = "o_groups"
        case oLoraRank = "o_lora_rank"
        case windowSize = "window_size"
        case compressRatios = "compress_ratios"
        case compressRopeTheta = "compress_rope_theta"
        case originalSeqLen = "original_seq_len"
        case ropeTheta = "rope_theta"
        case ropeFactor = "rope_factor"
        case betaFast = "beta_fast"
        case betaSlow = "beta_slow"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopk = "index_topk"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
    }

    public init() {}

    public static func load(from url: URL) throws -> ModelConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        // Allow extra/missing keys: many configs may omit defaults.
        return try decoder.decode(ModelConfig.self, from: data)
    }

    /// Per-layer effective head dimensions.
    public var nopeHeadDim: Int { headDim - ropeHeadDim }
}
