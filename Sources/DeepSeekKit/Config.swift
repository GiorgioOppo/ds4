import Foundation

/// Mirrors the `ModelArgs` dataclass from `Reference/inference/model.py`.
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

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func g<T: Decodable>(_ k: CodingKeys, _ fallback: T) -> T {
            (try? c.decodeIfPresent(T.self, forKey: k)) ?? fallback
        }
        self.maxBatchSize       = g(.maxBatchSize, 4)
        self.maxSeqLen          = g(.maxSeqLen, 4096)
        self.dtype              = g(.dtype, "fp8")
        self.scaleFmt           = g(.scaleFmt, "ue8m0")
        self.expertDtype        = g(.expertDtype, nil)
        self.scaleDtype         = g(.scaleDtype, "fp8")
        self.vocabSize          = g(.vocabSize, 129280)
        self.dim                = g(.dim, 4096)
        self.moeInterDim        = g(.moeInterDim, 4096)
        self.nLayers            = g(.nLayers, 7)
        self.nHashLayers        = g(.nHashLayers, 0)
        self.nMtpLayers         = g(.nMtpLayers, 1)
        self.nHeads             = g(.nHeads, 64)
        self.nRoutedExperts     = g(.nRoutedExperts, 8)
        self.nSharedExperts     = g(.nSharedExperts, 1)
        self.nActivatedExperts  = g(.nActivatedExperts, 2)
        self.scoreFunc          = g(.scoreFunc, "sqrtsoftplus")
        self.routeScale         = g(.routeScale, 1.0)
        self.swigluLimit        = g(.swigluLimit, 0.0)
        self.qLoraRank          = g(.qLoraRank, 1024)
        self.headDim            = g(.headDim, 512)
        self.ropeHeadDim        = g(.ropeHeadDim, 64)
        self.normEps            = g(.normEps, 1e-6)
        self.oGroups            = g(.oGroups, 8)
        self.oLoraRank          = g(.oLoraRank, 1024)
        self.windowSize         = g(.windowSize, 128)
        self.compressRatios     = g(.compressRatios, [0, 0, 4, 128, 4, 128, 4, 0])
        self.compressRopeTheta  = g(.compressRopeTheta, 40000.0)
        self.originalSeqLen     = g(.originalSeqLen, 0)
        self.ropeTheta          = g(.ropeTheta, 10000.0)
        self.ropeFactor         = g(.ropeFactor, 40)
        self.betaFast           = g(.betaFast, 32)
        self.betaSlow           = g(.betaSlow, 1)
        self.indexNHeads        = g(.indexNHeads, 64)
        self.indexHeadDim       = g(.indexHeadDim, 128)
        self.indexTopk          = g(.indexTopk, 512)
        self.hcMult             = g(.hcMult, 4)
        self.hcSinkhornIters    = g(.hcSinkhornIters, 20)
        self.hcEps              = g(.hcEps, 1e-6)
    }

    public static func load(from url: URL) throws -> ModelConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(ModelConfig.self, from: data)
    }

    /// Per-layer effective head dimensions.
    public var nopeHeadDim: Int { headDim - ropeHeadDim }

    /// Returns a copy of `self` with key dimensions overridden to match
    /// the actual tensor shapes in `loader`. Use this to recover from a
    /// `config.json` that's missing fields (e.g. only `head_dim` and
    /// `compress_ratios` set, leaving `dim`, `n_heads`, `n_layers`, etc.
    /// at toy defaults that don't match the converted checkpoint).
    ///
    /// Each override is reported on stderr so the user can see what was
    /// inferred. Fields that can't be inferred from the checkpoint
    /// (max_batch_size, max_seq_len, hcMult, etc.) are left as-is.
    public func inferred(from loader: WeightLoader) -> ModelConfig {
        var c = self
        var notes: [String] = []

        // n_layers: trust compress_ratios.count if it differs from current.
        if c.compressRatios.count != c.nLayers {
            notes.append("n_layers: \(c.nLayers) → \(c.compressRatios.count) (from compress_ratios.count)")
            c.nLayers = c.compressRatios.count
        }

        // vocab_size + dim: from embed.weight shape [vocab, dim].
        if let s = loader.shape(ofAny: ["embed.weight", "model.embed.weight"]),
           s.count == 2 {
            if s[0] != c.vocabSize {
                notes.append("vocab_size: \(c.vocabSize) → \(s[0]) (from embed.weight)")
                c.vocabSize = s[0]
            }
            if s[1] != c.dim {
                notes.append("dim: \(c.dim) → \(s[1]) (from embed.weight)")
                c.dim = s[1]
            }
        }

        // q_lora_rank + n_heads: from layers.0.attn.wq_b.weight shape
        // [n_heads * head_dim, q_lora_rank].
        if let s = loader.shape(of: "layers.0.attn.wq_b.weight"), s.count == 2 {
            let inferredQLora = s[1]
            if inferredQLora != c.qLoraRank {
                notes.append("q_lora_rank: \(c.qLoraRank) → \(inferredQLora) (from wq_b)")
                c.qLoraRank = inferredQLora
            }
            if c.headDim > 0 && s[0] % c.headDim == 0 {
                let inferredHeads = s[0] / c.headDim
                if inferredHeads != c.nHeads {
                    notes.append("n_heads: \(c.nHeads) → \(inferredHeads) (from wq_b / head_dim)")
                    c.nHeads = inferredHeads
                }
            }
        }

        // o_lora_rank: from layers.0.attn.wo_b.weight shape [dim, o_groups * o_lora_rank].
        if let s = loader.shape(of: "layers.0.attn.wo_b.weight"),
           s.count == 2, c.oGroups > 0, s[1] % c.oGroups == 0 {
            let inferredOLora = s[1] / c.oGroups
            if inferredOLora != c.oLoraRank {
                notes.append("o_lora_rank: \(c.oLoraRank) → \(inferredOLora) (from wo_b / o_groups)")
                c.oLoraRank = inferredOLora
            }
        }

        // moe_inter_dim: from layers.0.ffn.experts.0.w1.weight shape
        // [moe_inter_dim, dim].
        if let s = loader.shape(of: "layers.0.ffn.experts.0.w1.weight"), s.count == 2 {
            if s[0] != c.moeInterDim {
                notes.append("moe_inter_dim: \(c.moeInterDim) → \(s[0]) (from expert.w1)")
                c.moeInterDim = s[0]
            }
        }

        // index_n_heads: from indexer wq_b on the first ratio==4 layer.
        // Shape [index_n_heads * index_head_dim, q_lora_rank].
        if let firstRatio4 = c.compressRatios.firstIndex(of: 4),
           let s = loader.shape(of: "layers.\(firstRatio4).attn.indexer.wq_b.weight"),
           s.count == 2, c.indexHeadDim > 0, s[0] % c.indexHeadDim == 0 {
            let inferredIdxH = s[0] / c.indexHeadDim
            if inferredIdxH != c.indexNHeads {
                notes.append("index_n_heads: \(c.indexNHeads) → \(inferredIdxH) (from indexer.wq_b / index_head_dim)")
                c.indexNHeads = inferredIdxH
            }
        }

        if !notes.isEmpty {
            FileHandle.standardError.write(Data(("config.json was missing or stale; inferred from checkpoint:\n  "
                + notes.joined(separator: "\n  ") + "\n").utf8))
        }
        return c
    }
}
