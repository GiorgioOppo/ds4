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
    /// YaRN mscale factor. Combined with `ropeFactor > 1` produces the
    /// attention-softmax rescaling `mscale = 0.1 * mscale * log(factor) + 1`
    /// that compensates the entropy growth from extending the context
    /// past the training length. Used as `softmax_scale *= mscale²`.
    /// Default 1.0 matches the DeepSeek-V3 / V4 reference. Reads
    /// `mscale` or `rope_scaling.mscale` from config.json.
    public var mscale: Float = 1.0

    // Indexer
    public var indexNHeads: Int = 64
    public var indexHeadDim: Int = 128
    public var indexTopk: Int = 512

    // Hyper-Connections
    public var hcMult: Int = 4
    public var hcSinkhornIters: Int = 20
    public var hcEps: Float = 1e-6

    /// Per-layer set of routed-expert indices that were dropped by
    /// the expert pruner (`DeepSeekVocabPruner/ExpertRewriter.swift`).
    /// Outer index = main layer id (0..<nLayers); each inner array
    /// lists the expert ids in that layer whose weights are intentionally
    /// absent from the safetensors. The loader (`Assembly.swift`) skips
    /// allocation for these slots and stores `nil` in `MoEFFN.experts`,
    /// so the dispatch path naturally bypasses them. The gate weight
    /// rows for these experts are set to large-negative by the rewriter
    /// so the kernel never picks them in top-K.
    ///
    /// Empty array per layer (or absent in config.json) means no
    /// pruning — the standard path. The field is parsed from
    /// `pruned_experts` in config.json; serializes as a list of lists.
    public var prunedExperts: [[Int]] = []

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
        case mscale = "mscale"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopk = "index_topk"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case prunedExperts = "pruned_experts"
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
        self.mscale             = g(.mscale, 1.0)
        self.indexNHeads        = g(.indexNHeads, 64)
        self.indexHeadDim       = g(.indexHeadDim, 128)
        self.indexTopk          = g(.indexTopk, 512)
        self.hcMult             = g(.hcMult, 4)
        self.hcSinkhornIters    = g(.hcSinkhornIters, 20)
        self.hcEps              = g(.hcEps, 1e-6)
        self.prunedExperts      = g(.prunedExperts, [])
    }

    /// Loads a config.json. Accepts both the Swift-port snake_case names
    /// (used by Reference/inference/model.py and tests in this repo) and
    /// the HuggingFace `transformers` field names (used by the upstream
    /// model card's config.json). For HF, also flattens nested
    /// `rope_scaling.{factor,original_max_position_embeddings,beta_fast,beta_slow}`.
    public static func load(from url: URL) throws -> ModelConfig {
        let data = try Data(contentsOf: url)
        guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "ModelConfig", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "config.json is not a JSON object"])
        }
        return ModelConfig(fromDict: dict)
    }

    /// Initializes from a parsed JSON dictionary, accepting multiple
    /// key aliases per field (Swift-port name + HF name).
    public init(fromDict dict: [String: Any]) {
        // Flatten HF's nested rope_scaling so the lookup helpers below
        // can treat its members like top-level keys.
        var flat = dict
        if let rs = dict["rope_scaling"] as? [String: Any] {
            for (k, v) in rs { flat["rope_scaling.\(k)"] = v }
        }

        func intOf(_ any: Any) -> Int? {
            if let v = any as? Int { return v }
            if let v = any as? Double { return Int(v) }
            if let v = any as? NSNumber { return v.intValue }
            return nil
        }
        func floatOf(_ any: Any) -> Float? {
            if let v = any as? Double { return Float(v) }
            if let v = any as? Float  { return v }
            if let v = any as? Int    { return Float(v) }
            if let v = any as? NSNumber { return v.floatValue }
            return nil
        }
        func getInt(_ keys: [String], _ fallback: Int) -> Int {
            for k in keys { if let v = flat[k], let i = intOf(v) { return i } }
            return fallback
        }
        func getFloat(_ keys: [String], _ fallback: Float) -> Float {
            for k in keys { if let v = flat[k], let f = floatOf(v) { return f } }
            return fallback
        }
        func getString(_ keys: [String], _ fallback: String) -> String {
            for k in keys { if let v = flat[k] as? String { return v } }
            return fallback
        }
        func getOptString(_ keys: [String]) -> String? {
            for k in keys {
                if let v = flat[k] as? String { return v }
                if flat[k] is NSNull { return nil }
            }
            return nil
        }
        func getIntArray(_ keys: [String], _ fallback: [Int]) -> [Int] {
            for k in keys {
                if let arr = flat[k] as? [Any] {
                    return arr.compactMap(intOf)
                }
            }
            return fallback
        }

        self.maxBatchSize       = getInt(["max_batch_size"], 4)
        self.maxSeqLen          = getInt(["max_seq_len", "max_position_embeddings"], 4096)
        self.dtype              = getString(["dtype", "torch_dtype"], "fp8")
        self.scaleFmt           = getOptString(["scale_fmt"]) ?? "ue8m0"
        self.expertDtype        = getOptString(["expert_dtype"])
        self.scaleDtype         = getString(["scale_dtype"], "fp8")
        self.vocabSize          = getInt(["vocab_size"], 129280)
        self.dim                = getInt(["dim", "hidden_size"], 4096)
        self.moeInterDim        = getInt(["moe_inter_dim", "moe_intermediate_size"], 4096)
        self.nLayers            = getInt(["n_layers", "num_hidden_layers"], 7)
        self.nHashLayers        = getInt(["n_hash_layers", "num_hash_layers"], 0)
        self.nMtpLayers         = getInt(["n_mtp_layers", "num_nextn_predict_layers"], 1)
        self.nHeads             = getInt(["n_heads", "num_attention_heads"], 64)
        self.nRoutedExperts     = getInt(["n_routed_experts"], 8)
        self.nSharedExperts     = getInt(["n_shared_experts"], 1)
        self.nActivatedExperts  = getInt(["n_activated_experts", "num_experts_per_tok"], 2)
        self.scoreFunc          = getString(["score_func", "scoring_func"], "sqrtsoftplus")
        self.routeScale         = getFloat(["route_scale", "routed_scaling_factor"], 1.0)
        self.swigluLimit        = getFloat(["swiglu_limit"], 0.0)
        self.qLoraRank          = getInt(["q_lora_rank"], 1024)
        self.headDim            = getInt(["head_dim"], 512)
        self.ropeHeadDim        = getInt(["rope_head_dim", "qk_rope_head_dim"], 64)
        self.normEps            = getFloat(["norm_eps", "rms_norm_eps"], 1e-6)
        self.oGroups            = getInt(["o_groups"], 8)
        self.oLoraRank          = getInt(["o_lora_rank"], 1024)
        self.windowSize         = getInt(["window_size", "sliding_window"], 128)
        self.compressRatios     = getIntArray(["compress_ratios"], [0, 0, 4, 128, 4, 128, 4, 0])
        self.compressRopeTheta  = getFloat(["compress_rope_theta"], 40000.0)
        self.originalSeqLen     = getInt(["original_seq_len", "rope_scaling.original_max_position_embeddings"], 0)
        self.ropeTheta          = getFloat(["rope_theta"], 10000.0)
        self.ropeFactor         = getFloat(["rope_factor", "rope_scaling.factor"], 40)
        self.betaFast           = getInt(["beta_fast", "rope_scaling.beta_fast"], 32)
        self.betaSlow           = getInt(["beta_slow", "rope_scaling.beta_slow"], 1)
        self.mscale             = getFloat(["mscale", "rope_scaling.mscale"], 1.0)
        self.indexNHeads        = getInt(["index_n_heads"], 64)
        self.indexHeadDim       = getInt(["index_head_dim"], 128)
        self.indexTopk          = getInt(["index_topk"], 512)
        self.hcMult             = getInt(["hc_mult"], 4)
        self.hcSinkhornIters    = getInt(["hc_sinkhorn_iters"], 20)
        self.hcEps              = getFloat(["hc_eps"], 1e-6)
        // pruned_experts: per-layer list of dropped expert ids.
        // Tolerate both [[int]] and missing field (empty default).
        if let outer = flat["pruned_experts"] as? [Any] {
            var rows: [[Int]] = []
            for row in outer {
                if let inner = row as? [Any] {
                    rows.append(inner.compactMap(intOf))
                } else {
                    rows.append([])
                }
            }
            self.prunedExperts = rows
        } else {
            self.prunedExperts = []
        }

        // Per-token active-expert count. Defaults to 8 on explicit
        // request. This may differ from the checkpoint's trained value
        // — fewer experts is cheaper (FFN dispatch cost scales
        // ~linearly with the count), more is costlier; either way it
        // is off-distribution, so this is an A/B knob.
        // `DEEPSEEK_TOPK_EXPERTS=N` overrides the default. Clamped to
        // the experts that actually exist and to the gate kernel's
        // 16-slot limit (moe.metal `bestV[16]`). Applies only to
        // learned-routing layers — the first `nHashLayers` hash-routed
        // layers read a fixed-shape `tid2eid` table whose K was set at
        // training time, and `Gate.init` snaps them back to it
        // regardless of this value.
        let requestedActiveExperts: Int = {
            if let raw = ProcessInfo.processInfo
                .environment["DEEPSEEK_TOPK_EXPERTS"],
               let n = Int(raw), n > 0 { return n }
            return 8
        }()
        let resolvedActiveExperts = min(requestedActiveExperts,
                                         max(self.nRoutedExperts, 1), 16)
        if resolvedActiveExperts != self.nActivatedExperts {
            let oldK = self.nActivatedExperts
            self.nActivatedExperts = resolvedActiveExperts
            let line = "[config] active experts/token: \(oldK) → " +
                       "\(resolvedActiveExperts) (first " +
                       "\(self.nHashLayers) hash-routed layer(s) keep " +
                       "their trained K)\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
    }

    /// Per-layer effective head dimensions.
    public var nopeHeadDim: Int { headDim - ropeHeadDim }

    /// Coarse upper bound on the total bytes of attention KV cache
    /// the model will allocate at load time (per Assembly.swift's
    /// `Tensor.empty(...)` calls for each layer's kvCache + the
    /// indexer/compressor caches). Streaming the weights doesn't
    /// help these — they're real `storageModeShared` MTLBuffers
    /// the GPU writes to during forward, so they have to stay
    /// resident at full size.
    ///
    /// Formula (rough): for each layer with compress ratio R,
    ///   maxBatchSize × (windowSize + maxSeqLen / max(R, 1)) × headDim × 4
    /// plus a comparable indexer cache on layers where R == 4.
    /// MTP layers add another batch's worth.
    public var projectedKVCacheBytes: UInt64 {
        var total: UInt64 = 0
        let perRow = UInt64(maxBatchSize) * UInt64(headDim) * 4   // f32 bytes
        let win = UInt64(windowSize)
        let seq = UInt64(maxSeqLen)

        for i in 0..<min(nLayers, compressRatios.count) {
            let r = compressRatios[i]
            let cacheRows = r > 0
                ? win &+ (seq / UInt64(r))
                : win
            // Attention KV cache.
            total &+= cacheRows &* perRow
            // Indexer KV cache lives on ratio==4 layers and is
            // roughly the same size.
            if r == 4 {
                total &+= cacheRows &* perRow
            }
            // Compressor `kvState` + `cacheView` (alias of kvState,
            // but I count once; cacheView shares storage).
            let coff = UInt64(hcMult)
            if r > 0 {
                total &+= coff &* UInt64(r) &* UInt64(maxBatchSize)
                    &* UInt64(headDim) &* 4
            }
        }
        // MTP layers reuse the same shapes; ~1 batch each.
        for j in 0..<nMtpLayers {
            let idx = nLayers + j
            guard idx < compressRatios.count else { break }
            let r = compressRatios[idx]
            let cacheRows = r > 0 ? win &+ (seq / UInt64(r)) : win
            total &+= cacheRows &* perRow
        }
        return total
    }

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

        // n_layers: compress_ratios covers main + MTP layers (model.py:785-791
        // appends MTPBlock at indices [n_layers, n_layers + n_mtp_layers)
        // and they index into compress_ratios). Only override when the
        // config is internally inconsistent.
        let inferredNLayers = c.compressRatios.count - c.nMtpLayers
        if inferredNLayers > 0 && inferredNLayers != c.nLayers {
            notes.append("n_layers: \(c.nLayers) → \(inferredNLayers) (from compress_ratios.count - n_mtp_layers)")
            c.nLayers = inferredNLayers
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

        // Sliding-window size — how many of the most recent tokens are
        // kept raw / uncompressed. Multiplied by `DEEPSEEK_WINDOW_MULT`
        // (default 16, so the raw window is 16× the checkpoint's value)
        // on explicit request, to keep more recent context exact.
        //
        // WARNING: out-of-distribution. The DSA sliding-window
        // attention was trained with the checkpoint's window
        // (typically 128); a wider window changes the softmax
        // distribution the trained weights see and usually degrades
        // output. It also enlarges every layer's KV cache and the
        // sparse-attention cost (SparseAttention tiles down to
        // compensate for the watchdog). `DEEPSEEK_WINDOW_MULT=1`
        // restores the trained value. Applied before the maxSeqLen
        // auto-grow below so the wider window is counted in the budget.
        let windowMult: Int = {
            if let raw = ProcessInfo.processInfo
                .environment["DEEPSEEK_WINDOW_MULT"],
               let n = Int(raw), n > 0 { return n }
            return 16
        }()
        if windowMult > 1 {
            let oldWin = c.windowSize
            c.windowSize = oldWin * windowMult
            FileHandle.standardError.write(Data(
                ("[config] windowSize ×\(windowMult): \(oldWin) → "
                 + "\(c.windowSize) — raw window widened "
                 + "(out-of-distribution; DEEPSEEK_WINDOW_MULT=1 to "
                 + "restore the trained value)\n").utf8))
        }

        // maxSeqLen resolution — it sizes the RoPE table and every
        // per-layer KV cache; per-layer compression keeps the cache
        // sub-linear (ratio-R layers grow at N/R, ratio-0 layers stay
        // at `windowSize`).
        //
        // `DEEPSEEK_MAX_SEQ_LEN=N` pins it explicitly. Otherwise it is
        // auto-grown to the largest value whose projected KV cache
        // (`projectedKVCacheBytes`) still fits ~50% of the process
        // memory budget — more retained context on machines that can
        // afford it, without ever risking the `kvCacheTooLarge` abort
        // at load. Only raised, never lowered, hard-capped at 128K.
        //
        // YaRN RoPE scaling keys off `originalSeqLen`, not this value,
        // so changing maxSeqLen only resizes buffers — it does not
        // re-tune positional scaling.
        if let raw = ProcessInfo.processInfo
            .environment["DEEPSEEK_MAX_SEQ_LEN"],
           let pinned = Int(raw), pinned > 0 {
            if pinned != c.maxSeqLen {
                let oldLen = c.maxSeqLen
                c.maxSeqLen = pinned
                FileHandle.standardError.write(Data(
                    ("[config] DEEPSEEK_MAX_SEQ_LEN override: maxSeqLen "
                     + "\(oldLen) → \(pinned)\n").utf8))
            }
        } else {
            let budget = SystemProbe.effectiveProcessBudget()
            if budget > 0 {
                let target = UInt64(Double(budget) * 0.5)
                let hardCap = 131_072
                func fits(_ length: Int) -> Bool {
                    var probe = c
                    probe.maxSeqLen = length
                    return probe.projectedKVCacheBytes <= target
                }
                if c.maxSeqLen < hardCap, fits(c.maxSeqLen) {
                    var lo = c.maxSeqLen
                    var hi = hardCap
                    while lo < hi {
                        let mid = lo + (hi - lo + 1) / 2
                        if fits(mid) { lo = mid } else { hi = mid - 1 }
                    }
                    if lo > c.maxSeqLen {
                        let oldLen = c.maxSeqLen
                        c.maxSeqLen = lo
                        FileHandle.standardError.write(Data(
                            ("[config] maxSeqLen auto-grown \(oldLen) → "
                             + "\(lo) (KV cache ≈ 50% of memory budget; "
                             + "set DEEPSEEK_MAX_SEQ_LEN to override)\n").utf8))
                    }
                }
            }
        }

        return c
    }

    /// Multi-line human-readable dump of every config field. For
    /// `--print-config` diagnostics: lets the caller diff the loaded
    /// values against config.json field-by-field to catch keys that
    /// silently fell back to the hard-coded defaults.
    public var summary: String {
        var s = ""
        s += "ModelConfig {\n"
        s += "  // shape\n"
        s += "  vocabSize=\(vocabSize) dim=\(dim) maxSeqLen=\(maxSeqLen) maxBatchSize=\(maxBatchSize)\n"
        s += "  nLayers=\(nLayers) nHashLayers=\(nHashLayers) nMtpLayers=\(nMtpLayers)\n"
        s += "  nHeads=\(nHeads) headDim=\(headDim) ropeHeadDim=\(ropeHeadDim)\n"
        s += "  qLoraRank=\(qLoraRank) oLoraRank=\(oLoraRank) oGroups=\(oGroups)\n"
        s += "  windowSize=\(windowSize)\n"
        s += "  // MoE\n"
        s += "  nRoutedExperts=\(nRoutedExperts) nSharedExperts=\(nSharedExperts) nActivatedExperts=\(nActivatedExperts)\n"
        s += "  moeInterDim=\(moeInterDim) routeScale=\(routeScale) scoreFunc=\"\(scoreFunc)\" swigluLimit=\(swigluLimit)\n"
        s += "  // dtype + scale\n"
        s += "  dtype=\"\(dtype)\" scaleDtype=\"\(scaleDtype)\" "
        s += "expertDtype=\(expertDtype.map { "\"\($0)\"" } ?? "nil") "
        s += "scaleFmt=\(scaleFmt.map { "\"\($0)\"" } ?? "nil")\n"
        s += "  // RoPE / YaRN\n"
        s += "  ropeTheta=\(ropeTheta) compressRopeTheta=\(compressRopeTheta)\n"
        s += "  ropeFactor=\(ropeFactor) originalSeqLen=\(originalSeqLen)\n"
        s += "  betaFast=\(betaFast) betaSlow=\(betaSlow) mscale=\(mscale)\n"
        s += "  // per-layer compress_ratios (count=\(compressRatios.count))\n"
        s += "  compressRatios=\(compressRatios)\n"
        s += "  // norms\n"
        s += "  normEps=\(normEps)\n"
        s += "  // indexer\n"
        s += "  indexNHeads=\(indexNHeads) indexHeadDim=\(indexHeadDim) indexTopk=\(indexTopk)\n"
        s += "  // hyper-connections\n"
        s += "  hcMult=\(hcMult) hcSinkhornIters=\(hcSinkhornIters) hcEps=\(hcEps)\n"
        if !prunedExperts.isEmpty {
            let totalDropped = prunedExperts.reduce(0) { $0 + $1.count }
            s += "  // expert pruning\n"
            s += "  prunedExperts: \(totalDropped) total across "
            s += "\(prunedExperts.count) layers "
            s += "(per-layer counts=\(prunedExperts.map { $0.count }))\n"
        }
        s += "}\n"
        return s
    }

    /// LCM dei `compressRatios` non-zero. Usato dal KV-cache rewind:
    /// per rewindare safely a una position `P` su tutti i layer, `P`
    /// deve essere multiplo di questo LCM (così tutti i compressor
    /// — ratio=4, ratio=128, ecc. — trovano un inizio-window valido).
    ///
    /// Layer con ratio=0 (pure sliding window, no compressor) non
    /// vincolano il LCM perché non hanno rolling state da rewindare.
    /// Se TUTTI i layer hanno ratio=0, ritorna 1 (qualsiasi P è
    /// valida).
    ///
    /// Per V4 con ratios `[0, 0, 4, 128, 4, 128, 4, 0]` → LCM = 128.
    public var compressRatioLCM: Int {
        let nonZero = Set(compressRatios.filter { $0 > 0 })
        if nonZero.isEmpty { return 1 }
        return nonZero.reduce(1) { Self.lcm($0, $1) }
    }

    @inline(__always)
    private static func gcd(_ a: Int, _ b: Int) -> Int {
        var (x, y) = (abs(a), abs(b))
        while y != 0 { (x, y) = (y, x % y) }
        return x
    }

    @inline(__always)
    private static func lcm(_ a: Int, _ b: Int) -> Int {
        if a == 0 || b == 0 { return 0 }
        return a / gcd(a, b) * b
    }
}
