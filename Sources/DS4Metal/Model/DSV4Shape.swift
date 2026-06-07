import Foundation

// Stage C: DeepSeek-V4 Flash model shape constants (from ds4.c DS4_SHAPE_FLASH)
// and the GGUF tensor naming scheme (from ds4.c load_layer / load_output_head).
// Used to build LayerWeights/OutputHeadWeights from a GGUF file.

public enum DSV4Shape {
    // Flash architecture constants.
    public static let nLayer = 43
    public static let nEmbd = 4096
    public static let nVocab = 129280
    public static let nHead = 64
    public static let nHeadKV = 1
    public static let nHeadDim = 512
    public static let nRot = 64
    public static let nOutGroup = 8
    public static let nLoraQ = 1024          // q_rank
    public static let nLoraO = 1024          // attention output low-rank
    public static let nExpert = 256
    public static let nExpertUsed = 6        // k
    public static let nExpertShared = 1
    public static let nFFExp = 2048          // expert ffn dim
    public static let nHC = 4
    public static let nHCSinkhornIter = 20   // NOTE: 20, not 1
    public static let mixHc = 2 * 4 + 4 * 4  // 24
    public static var qDim: Int { nHead * nHeadDim }   // 64*512 = 32768
    public static var hcDim: Int { nHC * nEmbd }       // 16384

    // --- Compression + sparse-indexer (ds4.c DS4_SHAPE_FLASH) ---
    // Sliding-window raw KV cap on compressed layers: older tokens are pooled into
    // compressed rows, so the per-token raw cache only keeps the most recent nSWA.
    public static let nSWA = 128
    public static let nIndexerHead = 64
    public static let nIndexerHeadDim = 128
    public static let nIndexerTopK = 512

    /// Compressor lane factor: ratio-4 layers keep two overlapping pooling lanes
    /// (so width doubles); ratio-128 keeps one. ds4.c: coff = ratio==4 ? 2 : 1.
    public static func coff(ratio: Int) -> Int { ratio == 4 ? 2 : 1 }
    /// Attention compressor projection width (coff * headDim).
    public static func attnCompWidth(ratio: Int) -> Int { coff(ratio: ratio) * nHeadDim }
    /// Indexer compressor projection width (coff * nIndexerHeadDim).
    public static func indexCompWidth(ratio: Int) -> Int { coff(ratio: ratio) * nIndexerHeadDim }
    /// Indexer query width (nIndexerHead * nIndexerHeadDim = 64*128 = 8192).
    public static var indexerQDim: Int { nIndexerHead * nIndexerHeadDim }
    /// Max emitted compressed rows for a layer at a given context size (ds4.c:
    /// comp_cap = ctx_size/4 + 2). One emit per `ratio` tokens, so ratio-4 is the
    /// densest; this caps that worst case.
    public static func compCap(ctxSize: Int) -> Int { ctxSize / 4 + 2 }
    /// A compressed row is emitted when (pos+1) % ratio == 0 (state updates every
    /// token regardless). ds4.c `should_compress`/`emit`.
    public static func emits(pos: Int, ratio: Int) -> Bool { ratio != 0 && ((pos + 1) % ratio) == 0 }
    /// Position fed to RoPE for an emitted compressed row: the window start.
    public static func compRopePos(pos: Int, ratio: Int) -> Int { pos + 1 - ratio }

    // RoPE constants (ds4.c DS4_SHAPE_FLASH). Compressed layers (ratio != 0) use a
    // different freq_base + YaRN; layers 0,1 (ratio 0) use the plain base.
    public static let ropeFreqBase: Float = 10000
    public static let compressRopeFreqBase: Float = 160000
    public static let ropeScaleFactor: Float = 16
    public static let ropeOrigCtx = 65536
    public static let ropeBetaFast: Float = 32
    public static let ropeBetaSlow: Float = 1

    /// Per-layer compression ratio (Flash): layers 0,1 -> 0; even>=2 -> 4; odd>=3 -> 128.
    public static func compressRatio(layer il: Int) -> Int {
        if il < 2 { return 0 }
        return (il & 1) == 0 ? 4 : 128
    }

    /// Per-layer RoPE params. Compressed layers (ratio != 0) use compress_rope_freq_base
    /// (160000) + YaRN (ext_factor=1, freq_scale=1/16, n_ctx_orig=65536); others plain.
    public static func ropeParams(layer il: Int) -> RopeParams {
        if compressRatio(layer: il) != 0 {
            let freqScale = 1.0 / ropeScaleFactor
            let attnFactor = 1.0 / (1.0 + 0.1 * Foundation.log(1.0 / freqScale))
            return RopeParams(nCtxOrig: ropeOrigCtx, freqBase: compressRopeFreqBase, freqScale: freqScale,
                              extFactor: 1, attnFactor: attnFactor, betaFast: ropeBetaFast, betaSlow: ropeBetaSlow)
        }
        return RopeParams(nCtxOrig: 0, freqBase: ropeFreqBase, freqScale: 1, extFactor: 0,
                          attnFactor: 1, betaFast: ropeBetaFast, betaSlow: ropeBetaSlow)
    }

    public static var dims: DSV4Dims {
        DSV4Dims(nEmbd: nEmbd, nHC: nHC, headDim: nHeadDim, nHead: nHead, qRank: nLoraQ, qDim: qDim,
                 sharedFfn: nFFExp, nExperts: nExpert, expertFfn: nFFExp, k: nExpertUsed, nRot: nRot, vocab: nVocab)
    }

    /// Per-layer tensor names (blk.<il>.*) required for the dense decode path
    /// (compression/indexer tensors omitted — ratio==0 first faithful version).
    public static func layerTensorNames(_ il: Int) -> [String] {
        let p = "blk.\(il)."
        return [
            p + "hc_attn_fn.weight", p + "hc_attn_scale.weight", p + "hc_attn_base.weight",
            p + "attn_norm.weight", p + "attn_q_a.weight", p + "attn_q_a_norm.weight",
            p + "attn_q_b.weight", p + "attn_kv.weight", p + "attn_kv_a_norm.weight",
            p + "attn_sinks.weight", p + "attn_output_a.weight", p + "attn_output_b.weight",
            p + "hc_ffn_fn.weight", p + "hc_ffn_scale.weight", p + "hc_ffn_base.weight",
            p + "ffn_norm.weight", p + "ffn_gate_inp.weight",
            p + "ffn_gate_exps.weight", p + "ffn_up_exps.weight", p + "ffn_down_exps.weight",
            p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight", p + "ffn_down_shexp.weight",
        ]
    }

    public static let outputTensorNames = [
        "token_embd.weight", "output_norm.weight", "output.weight",
        "output_hc_fn.weight", "output_hc_scale.weight", "output_hc_base.weight",
    ]
}
