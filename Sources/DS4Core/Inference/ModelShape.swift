import Foundation

// Faithful Swift port of the DeepSeek V4 model shape configuration in ds4.c:
// the Flash/Pro shape constants, shape selection from GGUF metadata, the
// per-layer compression-ratio formula, and the metadata validation that the C
// engine performs at load. Phase 4 of the C->Swift port.
//
// Validation: cross-checked against `./ds4 --inspect` (which prints the shape
// the C engine selected) and against the GGUF's own metadata arrays.

public enum ModelVariant: Int, Sendable {
    case flash = 0
    case pro = 1
}

public enum ModelConfigError: Error, CustomStringConvertible {
    case missing(String)
    case mismatch(String, expected: String, got: String)
    case unsupportedShape(String)

    public var description: String {
        switch self {
        case .missing(let k): return "required metadata key is missing: \(k)"
        case .mismatch(let k, let e, let g): return "metadata \(k): expected \(e), got \(g)"
        case .unsupportedShape(let s): return "unsupported DeepSeek4 shape: \(s)"
        }
    }
}

/// Engine-level defaults (port of the DS4_DEFAULT_* macros).
public enum ModelDefaults {
    public static let rmsEps: Float = 1.0e-6
    public static let hcEps: Float = 1.0e-6
    public static let swigluClampExp: Float = 10.0
    public static let ropeFreqBase: Float = 10000.0
    public static let ropeScaleFactor: Float = 16.0
    public static let ropeYarnBetaFast: Float = 32.0
    public static let ropeYarnBetaSlow: Float = 1.0
    public static let compressRopeFreqBase: Float = 160000.0
    public static let ropeOrigCtx: UInt64 = 65536
}

/// Architectural shape of a DeepSeek V4 model (port of `ds4_shape`).
public struct ModelShape: Sendable, Equatable {
    public let name: String
    public let variant: ModelVariant
    public let nLayer: UInt32
    public let nEmbd: UInt32
    public let nVocab: UInt32
    public let nHead: UInt32
    public let nHeadKV: UInt32
    public let nHeadDim: UInt32
    public let nValueDim: UInt32
    public let nRot: UInt32
    public let nOutGroup: UInt32
    public let nLoraQ: UInt32
    public let nLoraO: UInt32
    public let nExpert: UInt32
    public let nExpertUsed: UInt32
    public let nExpertShared: UInt32
    public let nFFExp: UInt32
    public let nHashLayer: UInt32
    public let nSWA: UInt32
    public let nIndexerHead: UInt32
    public let nIndexerHeadDim: UInt32
    public let nIndexerTopK: UInt32
    public let nHC: UInt32
    public let nHCSinkhornIter: UInt32
    public let expertWeightScale: Float

    public static let flash = ModelShape(
        name: "DeepSeek V4 Flash", variant: .flash,
        nLayer: 43, nEmbd: 4096, nVocab: 129280, nHead: 64, nHeadKV: 1,
        nHeadDim: 512, nValueDim: 512, nRot: 64, nOutGroup: 8, nLoraQ: 1024,
        nLoraO: 1024, nExpert: 256, nExpertUsed: 6, nExpertShared: 1, nFFExp: 2048,
        nHashLayer: 3, nSWA: 128, nIndexerHead: 64, nIndexerHeadDim: 128,
        nIndexerTopK: 512, nHC: 4, nHCSinkhornIter: 20, expertWeightScale: 1.5)

    public static let pro = ModelShape(
        name: "DeepSeek V4 Pro", variant: .pro,
        nLayer: 61, nEmbd: 7168, nVocab: 129280, nHead: 128, nHeadKV: 1,
        nHeadDim: 512, nValueDim: 512, nRot: 64, nOutGroup: 16, nLoraQ: 1536,
        nLoraO: 1024, nExpert: 384, nExpertUsed: 6, nExpertShared: 1, nFFExp: 3072,
        nHashLayer: 3, nSWA: 128, nIndexerHead: 64, nIndexerHeadDim: 128,
        nIndexerTopK: 1024, nHC: 4, nHCSinkhornIter: 20, expertWeightScale: 2.5)

    /// Port of ds4_expected_layer_compress_ratio.
    public func expectedCompressRatio(layer: UInt32) -> UInt32 {
        switch variant {
        case .flash:
            if layer < 2 { return 0 }
            return (layer & 1) == 0 ? 4 : 128
        case .pro:
            if layer < 2 { return 128 }
            return (layer & 1) == 0 ? 4 : 128
        }
    }
}

/// A validated model configuration: the selected shape, per-layer compression
/// ratios, and per-layer SwiGLU clamp. Port of config_validate_model.
public struct ModelConfig: Sendable {
    public let shape: ModelShape
    public let compressRatios: [UInt32]      // one per layer
    public let swigluClampExp: [Float]       // one per layer
    public let ropeFreqBase: Float
    public let ropeScaleFactor: Float
    public let ropeYarnBetaFast: Float
    public let ropeYarnBetaSlow: Float
    public let compressRopeFreqBase: Float
    public let ropeOrigCtx: UInt64

    public init(model: GGUFModel) throws {
        func reqU32(_ key: String) throws -> UInt32 {
            guard let v = model.u32("deepseek4." + key) else { throw ModelConfigError.missing("deepseek4." + key) }
            return v
        }
        func reqF32(_ key: String) throws -> Float {
            guard let v = model.f32Compat("deepseek4." + key) else { throw ModelConfigError.missing("deepseek4." + key) }
            return v
        }

        // Read the shape-defining metadata.
        let nLayer = try reqU32("block_count")
        let nEmbd = try reqU32("embedding_length")
        let nVocab = try reqU32("vocab_size")
        let nHead = try reqU32("attention.head_count")
        let nHeadKV = try reqU32("attention.head_count_kv")
        let nHeadDim = try reqU32("attention.key_length")
        let nValueDim = try reqU32("attention.value_length")
        let nRot = try reqU32("rope.dimension_count")
        let nLoraQ = try reqU32("attention.q_lora_rank")
        let nLoraO = try reqU32("attention.output_lora_rank")
        let nOutGroup = try reqU32("attention.output_group_count")
        let nExpert = try reqU32("expert_count")
        let nExpertUsed = try reqU32("expert_used_count")
        let nFFExp = try reqU32("expert_feed_forward_length")
        let nExpertShared = try reqU32("expert_shared_count")
        let nHashLayer = try reqU32("hash_layer_count")
        let nSWA = try reqU32("attention.sliding_window")
        let nIndexerHead = try reqU32("attention.indexer.head_count")
        let nIndexerHeadDim = try reqU32("attention.indexer.key_length")
        let nIndexerTopK = try reqU32("attention.indexer.top_k")
        let nHC = try reqU32("hyper_connection.count")
        let nHCIter = try reqU32("hyper_connection.sinkhorn_iterations")

        // Select Flash or Pro by exact match of the shape-defining fields.
        func matches(_ s: ModelShape) -> Bool {
            s.nLayer == nLayer && s.nEmbd == nEmbd && s.nVocab == nVocab &&
            s.nHead == nHead && s.nHeadKV == nHeadKV && s.nHeadDim == nHeadDim &&
            s.nValueDim == nValueDim && s.nRot == nRot && s.nLoraQ == nLoraQ &&
            s.nLoraO == nLoraO && s.nOutGroup == nOutGroup && s.nExpert == nExpert &&
            s.nExpertUsed == nExpertUsed && s.nFFExp == nFFExp &&
            s.nExpertShared == nExpertShared && s.nHashLayer == nHashLayer &&
            s.nSWA == nSWA && s.nIndexerHead == nIndexerHead &&
            s.nIndexerHeadDim == nIndexerHeadDim && s.nIndexerTopK == nIndexerTopK &&
            s.nHC == nHC && s.nHCSinkhornIter == nHCIter
        }
        let selected: ModelShape
        if matches(.flash) { selected = .flash }
        else if matches(.pro) { selected = .pro }
        else {
            throw ModelConfigError.unsupportedShape(
                "layers=\(nLayer) embd=\(nEmbd) heads=\(nHead) q_lora=\(nLoraQ) out_groups=\(nOutGroup) experts=\(nExpert) ff_exp=\(nFFExp) indexer_top_k=\(nIndexerTopK)")
        }
        self.shape = selected

        // Expert groups must be absent/zero (config_expect_u32 == 0).
        let nGroups = model.u32("deepseek4.expert_group_count") ?? 0
        let nGroupsUsed = model.u32("deepseek4.expert_group_used_count") ?? 0
        if nGroups != 0 { throw ModelConfigError.mismatch("expert_group_count", expected: "0", got: "\(nGroups)") }
        if nGroupsUsed != 0 { throw ModelConfigError.mismatch("expert_group_used_count", expected: "0", got: "\(nGroupsUsed)") }

        // Per-layer compression ratios: validate against the expected formula.
        guard let ratios = model.intArray("deepseek4.attention.compress_ratios") else {
            throw ModelConfigError.missing("deepseek4.attention.compress_ratios")
        }
        if ratios.count < Int(nLayer) {
            throw ModelConfigError.mismatch("attention.compress_ratios.count",
                                            expected: ">= \(nLayer)", got: "\(ratios.count)")
        }
        var cr: [UInt32] = []
        cr.reserveCapacity(Int(nLayer))
        for il in 0..<nLayer {
            let raw = ratios[Int(il)]
            if raw < 0 { throw ModelConfigError.mismatch("compress_ratios[\(il)]", expected: ">= 0", got: "\(raw)") }
            let got = UInt32(truncatingIfNeeded: raw)
            let expected = selected.expectedCompressRatio(layer: il)
            if got != expected {
                throw ModelConfigError.mismatch("compress_ratios[\(il)]", expected: "\(expected)", got: "\(got)")
            }
            cr.append(got)
        }
        self.compressRatios = cr

        // Per-layer SwiGLU clamp: every entry must equal the shape default.
        guard let clamps = model.floatArray("deepseek4.swiglu_clamp_exp") else {
            throw ModelConfigError.missing("deepseek4.swiglu_clamp_exp")
        }
        if clamps.count < Int(nLayer) {
            throw ModelConfigError.mismatch("swiglu_clamp_exp.count", expected: ">= \(nLayer)", got: "\(clamps.count)")
        }
        var clampOut: [Float] = []
        clampOut.reserveCapacity(Int(nLayer))
        for il in 0..<Int(nLayer) {
            let got = Float(clamps[il])
            if got != ModelDefaults.swigluClampExp {
                throw ModelConfigError.mismatch("swiglu_clamp_exp[\(il)]",
                                                expected: "\(ModelDefaults.swigluClampExp)", got: "\(got)")
            }
            clampOut.append(got)
        }
        self.swigluClampExp = clampOut

        // RoPE scaling parameters (defaults when the optional keys are absent).
        let origCtx = model.u64Compat("deepseek4.rope.scaling.original_context_length") ?? ModelDefaults.ropeOrigCtx
        if origCtx != ModelDefaults.ropeOrigCtx {
            throw ModelConfigError.mismatch("rope.scaling.original_context_length",
                                            expected: "\(ModelDefaults.ropeOrigCtx)", got: "\(origCtx)")
        }
        self.ropeOrigCtx = origCtx

        let freqBase = try reqF32("rope.freq_base")
        if freqBase != ModelDefaults.ropeFreqBase {
            throw ModelConfigError.mismatch("rope.freq_base", expected: "\(ModelDefaults.ropeFreqBase)", got: "\(freqBase)")
        }
        self.ropeFreqBase = freqBase

        let scaleFactor = model.f32Compat("deepseek4.rope.scaling.factor") ?? ModelDefaults.ropeScaleFactor
        if scaleFactor != ModelDefaults.ropeScaleFactor {
            throw ModelConfigError.mismatch("rope.scaling.factor", expected: "\(ModelDefaults.ropeScaleFactor)", got: "\(scaleFactor)")
        }
        self.ropeScaleFactor = scaleFactor

        let betaFast = model.f32Compat("deepseek4.rope.scaling.yarn_beta_fast") ?? ModelDefaults.ropeYarnBetaFast
        if betaFast != ModelDefaults.ropeYarnBetaFast {
            throw ModelConfigError.mismatch("rope.scaling.yarn_beta_fast", expected: "\(ModelDefaults.ropeYarnBetaFast)", got: "\(betaFast)")
        }
        self.ropeYarnBetaFast = betaFast

        let betaSlow = model.f32Compat("deepseek4.rope.scaling.yarn_beta_slow") ?? ModelDefaults.ropeYarnBetaSlow
        if betaSlow != ModelDefaults.ropeYarnBetaSlow {
            throw ModelConfigError.mismatch("rope.scaling.yarn_beta_slow", expected: "\(ModelDefaults.ropeYarnBetaSlow)", got: "\(betaSlow)")
        }
        self.ropeYarnBetaSlow = betaSlow

        let compressFreq = try reqF32("attention.compress_rope_freq_base")
        if compressFreq != ModelDefaults.compressRopeFreqBase {
            throw ModelConfigError.mismatch("attention.compress_rope_freq_base", expected: "\(ModelDefaults.compressRopeFreqBase)", got: "\(compressFreq)")
        }
        self.compressRopeFreqBase = compressFreq
    }
}
