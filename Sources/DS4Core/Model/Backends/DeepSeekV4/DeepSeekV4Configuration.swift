import Foundation

// Faithful Swift port of the DeepSeek V4 model shape configuration in ds4.c:
// the Flash/Pro shape constants, shape selection from GGUF metadata, the
// per-layer compression-ratio formula, and the metadata validation that the C
// engine performs at load. Phase 4 of the C->Swift port.
//
// Validation: cross-checked against `./ds4 --inspect` (which prints the shape
// the C engine selected) and against the GGUF's own metadata arrays.

public enum DeepSeekV4Variant: Int, Sendable {
    case flash = 0
    case pro = 1
}

public enum DeepSeekV4ConfigurationError: Error, CustomStringConvertible {
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
public enum DeepSeekV4Defaults {
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
public struct DeepSeekV4Shape: Sendable, Equatable {
    public let name: String
    public let variant: DeepSeekV4Variant
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

    public static let flash = DeepSeekV4Shape(
        name: "DeepSeek V4 Flash", variant: .flash,
        nLayer: 43, nEmbd: 4096, nVocab: 129280, nHead: 64, nHeadKV: 1,
        nHeadDim: 512, nValueDim: 512, nRot: 64, nOutGroup: 8, nLoraQ: 1024,
        nLoraO: 1024, nExpert: 256, nExpertUsed: 6, nExpertShared: 1, nFFExp: 2048,
        nHashLayer: 3, nSWA: 128, nIndexerHead: 64, nIndexerHeadDim: 128,
        nIndexerTopK: 512, nHC: 4, nHCSinkhornIter: 20, expertWeightScale: 1.5)

    public static let pro = DeepSeekV4Shape(
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
public struct DeepSeekV4Configuration: Sendable {
    public let shape: DeepSeekV4Shape
    public let compressRatios: [UInt32]      // one per layer
    public let swigluClampExp: [Float]       // one per layer
    public let ropeFreqBase: Float
    public let ropeScaleFactor: Float
    public let ropeYarnBetaFast: Float
    public let ropeYarnBetaSlow: Float
    public let compressRopeFreqBase: Float
    public let ropeOrigCtx: UInt64

    /// Port of config_expect_f32 (ds4.c:3867): floats from the GGUF are
    /// compared with a RELATIVE 1e-6 tolerance, not bit-exactly — f32/f64
    /// re-encodings of the same constant must not be rejected.
    static func f32Matches(_ got: Float, _ expected: Float) -> Bool {
        let scale = abs(expected) > 1 ? abs(expected) : 1
        return abs(got - expected) <= scale * 1.0e-6
    }

    public var descriptor: ModelDescriptor {
        ModelDescriptor(
            architecture: DetectedModelArchitecture(
                id: .deepSeekV4,
                family: .deepSeek,
                backendAvailability: .implemented
            ),
            name: shape.name,
            layerCount: Int(shape.nLayer),
            embeddingLength: Int(shape.nEmbd),
            vocabularySize: Int(shape.nVocab),
            capabilities: [.chat, .tools, .reasoning, .mixtureOfExperts, .compressedAttention]
        )
    }

    public init(model: GGUFModel) throws {
        let detected = try ModelArchitectureDetector.detect(in: model)
        try ModelArchitectureDetector.requireImplemented(detected)
        guard detected.id == .deepSeekV4 else {
            throw ModelArchitectureError.unsupportedArchitecture(detected.id)
        }

        func reqU32(_ key: String) throws -> UInt32 {
            guard let v = model.u32("deepseek4." + key) else { throw DeepSeekV4ConfigurationError.missing("deepseek4." + key) }
            return v
        }
        func reqF32(_ key: String) throws -> Float {
            guard let v = model.f32Compat("deepseek4." + key) else { throw DeepSeekV4ConfigurationError.missing("deepseek4." + key) }
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
        func matches(_ s: DeepSeekV4Shape) -> Bool {
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
        let selected: DeepSeekV4Shape
        if matches(.flash) { selected = .flash }
        else if matches(.pro) { selected = .pro }
        else {
            throw DeepSeekV4ConfigurationError.unsupportedShape(
                "layers=\(nLayer) embd=\(nEmbd) heads=\(nHead) q_lora=\(nLoraQ) out_groups=\(nOutGroup) experts=\(nExpert) ff_exp=\(nFFExp) indexer_top_k=\(nIndexerTopK)")
        }
        self.shape = selected

        // Expert groups must be absent/zero (config_expect_u32 == 0).
        let nGroups = model.u32("deepseek4.expert_group_count") ?? 0
        let nGroupsUsed = model.u32("deepseek4.expert_group_used_count") ?? 0
        if nGroups != 0 { throw DeepSeekV4ConfigurationError.mismatch("expert_group_count", expected: "0", got: "\(nGroups)") }
        if nGroupsUsed != 0 { throw DeepSeekV4ConfigurationError.mismatch("expert_group_used_count", expected: "0", got: "\(nGroupsUsed)") }

        // Per-layer compression ratios: validate against the expected formula.
        guard let ratios = model.intArray("deepseek4.attention.compress_ratios") else {
            throw DeepSeekV4ConfigurationError.missing("deepseek4.attention.compress_ratios")
        }
        if ratios.count < Int(nLayer) {
            throw DeepSeekV4ConfigurationError.mismatch("attention.compress_ratios.count",
                                            expected: ">= \(nLayer)", got: "\(ratios.count)")
        }
        var cr: [UInt32] = []
        cr.reserveCapacity(Int(nLayer))
        for il in 0..<nLayer {
            let raw = ratios[Int(il)]
            if raw < 0 { throw DeepSeekV4ConfigurationError.mismatch("compress_ratios[\(il)]", expected: ">= 0", got: "\(raw)") }
            let got = UInt32(truncatingIfNeeded: raw)
            let expected = selected.expectedCompressRatio(layer: il)
            if got != expected {
                throw DeepSeekV4ConfigurationError.mismatch("compress_ratios[\(il)]", expected: "\(expected)", got: "\(got)")
            }
            cr.append(got)
        }
        self.compressRatios = cr

        // Per-layer SwiGLU clamp: every entry must equal the shape default.
        guard let clamps = model.floatArray("deepseek4.swiglu_clamp_exp") else {
            throw DeepSeekV4ConfigurationError.missing("deepseek4.swiglu_clamp_exp")
        }
        if clamps.count < Int(nLayer) {
            throw DeepSeekV4ConfigurationError.mismatch("swiglu_clamp_exp.count", expected: ">= \(nLayer)", got: "\(clamps.count)")
        }
        var clampOut: [Float] = []
        clampOut.reserveCapacity(Int(nLayer))
        for il in 0..<Int(nLayer) {
            let got = Float(clamps[il])
            if !Self.f32Matches(got, DeepSeekV4Defaults.swigluClampExp) {
                throw DeepSeekV4ConfigurationError.mismatch("swiglu_clamp_exp[\(il)]",
                                                expected: "\(DeepSeekV4Defaults.swigluClampExp)", got: "\(got)")
            }
            clampOut.append(got)
        }
        self.swigluClampExp = clampOut

        // RoPE scaling parameters (defaults when the optional keys are absent).
        let origCtx = model.u64Compat("deepseek4.rope.scaling.original_context_length") ?? DeepSeekV4Defaults.ropeOrigCtx
        if origCtx != DeepSeekV4Defaults.ropeOrigCtx {
            throw DeepSeekV4ConfigurationError.mismatch("rope.scaling.original_context_length",
                                            expected: "\(DeepSeekV4Defaults.ropeOrigCtx)", got: "\(origCtx)")
        }
        self.ropeOrigCtx = origCtx

        let freqBase = try reqF32("rope.freq_base")
        if !Self.f32Matches(freqBase, DeepSeekV4Defaults.ropeFreqBase) {
            throw DeepSeekV4ConfigurationError.mismatch("rope.freq_base", expected: "\(DeepSeekV4Defaults.ropeFreqBase)", got: "\(freqBase)")
        }
        self.ropeFreqBase = freqBase

        let scaleFactor = model.f32Compat("deepseek4.rope.scaling.factor") ?? DeepSeekV4Defaults.ropeScaleFactor
        if !Self.f32Matches(scaleFactor, DeepSeekV4Defaults.ropeScaleFactor) {
            throw DeepSeekV4ConfigurationError.mismatch("rope.scaling.factor", expected: "\(DeepSeekV4Defaults.ropeScaleFactor)", got: "\(scaleFactor)")
        }
        self.ropeScaleFactor = scaleFactor

        let betaFast = model.f32Compat("deepseek4.rope.scaling.yarn_beta_fast") ?? DeepSeekV4Defaults.ropeYarnBetaFast
        if !Self.f32Matches(betaFast, DeepSeekV4Defaults.ropeYarnBetaFast) {
            throw DeepSeekV4ConfigurationError.mismatch("rope.scaling.yarn_beta_fast", expected: "\(DeepSeekV4Defaults.ropeYarnBetaFast)", got: "\(betaFast)")
        }
        self.ropeYarnBetaFast = betaFast

        let betaSlow = model.f32Compat("deepseek4.rope.scaling.yarn_beta_slow") ?? DeepSeekV4Defaults.ropeYarnBetaSlow
        if !Self.f32Matches(betaSlow, DeepSeekV4Defaults.ropeYarnBetaSlow) {
            throw DeepSeekV4ConfigurationError.mismatch("rope.scaling.yarn_beta_slow", expected: "\(DeepSeekV4Defaults.ropeYarnBetaSlow)", got: "\(betaSlow)")
        }
        self.ropeYarnBetaSlow = betaSlow

        let compressFreq = try reqF32("attention.compress_rope_freq_base")
        if !Self.f32Matches(compressFreq, DeepSeekV4Defaults.compressRopeFreqBase) {
            throw DeepSeekV4ConfigurationError.mismatch("attention.compress_rope_freq_base", expected: "\(DeepSeekV4Defaults.compressRopeFreqBase)", got: "\(compressFreq)")
        }
        self.compressRopeFreqBase = compressFreq

        // Engine-level constants the C validates at load (ds4.c:3990-3997): the
        // declared training config must match what the engine hardcodes, or the
        // file is refused instead of silently run with different math.
        let ews = try reqF32("expert_weights_scale")
        if !Self.f32Matches(ews, selected.expertWeightScale) {
            throw DeepSeekV4ConfigurationError.mismatch("expert_weights_scale",
                                            expected: "\(selected.expertWeightScale)", got: "\(ews)")
        }
        let rmsEps = try reqF32("attention.layer_norm_rms_epsilon")
        if !Self.f32Matches(rmsEps, DeepSeekV4Defaults.rmsEps) {
            throw DeepSeekV4ConfigurationError.mismatch("attention.layer_norm_rms_epsilon",
                                            expected: "\(DeepSeekV4Defaults.rmsEps)", got: "\(rmsEps)")
        }
        let hcEps = try reqF32("hyper_connection.epsilon")
        if !Self.f32Matches(hcEps, DeepSeekV4Defaults.hcEps) {
            throw DeepSeekV4ConfigurationError.mismatch("hyper_connection.epsilon",
                                            expected: "\(DeepSeekV4Defaults.hcEps)", got: "\(hcEps)")
        }
        guard let ewNorm = model.bool("deepseek4.expert_weights_norm") else {
            throw DeepSeekV4ConfigurationError.missing("deepseek4.expert_weights_norm")
        }
        if !ewNorm {
            throw DeepSeekV4ConfigurationError.mismatch("expert_weights_norm", expected: "true", got: "false")
        }
    }
}

// MARK: - Source-compatible DeepSeek-V4 aliases

/// Legacy names retained while DS4Engine/DS4Metal migrate to the explicit
/// backend namespace. They resolve to the same concrete types and behavior.
public typealias ModelVariant = DeepSeekV4Variant
public typealias ModelConfigError = DeepSeekV4ConfigurationError
public typealias ModelDefaults = DeepSeekV4Defaults
public typealias ModelShape = DeepSeekV4Shape
public typealias ModelConfig = DeepSeekV4Configuration
