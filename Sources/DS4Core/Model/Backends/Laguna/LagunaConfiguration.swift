import Foundation

/// Load-time validation errors for the one Laguna geometry supported by this
/// backend.  Keeping these separate from DeepSeek/GLM errors prevents a future
/// Laguna variant from being silently interpreted with another family's
/// constants.
public enum LagunaConfigurationError: Error, Sendable, Equatable, CustomStringConvertible {
    case wrongArchitecture(expected: String, got: String?)
    case missing(String)
    case mismatch(String, expected: String, got: String)
    case invalidHeadCountArray(String)
    case unexpectedHeadCount(layer: Int, expected: UInt32, got: UInt32)

    public var description: String {
        switch self {
        case .wrongArchitecture(let expected, let got):
            let actual = got ?? "missing"
            return "GGUF general.architecture: expected \(expected), got \(actual)"
        case .missing(let key):
            return "required Laguna metadata key is missing or has the wrong type: \(key)"
        case .mismatch(let key, let expected, let got):
            return "Laguna metadata \(key): expected \(expected), got \(got)"
        case .invalidHeadCountArray(let key):
            return "\(key) must be an int32/uint32 array with one entry per layer"
        case .unexpectedHeadCount(let layer, let expected, let got):
            return "unexpected Laguna head count at layer \(layer): got \(got), expected \(expected)"
        }
    }
}

/// Exact Laguna S 2.1 geometry implemented by antirez/ds4's `laguna-s2.1`
/// branch (`DS4_SHAPE_LAGUNA_S21` in `ds4.c`).
///
/// Laguna is a Poolside GQA + MoE transformer: 48 blocks, 8 KV heads shared by
/// a per-layer query-head count (48 heads on every fourth block, 72 elsewhere),
/// one leading dense block, then 256 routed experts with 10 active plus one
/// shared expert per MoE block.  Every fourth block (those with 48 heads) uses
/// full attention with 64 YaRN-scaled RoPE dimensions; the other blocks use
/// 512-token sliding-window attention with full-width 128-dimension RoPE on an
/// independent frequency base.
public struct LagunaShape: Sendable, Equatable {
    public let name: String
    public let metadataNamespace: String
    public let nLayer: UInt32
    public let nEmbd: UInt32
    public let nVocab: UInt32
    public let nHead: UInt32
    public let nHeadFull: UInt32
    public let nHeadKV: UInt32
    public let nHeadDim: UInt32
    public let nValueDim: UInt32
    public let nRot: UInt32
    public let nRotSWA: UInt32
    public let nSWA: UInt32
    public let nExpert: UInt32
    public let nExpertUsed: UInt32
    public let nExpertShared: UInt32
    public let nFFExpert: UInt32
    public let nFFShared: UInt32
    public let nFFDense: UInt32
    public let nLeadingDense: UInt32
    public let expertGatingFunction: UInt32
    public let rmsEpsilon: Float
    public let expertWeightScale: Float
    public let ropeFrequencyBase: Float
    public let ropeFrequencyBaseSWA: Float
    public let ropeScaleFactor: Float
    public let ropeYarnBetaFast: Float
    public let ropeYarnBetaSlow: Float
    public let ropeYarnAttnFactor: Float
    public let contextLength: UInt64
    public let ropeOriginalContext: UInt64

    /// Query-head count of a block.  Upstream reads the per-layer array from
    /// the GGUF and enforces this exact alternation: every fourth block runs
    /// 48 full-attention heads, the rest run 72 sliding-window heads.
    public func layerHeadCount(_ layer: Int) -> UInt32 {
        layer % 4 == 0 ? nHeadFull : nHead
    }

    /// Sliding-window blocks are exactly those with the full 72-head count
    /// (`ds4_laguna_layer_is_swa` in `ds4.c`).
    public func isSlidingWindowLayer(_ layer: Int) -> Bool {
        layerHeadCount(layer) == nHead
    }

    public func queryProjectionWidth(_ layer: Int) -> UInt32 {
        layerHeadCount(layer) * nHeadDim
    }

    public var keyValueProjectionWidth: UInt32 { nHeadKV * nHeadDim }

    public static let s2_1 = LagunaShape(
        name: "Laguna S 2.1",
        metadataNamespace: "laguna",
        nLayer: 48,
        nEmbd: 3_072,
        nVocab: 100_352,
        nHead: 72,
        nHeadFull: 48,
        nHeadKV: 8,
        nHeadDim: 128,
        nValueDim: 128,
        nRot: 64,
        nRotSWA: 128,
        nSWA: 512,
        nExpert: 256,
        nExpertUsed: 10,
        nExpertShared: 1,
        nFFExpert: 1_024,
        nFFShared: 1_024,
        nFFDense: 12_288,
        nLeadingDense: 1,
        expertGatingFunction: 2,
        rmsEpsilon: 1.0e-6,
        expertWeightScale: 2.5,
        ropeFrequencyBase: 500_000,
        ropeFrequencyBaseSWA: 10_000,
        ropeScaleFactor: 32,
        ropeYarnBetaFast: 32,
        ropeYarnBetaSlow: 1,
        ropeYarnAttnFactor: 1,
        contextLength: 262_144,
        ropeOriginalContext: 8_192
    )
}

/// Validated Laguna S 2.1 model metadata.
///
/// This type intentionally validates `general.architecture` itself rather than
/// depending on the shared backend registry, mirroring
/// `config_validate_laguna_model` in upstream `ds4.c`: every geometry constant
/// present in the GGUF must match `DS4_SHAPE_LAGUNA_S21` exactly, the per-layer
/// head-count array must follow the 48/72 alternation, and RoPE scaling must be
/// YaRN with the pinned parameters.
public struct LagunaConfiguration: Sendable, Equatable {
    public let shape: LagunaShape
    public let headCounts: [UInt32]
    public let expertWeightsNormalized: Bool

    /// Portable information may be exposed by model management before the
    /// numerical backend is enabled.  Availability remains explicit so a valid
    /// Laguna file cannot accidentally fall through to another decoder.
    public var descriptor: ModelDescriptor {
        ModelDescriptor(
            architecture: DetectedModelArchitecture(
                id: .laguna,
                family: .laguna,
                backendAvailability: .recognizedButNotImplemented
            ),
            name: shape.name,
            layerCount: Int(shape.nLayer),
            embeddingLength: Int(shape.nEmbd),
            vocabularySize: Int(shape.nVocab),
            capabilities: [.chat, .tools, .reasoning, .mixtureOfExperts]
        )
    }

    public init(model: GGUFModel, shape: LagunaShape = .s2_1) throws {
        guard model.string("general.architecture") == shape.metadataNamespace else {
            throw LagunaConfigurationError.wrongArchitecture(
                expected: shape.metadataNamespace,
                got: model.string("general.architecture")
            )
        }

        let prefix = shape.metadataNamespace + "."

        func requiredU32(_ suffix: String) throws -> UInt32 {
            let key = prefix + suffix
            guard let value = model.u32(key) else {
                throw LagunaConfigurationError.missing(key)
            }
            return value
        }

        func requiredU64(_ suffix: String) throws -> UInt64 {
            let key = prefix + suffix
            guard let value = model.u64Compat(key) else {
                throw LagunaConfigurationError.missing(key)
            }
            return value
        }

        func requiredFloat(_ suffix: String) throws -> Float {
            let key = prefix + suffix
            guard let value = model.f32Compat(key) else {
                throw LagunaConfigurationError.missing(key)
            }
            return value
        }

        func requiredBool(_ suffix: String) throws -> Bool {
            let key = prefix + suffix
            guard let value = model.bool(key) else {
                throw LagunaConfigurationError.missing(key)
            }
            return value
        }

        func requiredString(_ suffix: String) throws -> String {
            let key = prefix + suffix
            guard let value = model.string(key) else {
                throw LagunaConfigurationError.missing(key)
            }
            return value
        }

        func expect(_ suffix: String, _ expected: UInt32) throws {
            let got = try requiredU32(suffix)
            guard got == expected else {
                throw LagunaConfigurationError.mismatch(
                    prefix + suffix, expected: String(expected), got: String(got))
            }
        }

        func expect(_ suffix: String, _ expected: UInt64) throws {
            let got = try requiredU64(suffix)
            guard got == expected else {
                throw LagunaConfigurationError.mismatch(
                    prefix + suffix, expected: String(expected), got: String(got))
            }
        }

        func expect(_ suffix: String, _ expected: Float) throws {
            let got = try requiredFloat(suffix)
            guard Self.floatMatches(got, expected) else {
                throw LagunaConfigurationError.mismatch(
                    prefix + suffix, expected: String(expected), got: String(got))
            }
        }

        try expect("block_count", shape.nLayer)
        try expect("context_length", shape.contextLength)
        try expect("embedding_length", shape.nEmbd)
        try expect("vocab_size", shape.nVocab)
        try expect("feed_forward_length", shape.nFFDense)
        try expect("attention.head_count_kv", shape.nHeadKV)
        try expect("attention.key_length", shape.nHeadDim)
        try expect("attention.value_length", shape.nValueDim)
        try expect("rope.dimension_count", shape.nRot)
        try expect("rope.dimension_count_swa", shape.nRotSWA)
        try expect("attention.sliding_window", shape.nSWA)
        try expect("expert_count", shape.nExpert)
        try expect("expert_used_count", shape.nExpertUsed)
        try expect("expert_feed_forward_length", shape.nFFExpert)
        try expect("expert_shared_feed_forward_length", shape.nFFShared)
        try expect("expert_gating_func", shape.expertGatingFunction)
        try expect("leading_dense_block_count", shape.nLeadingDense)

        let headCountKey = prefix + "attention.head_count"
        guard let rawHeadCounts = model.intArray(headCountKey),
              rawHeadCounts.count == Int(shape.nLayer) else {
            throw LagunaConfigurationError.invalidHeadCountArray(headCountKey)
        }
        var headCounts: [UInt32] = []
        headCounts.reserveCapacity(rawHeadCounts.count)
        for (layer, raw) in rawHeadCounts.enumerated() {
            let expected = shape.layerHeadCount(layer)
            guard raw > 0, raw <= Int64(UInt32.max), UInt32(raw) == expected else {
                throw LagunaConfigurationError.unexpectedHeadCount(
                    layer: layer,
                    expected: expected,
                    got: raw > 0 && raw <= Int64(UInt32.max) ? UInt32(raw) : 0
                )
            }
            headCounts.append(expected)
        }

        let ropeScalingType = try requiredString("rope.scaling.type")
        guard ropeScalingType == "yarn" else {
            throw LagunaConfigurationError.mismatch(
                prefix + "rope.scaling.type", expected: "yarn", got: ropeScalingType)
        }
        try expect("rope.scaling.original_context_length", shape.ropeOriginalContext)
        try expect("rope.freq_base", shape.ropeFrequencyBase)
        try expect("rope.freq_base_swa", shape.ropeFrequencyBaseSWA)
        try expect("rope.scaling.factor", shape.ropeScaleFactor)
        try expect("rope.scaling.yarn_attn_factor", shape.ropeYarnAttnFactor)
        try expect("rope.scaling.yarn_beta_fast", shape.ropeYarnBetaFast)
        try expect("rope.scaling.yarn_beta_slow", shape.ropeYarnBetaSlow)
        try expect("attention.layer_norm_rms_epsilon", shape.rmsEpsilon)
        try expect("expert_weights_scale", shape.expertWeightScale)

        let normalized = try requiredBool("expert_weights_norm")
        guard normalized else {
            throw LagunaConfigurationError.mismatch(
                prefix + "expert_weights_norm", expected: "true", got: "false")
        }

        self.shape = shape
        self.headCounts = headCounts
        self.expertWeightsNormalized = normalized
    }

    /// Same relative tolerance used by the reference loader.  It accepts a
    /// semantically identical F64-to-F32 metadata conversion without making
    /// genuinely different inference constants pass validation.
    static func floatMatches(_ got: Float, _ expected: Float) -> Bool {
        let scale = max(abs(expected), 1)
        return abs(got - expected) <= scale * 1.0e-6
    }
}
