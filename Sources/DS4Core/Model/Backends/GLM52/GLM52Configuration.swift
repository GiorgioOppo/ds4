import Foundation

/// Load-time validation errors for the one GLM-DSA geometry supported by this
/// backend.  Keeping these separate from DeepSeek errors prevents a future GLM
/// variant from being silently interpreted with DeepSeek constants.
public enum GLM52ConfigurationError: Error, Sendable, Equatable, CustomStringConvertible {
    case wrongArchitecture(expected: String, got: String?)
    case missing(String)
    case mismatch(String, expected: String, got: String)

    public var description: String {
        switch self {
        case .wrongArchitecture(let expected, let got):
            let actual = got ?? "missing"
            return "GGUF general.architecture: expected \(expected), got \(actual)"
        case .missing(let key):
            return "required GLM 5.2 metadata key is missing or has the wrong type: \(key)"
        case .mismatch(let key, let expected, let got):
            return "GLM 5.2 metadata \(key): expected \(expected), got \(got)"
        }
    }
}

/// Exact GLM 5.2 geometry implemented by antirez/ds4's `glm5.2` branch.
///
/// `nLayer` is the GGUF block count.  The last block is the one stored next-token
/// prediction (MTP/nextn) block, so ordinary autoregressive inference executes
/// `inferenceLayerCount` transformer blocks.
public struct GLM52Shape: Sendable, Equatable {
    public let name: String
    public let metadataNamespace: String
    public let nLayer: UInt32
    public let nEmbd: UInt32
    public let nVocab: UInt32
    public let nHead: UInt32
    public let nHeadKV: UInt32
    public let nHeadDim: UInt32
    public let nValueDim: UInt32
    public let nRot: UInt32
    public let nLoraQ: UInt32
    public let nKVLoRA: UInt32
    public let nKeyMLA: UInt32
    public let nValueMLA: UInt32
    public let nExpert: UInt32
    public let nExpertUsed: UInt32
    public let nExpertShared: UInt32
    public let nFFExpert: UInt32
    public let nFFDense: UInt32
    public let nExpertGroup: UInt32
    public let nExpertGroupUsed: UInt32
    public let expertGatingFunction: UInt32
    public let nLeadingDense: UInt32
    public let nNextNPredict: UInt32
    public let nIndexerHead: UInt32
    public let nIndexerHeadDim: UInt32
    public let nIndexerTopK: UInt32
    public let rmsEpsilon: Float
    public let expertWeightScale: Float
    public let ropeFrequencyBase: Float
    public let originalContextLength: UInt64

    public var inferenceLayerCount: UInt32 { nLayer - nNextNPredict }
    public var queryProjectionWidth: UInt32 { nHead * nKeyMLA }
    public var queryNonRoPEWidth: UInt32 { nKeyMLA - nRot }
    public var indexerQueryWidth: UInt32 { nIndexerHead * nIndexerHeadDim }

    public static let v5_2 = GLM52Shape(
        name: "GLM 5.2",
        metadataNamespace: "glm-dsa",
        nLayer: 79,
        nEmbd: 6_144,
        nVocab: 154_880,
        nHead: 64,
        nHeadKV: 1,
        nHeadDim: 576,
        nValueDim: 512,
        nRot: 64,
        nLoraQ: 2_048,
        nKVLoRA: 512,
        nKeyMLA: 256,
        nValueMLA: 256,
        nExpert: 256,
        nExpertUsed: 8,
        nExpertShared: 1,
        nFFExpert: 2_048,
        nFFDense: 12_288,
        nExpertGroup: 1,
        nExpertGroupUsed: 1,
        expertGatingFunction: 2,
        nLeadingDense: 3,
        nNextNPredict: 1,
        nIndexerHead: 32,
        nIndexerHeadDim: 128,
        nIndexerTopK: 2_048,
        rmsEpsilon: 1.0e-5,
        expertWeightScale: 2.5,
        ropeFrequencyBase: 8_000_000,
        originalContextLength: 1_048_576
    )
}

/// Validated GLM 5.2 model metadata.
///
/// This type intentionally validates `general.architecture` itself rather than
/// depending on the shared backend registry.  Architecture registration and
/// runtime selection can therefore be landed independently without weakening
/// the backend-owned GGUF contract.
public struct GLM52Configuration: Sendable, Equatable {
    public let shape: GLM52Shape
    public let expertWeightsNormalized: Bool

    /// Portable information may be exposed by model management before the
    /// numerical backend is enabled.  Availability remains explicit so a valid
    /// GLM file cannot accidentally fall through to the DeepSeek decoder.
    public var descriptor: ModelDescriptor {
        ModelDescriptor(
            architecture: DetectedModelArchitecture(
                id: .glmDSA,
                family: .glm,
                backendAvailability: .recognizedButNotImplemented
            ),
            name: shape.name,
            layerCount: Int(shape.inferenceLayerCount),
            embeddingLength: Int(shape.nEmbd),
            vocabularySize: Int(shape.nVocab),
            capabilities: [.chat, .tools, .reasoning, .mixtureOfExperts, .compressedAttention]
        )
    }

    public init(model: GGUFModel, shape: GLM52Shape = .v5_2) throws {
        guard model.string("general.architecture") == shape.metadataNamespace else {
            throw GLM52ConfigurationError.wrongArchitecture(
                expected: shape.metadataNamespace,
                got: model.string("general.architecture")
            )
        }

        let prefix = shape.metadataNamespace + "."

        func requiredU32(_ suffix: String) throws -> UInt32 {
            let key = prefix + suffix
            guard let value = model.u32(key) else {
                throw GLM52ConfigurationError.missing(key)
            }
            return value
        }

        func requiredU64(_ suffix: String) throws -> UInt64 {
            let key = prefix + suffix
            guard let value = model.u64Compat(key) else {
                throw GLM52ConfigurationError.missing(key)
            }
            return value
        }

        func requiredFloat(_ suffix: String) throws -> Float {
            let key = prefix + suffix
            guard let value = model.f32Compat(key) else {
                throw GLM52ConfigurationError.missing(key)
            }
            return value
        }

        func requiredBool(_ suffix: String) throws -> Bool {
            let key = prefix + suffix
            guard let value = model.bool(key) else {
                throw GLM52ConfigurationError.missing(key)
            }
            return value
        }

        func expect(_ suffix: String, _ expected: UInt32) throws {
            let got = try requiredU32(suffix)
            guard got == expected else {
                throw GLM52ConfigurationError.mismatch(
                    prefix + suffix, expected: String(expected), got: String(got))
            }
        }

        func expect(_ suffix: String, _ expected: UInt64) throws {
            let got = try requiredU64(suffix)
            guard got == expected else {
                throw GLM52ConfigurationError.mismatch(
                    prefix + suffix, expected: String(expected), got: String(got))
            }
        }

        func expect(_ suffix: String, _ expected: Float) throws {
            let got = try requiredFloat(suffix)
            guard Self.floatMatches(got, expected) else {
                throw GLM52ConfigurationError.mismatch(
                    prefix + suffix, expected: String(expected), got: String(got))
            }
        }

        try expect("block_count", shape.nLayer)
        try expect("context_length", shape.originalContextLength)
        try expect("embedding_length", shape.nEmbd)
        try expect("vocab_size", shape.nVocab)
        try expect("feed_forward_length", shape.nFFDense)
        try expect("attention.head_count", shape.nHead)
        try expect("attention.head_count_kv", shape.nHeadKV)
        try expect("attention.key_length", shape.nHeadDim)
        try expect("attention.value_length", shape.nValueDim)
        try expect("rope.dimension_count", shape.nRot)
        try expect("attention.q_lora_rank", shape.nLoraQ)
        try expect("attention.kv_lora_rank", shape.nKVLoRA)
        try expect("attention.key_length_mla", shape.nKeyMLA)
        try expect("attention.value_length_mla", shape.nValueMLA)
        try expect("expert_count", shape.nExpert)
        try expect("expert_used_count", shape.nExpertUsed)
        try expect("expert_feed_forward_length", shape.nFFExpert)
        try expect("expert_shared_count", shape.nExpertShared)
        try expect("expert_group_count", shape.nExpertGroup)
        try expect("expert_group_used_count", shape.nExpertGroupUsed)
        try expect("expert_gating_func", shape.expertGatingFunction)
        try expect("leading_dense_block_count", shape.nLeadingDense)
        try expect("nextn_predict_layers", shape.nNextNPredict)
        try expect("attention.indexer.head_count", shape.nIndexerHead)
        try expect("attention.indexer.key_length", shape.nIndexerHeadDim)
        try expect("attention.indexer.top_k", shape.nIndexerTopK)
        try expect("rope.freq_base", shape.ropeFrequencyBase)
        try expect("attention.layer_norm_rms_epsilon", shape.rmsEpsilon)
        try expect("expert_weights_scale", shape.expertWeightScale)

        let normalized = try requiredBool("expert_weights_norm")
        guard normalized else {
            throw GLM52ConfigurationError.mismatch(
                prefix + "expert_weights_norm", expected: "true", got: "false")
        }

        self.shape = shape
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
