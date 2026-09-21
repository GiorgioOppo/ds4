import Foundation

/// GLM 5.3 Flash's published glm5-next geometry. The final stored block is
/// MTP; ordinary causal inference executes the 45-block trunk only.
public struct GLM53Configuration: Sendable, Equatable {
    public let layerCount = 45
    public let embeddingLength = 4096
    public let vocabularySize = 154880
    public let maximumContextLength = 1048576
    public let headCount = 64
    public let kdaHeadDimension = 128
    public let kvRank = 512
    public let queryRank = 1536
    public let expertCount = 288
    public let expertsUsed = 8
    public let expertWidth = 2048
    public let rmsEpsilon: Float = 1e-5
    public let hcEpsilon: Float = 1e-6
    public let kdaLowerBound: Float = -5

    public static func isKDALayer(_ layer: Int) -> Bool { layer >= 0 && layer < 45 && layer % 4 != 3 }

    public init(model: GGUFModel) throws {
        func fail(_ key: String, _ expected: String) -> GLM53ConfigurationError {
            .invalidMetadata(key, expected: expected)
        }
        guard model.string("general.architecture") == "glm5-next" else {
            throw fail("general.architecture", "glm5-next")
        }
        let integers: [(String, UInt64)] = [
            ("block_count", 46), ("trunk_block_count", 45), ("nextn_predict_layers", 1),
            ("context_length", 1048576), ("embedding_length", 4096), ("vocab_size", 154880),
            ("feed_forward_length", 12288), ("expert_feed_forward_length", 2048),
            ("expert_count", 288), ("expert_used_count", 8), ("expert_shared_count", 1),
            ("leading_dense_block_count", 3), ("attention.head_count", 64),
            ("attention.key_length", 256), ("attention.value_length", 256),
            ("attention.q_lora_rank", 1536), ("attention.kv_lora_rank", 512),
            ("attention.rope_dimension_count", 0), ("attention.indexer.head_count", 32),
            ("attention.indexer.key_length", 128), ("attention.indexer.top_k", 2048),
            ("attention.indexer.pool_size", 4), ("linear_attention.head_count", 64),
            ("linear_attention.head_dimension", 128), ("linear_attention.conv_kernel", 4),
            ("hyper_connection.count", 4), ("hyper_connection.sinkhorn_iterations", 20)
        ]
        for (suffix, value) in integers {
            let key = "glm5-next." + suffix
            guard model.u64Compat(key) == value else { throw fail(key, String(value)) }
        }
        let floats: [(String, Float)] = [
            ("expert_weights_scale", 2.5), ("swiglu_limit", 10),
            ("attention.layer_norm_rms_epsilon", 1e-5),
            ("linear_attention.gate_lower_bound", -5), ("hyper_connection.epsilon", 1e-6)
        ]
        for (suffix, value) in floats {
            let key = "glm5-next." + suffix
            guard let actual = model.f32Compat(key), actual.isFinite,
                  abs(actual - value) <= abs(value) * 1e-6 else {
                throw fail(key, String(value))
            }
        }
        guard model.bool("glm5-next.expert_weights_norm") == true else {
            throw fail("glm5-next.expert_weights_norm", "true")
        }
        guard let array = model.array("glm5-next.layer_types"),
              [GGUFValueType.uint32.rawValue, GGUFValueType.int32.rawValue].contains(array.type),
              let types = model.intArray("glm5-next.layer_types"),
              types == (0..<46).map({ Self.isKDALayer($0) ? 0 : 1 }) else {
            throw fail("glm5-next.layer_types", "46 int32/uint32 entries: three KDA blocks followed by one DSA block, then DSA MTP")
        }
    }
}

public enum GLM53ConfigurationError: Error, Sendable, CustomStringConvertible {
    case invalidMetadata(String, expected: String)
    case invalidTensor(String, expected: String)
    public var description: String {
        switch self {
        case .invalidMetadata(let key, let expected): return "GLM 5.3: \(key) must be \(expected)."
        case .invalidTensor(let key, let expected): return "GLM 5.3 tensor \(key): expected \(expected)."
        }
    }
}
