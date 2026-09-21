import Foundation

/// The V4.1 checkpoint has a different graph and metadata schema from V4.
public struct DeepSeek41Configuration: Sendable {
    public let layerCount = 40
    public let embeddingLength = 5120
    public let vocabularySize = 129280
    public let maximumContextLength = 1048576
    public let headCount = 64
    public let headDimension = 512
    public let queryRank = 1280
    public let outputRank = 1024
    public let expertCount = 384
    public let expertsUsed = 6
    public let expertWidth = 2304
    public let rmsEpsilon: Float = 1e-20
    public static let kvSourceLayers = [2, 8, 14, 20]
    public static let indexSourceLayers = [2, 8, 14, 20, 24, 28, 32, 36]
    public static let engramLayers = [1, 14]
    public static let engramRows: [Int64] = [384006168, 384016682]
    public static func compressionRatio(layer: Int) -> Int { layer < 2 ? 0 : layer < 20 ? 2 : 1 }

    public init(model: GGUFModel) throws {
        guard model.string("general.architecture") == "deepseek41" else {
            throw DeepSeek41Error.invalidMetadata("general.architecture")
        }
        let integers: [(String, UInt64)] = [
            ("vocab_size",129280),("hidden_size",5120),("moe_intermediate_size",2304),
            ("num_hidden_layers",40),("num_attention_heads",64),("num_key_value_heads",1),
            ("head_dim",512),("qk_rope_head_dim",64),("q_lora_rank",1280),("o_lora_rank",1024),
            ("o_groups",8),("n_routed_experts",384),("n_shared_experts",1),("num_experts_per_tok",6),
            ("max_position_embeddings",1048576),("sliding_window",128),("index_n_heads",32),
            ("index_head_dim",128),("index_topk",512),("candidate_source_layer_id",20),
            ("candidate_topk_blocks",2048),("candidate_block_size",8),("hc_mult",4),
            ("hc_sinkhorn_iters",20),("rope_theta",10000),("compress_rope_theta",160000),
            ("engram.compressed_vocab_size",99092)
        ]
        for (suffix, value) in integers {
            let key = "deepseek41." + suffix
            guard model.u64Compat(key) == value else { throw DeepSeek41Error.invalidMetadata(key) }
        }
        let floats: [(String, Float)] = [
            ("rms_norm_eps",1e-20),("hc_eps",1e-6),("swiglu_limit",10),
            ("routed_scaling_factor",1.5),("rope_scaling.factor",16),
            ("rope_scaling.beta_fast",32),("rope_scaling.beta_slow",1),
            ("rope_scaling.original_max_position_embeddings",65536)
        ]
        for (suffix, value) in floats {
            let key = "deepseek41." + suffix
            guard let actual = model.f32Compat(key), actual.isFinite,
                  abs(actual - value) <= abs(value) * 1e-5 else {
                throw DeepSeek41Error.invalidMetadata(key)
            }
        }
        for (suffix, value) in [("scoring_func","sqrtsoftplus"),("hidden_act","silu"),
                                ("topk_method","noaux_tc"),("engram.encoding","e4m3_e8m0_32_row264")] {
            let key = "deepseek41." + suffix
            guard model.string(key) == value else { throw DeepSeek41Error.invalidMetadata(key) }
        }
        guard model.bool("deepseek41.norm_topk_prob") == true else {
            throw DeepSeek41Error.invalidMetadata("deepseek41.norm_topk_prob")
        }
        let arrays: [(String, [Int64])] = [
            ("kv_source_layer_ids",Self.kvSourceLayers.map(Int64.init)),
            ("index_source_layer_ids",Self.indexSourceLayers.map(Int64.init)),
            ("engram.layer_ids",Self.engramLayers.map(Int64.init)),("engram.rows",Self.engramRows),
            ("compress_ratios",(0..<40).map { Int64(Self.compressionRatio(layer: $0)) })
        ]
        for (suffix, values) in arrays {
            let key = "deepseek41." + suffix
            guard model.intArray(key) == values else { throw DeepSeek41Error.invalidMetadata(key) }
        }
    }
}

public enum DeepSeek41Error: Error, Sendable, CustomStringConvertible {
    case invalidMetadata(String)
    case invalidTensor(String)
    case engram(String)
    public var description: String {
        switch self {
        case .invalidMetadata(let key): return "DeepSeek V4.1: metadato non supportato: \(key)."
        case .invalidTensor(let key): return "DeepSeek V4.1: tensore non valido: \(key)."
        case .engram(let message): return "DeepSeek V4.1 Engram: \(message)"
        }
    }
}
