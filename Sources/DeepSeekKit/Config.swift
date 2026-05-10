import Foundation

/// Mirrors `config.json` from the HuggingFace repo.
/// Field names are placeholders until `deepseek-ai/DeepSeek-V4-Pro/config.json`
/// is fetched and inspected — names below follow DeepSeek-V3 conventions plus
/// the new V4 components described in the technical report.
public struct ModelConfig: Codable, Sendable {
    // Standard transformer
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float

    // MoE (DeepSeek-style: shared experts + routed experts, top-k gating)
    public var numRoutedExperts: Int
    public var numSharedExperts: Int
    public var numExpertsPerToken: Int
    public var moeIntermediateSize: Int
    public var firstKDenseLayers: Int

    // V4 hybrid attention
    public var csaCompressionRatio: Int      // Compressed Sparse Attention
    public var hcaCompressionRatio: Int      // Heavily Compressed Attention
    public var hcaLayerStride: Int           // every N layers use HCA instead of CSA

    // Manifold-Constrained Hyper-Connections
    public var mhcRank: Int
    public var mhcExpansion: Int

    public var torchDtype: String?
    public var bosTokenId: Int?
    public var eosTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case numRoutedExperts = "n_routed_experts"
        case numSharedExperts = "n_shared_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case firstKDenseLayers = "first_k_dense_replace"
        case csaCompressionRatio = "csa_compression_ratio"
        case hcaCompressionRatio = "hca_compression_ratio"
        case hcaLayerStride = "hca_layer_stride"
        case mhcRank = "mhc_rank"
        case mhcExpansion = "mhc_expansion"
        case torchDtype = "torch_dtype"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
    }

    public static func load(from url: URL) throws -> ModelConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ModelConfig.self, from: data)
    }

    public func usesHCA(layerIndex: Int) -> Bool {
        guard hcaLayerStride > 0 else { return false }
        return layerIndex % hcaLayerStride == 0
    }

    public func isDense(layerIndex: Int) -> Bool {
        layerIndex < firstKDenseLayers
    }
}
