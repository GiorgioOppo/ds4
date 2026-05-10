import Foundation
import Metal

/// Per-layer KV cache. With CSA/HCA the K/V stored here are the *compressed*
/// summaries, not the raw projections — so capacity is much smaller than
/// (numKVHeads * headDim * maxPositionEmbeddings). Exact compressed dims
/// come from the config once the kernels are implemented.
public final class KVCache {
    public var k: Tensor
    public var v: Tensor
    public var length: Int = 0

    public init(maxTokens: Int, kvHeads: Int, headDim: Int) {
        self.k = Tensor.empty(shape: [maxTokens, kvHeads, headDim], dtype: .f32)
        self.v = Tensor.empty(shape: [maxTokens, kvHeads, headDim], dtype: .f32)
    }
}

/// Container for one cache per decoder layer.
public final class CacheBank {
    public let layers: [KVCache]
    public init(numLayers: Int, maxTokens: Int, kvHeads: Int, headDim: Int) {
        self.layers = (0..<numLayers).map { _ in KVCache(maxTokens: maxTokens, kvHeads: kvHeads, headDim: headDim) }
    }
}
