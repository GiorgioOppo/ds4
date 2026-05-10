import Foundation
import Metal

/// Per-layer KV cache. Layout matches model.py:473
///
///   kv_cache[:, :window]  → sliding-window ring buffer (window_size = 128)
///   kv_cache[:, window:]  → compressed KV slots (max_seq_len / compress_ratio)
///
/// For layers with compress_ratio == 0 (pure window), only the first window
/// region is allocated.
public final class KVCache {
    public var data: Tensor        // [maxBatch, kvLen, headDim]
    public let windowSize: Int
    public let compressRatio: Int
    public let headDim: Int

    public init(maxBatch: Int, headDim: Int, windowSize: Int, compressRatio: Int, maxSeqLen: Int) {
        let compressed = compressRatio == 0 ? 0 : maxSeqLen / compressRatio
        let kvLen = windowSize + compressed
        self.data = Tensor.empty(shape: [maxBatch, max(kvLen, 1), headDim], dtype: .f32)
        self.windowSize = windowSize
        self.compressRatio = compressRatio
        self.headDim = headDim
    }
}

/// Bank of caches, one per layer.
public final class CacheBank {
    public let layers: [KVCache]

    public init(config: ModelConfig) {
        self.layers = (0..<config.nLayers).map { i in
            KVCache(maxBatch: config.maxBatchSize,
                    headDim: config.headDim,
                    windowSize: config.windowSize,
                    compressRatio: config.compressRatios[i],
                    maxSeqLen: config.maxSeqLen)
        }
    }
}
