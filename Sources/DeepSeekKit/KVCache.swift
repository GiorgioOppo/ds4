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
    /// Optional so `release()` can free the underlying `MTLBuffer`. The
    /// next access via `ensureData()` re-allocates a fresh buffer.
    public private(set) var data: Tensor?  // [maxBatch, kvLen, headDim]
    public let windowSize: Int
    public let compressRatio: Int
    public let headDim: Int

    private let dataShape: [Int]
    private let dataDType: DType = .f32

    public init(maxBatch: Int, headDim: Int, windowSize: Int, compressRatio: Int, maxSeqLen: Int) {
        let compressed = compressRatio == 0 ? 0 : maxSeqLen / compressRatio
        let kvLen = windowSize + compressed
        let shape = [maxBatch, max(kvLen, 1), headDim]
        self.data = Tensor.empty(shape: shape, dtype: .f32)
        self.windowSize = windowSize
        self.compressRatio = compressRatio
        self.headDim = headDim
        self.dataShape = shape
    }

    public func ensureData() -> Tensor {
        if let t = data { return t }
        let t = Tensor.empty(shape: dataShape, dtype: dataDType)
        data = t
        return t
    }

    /// Drop the cache buffer. ARC frees the underlying `MTLBuffer`,
    /// returning unified-memory pages to the system. Next call to
    /// `ensureData()` lazy-reallocates.
    public func release() {
        data = nil
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

    /// Release every layer's cache buffer in one shot.
    public func release() {
        for c in layers { c.release() }
    }
}
