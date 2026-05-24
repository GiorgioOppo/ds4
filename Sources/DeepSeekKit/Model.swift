import Foundation
import MLX
import MLXNN

public final class ParallelEmbedding {
    public let vocabSize: Int
    public let dim: Int
    public let weight: Tensor

    public init(vocabSize: Int, dim: Int, weight: Tensor) {
        self.vocabSize = vocabSize; self.dim = dim; self.weight = weight
    }

    public func lookup(_ ids: [Int32]) -> Tensor {
        let idsT = MLXArray(ids)
        let out = weight.array[idsT]
        return Tensor(array: out, dtype: .f32)
    }
}

public final class ParallelHead {
    public let vocabSize: Int
    public let dim: Int
    public let normEps: Float
    public let hcEps: Float
    public let weight: Tensor

    public init(vocabSize: Int, dim: Int, normEps: Float, hcEps: Float, weight: Tensor) {
        self.vocabSize = vocabSize; self.dim = dim
        self.normEps = normEps; self.hcEps = hcEps
        self.weight = weight
    }

    public func callAsFunction(_ x: Tensor,
                                hcFn: Tensor, hcScale: Tensor, hcBase: Tensor,
                                norm: RMSNorm) -> Tensor {
        let xArr = x.array
        let B = xArr.shape[0], S = xArr.shape[1], HC = xArr.shape[2], D = xArr.shape[3]
        let N = B * S
        let hcDim = HC * D

        let xFlat = xArr.reshaped([N, hcDim])

        let rsqrtVal = rsqrt(mean(square(xFlat), axes: [-1], keepDims: true) + normEps)

        // hc_head: xFlat [N, hcDim] · hcFn.T [hcDim, hcMult] → [N, hcMult]
        let mixes = matmul(xFlat, hcFn.array.transposed())
        let mixesScaled = mixes * rsqrtVal
        let pre = sigmoid(mixesScaled * hcScale.array + hcBase.array) + hcEps

        let yArr = (pre.expandedDimensions(axes: [2]) * xArr.reshaped([N, HC, D])).sum(axes: [1])

        let yNorm = norm(Tensor(array: yArr, dtype: .f32)).array.reshaped([B, S, D])

        // Match the reference contract: emit logits only for the last
        // position (sampling expects [B, V]). Per-position logits would
        // be needed for MTP / speculative decoding — left as a TODO.
        let lastTok = yNorm[0..., (S - 1)..., 0...].reshaped([B, D])

        // lm_head: lastTok [B, D] · weight.T [D, V] → [B, V]
        let logits = matmul(lastTok, weight.array.transposed())
        return Tensor(array: logits, dtype: .f32)
    }
}

public final class Transformer {
    public let config: ModelConfig
    public let embed: ParallelEmbedding
    public let layers: [Block]
    public let mtp: [MTPBlock]
    public let norm: RMSNorm
    public let head: ParallelHead

    public let hcHeadFn: Tensor
    public let hcHeadBase: Tensor
    public let hcHeadScale: Tensor

    internal var weightLoader: WeightLoader? = nil

    public init(config: ModelConfig,
                embed: ParallelEmbedding,
                layers: [Block],
                mtp: [MTPBlock],
                norm: RMSNorm,
                head: ParallelHead,
                hcHeadFn: Tensor, hcHeadBase: Tensor, hcHeadScale: Tensor) {
        self.config = config
        self.embed = embed
        self.layers = layers
        self.mtp = mtp
        self.norm = norm
        self.head = head
        self.hcHeadFn = hcHeadFn
        self.hcHeadBase = hcHeadBase
        self.hcHeadScale = hcHeadScale
    }

    public func forward(inputIds: [[Int]], startPos: Int) -> Tensor {
        let B = inputIds.count
        let S = inputIds[0].count

        let flatIds: [Int32] = inputIds.flatMap { $0.map(Int32.init) }

        let h = embed.lookup(flatIds).array
        
        let hc = config.hcMult
        let hExpanded = h.expandedDimensions(axes: [1])
        let hBroadcasted = broadcast(hExpanded, to: [B * S, hc, config.dim])
        var x = hBroadcasted.reshaped([B, S, hc, config.dim])

        // Prefetch layer 0 before we start
        weightLoader?.ensureLayer(0)

        let debugStream = ProcessInfo.processInfo.environment["DEEPSEEK_STREAMING_DEBUG"] != nil
        // Disable layer prefetch during prefill: when MoEFFN bulk-loads
        // the full routed-expert set (N > 1 path), having a second
        // layer's non-experts in flight alongside the current layer's
        // ~5–6 GB of routed experts doubles the peak. Decode (S=1)
        // keeps the prefetch since the streaming working-set is small.
        let allowPrefetch = (S == 1)

        for (k, layer) in layers.enumerated() {
            // Start prefetching next layer in background while current executes
            if allowPrefetch, k + 1 < layers.count {
                weightLoader?.prefetchLayer(k + 1)
            }
            x = layer(Tensor(array: x, dtype: .f32), startPos: startPos, inputIds: flatIds).array
            // Force evaluation before releasing weights
            MLX.eval(x)
            // Release current layer's weights to free RAM
            weightLoader?.releaseLayer(k)
            // Return MLX's internal buffer pool to the system. Without
            // this MLX keeps freed buffers for reuse, and on a tight-
            // RAM system that pool can grow to the size of the largest
            // recent peak — defeating per-expert streaming.
            MLX.GPU.clearCache()

            if debugStream, let loader = weightLoader {
                FileHandle.standardError.write(Data(
                    "[stream] layer \(k) done; cache=\(loader.cachedTensorCount) tensors\n".utf8))
            }
        }

        let logits = head(Tensor(array: x, dtype: .f32), hcFn: hcHeadFn, hcScale: hcHeadScale, hcBase: hcHeadBase, norm: norm)
        
        return logits
    }

    public func releaseCache() {
        for layer in layers {
            layer.attn.setKVCache(nil)
        }
    }

    @discardableResult
    public func rewindKVTo(pos: Int) -> Bool {
        var success = true
        for layer in layers {
            if !layer.attn.rewindKVTo(pos: pos) {
                success = false
            }
        }
        return success
    }

    public func snapshotKVCache() -> KVCacheSnapshot {
        let caches = layers.map { $0.attn.kvCache }
        return KVCacheSnapshot(layerCaches: caches)
    }

    public func canRestoreKVCache(_ snap: KVCacheSnapshot) -> Bool {
        return snap.layerCaches.count == layers.count
    }

    public func restoreKVCache(_ snap: KVCacheSnapshot) {
        guard canRestoreKVCache(snap) else { return }
        for (i, layer) in layers.enumerated() {
            layer.attn.setKVCache(snap.layerCaches[i])
        }
    }
}
