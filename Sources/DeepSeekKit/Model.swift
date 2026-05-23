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
        let hcMult = HC

        let xFlat = xArr.reshaped([N, hcDim])

        let rsqrtVal = rsqrt(mean(square(xFlat), axes: [-1], keepDims: true) + normEps)

        let lin = Linear(inFeatures: hcDim, outFeatures: hcMult,
                         weight: hcFn, scale: nil,
                         castOutputToBF16: false)
        let mixes = lin(Tensor(array: xFlat, dtype: .f32)).array
        
        let mixesScaled = mixes * rsqrtVal
        
        let pre = sigmoid(mixesScaled * hcScale.array + hcBase.array) + hcEps
        
        let yArr = (pre.expandedDimensions(axes: [2]) * xArr.reshaped([N, HC, D])).sum(axes: [1])
        
        let yNorm = norm(Tensor(array: yArr, dtype: .f32)).array.reshaped([B, S, D])
        
        let lastTok = yNorm[0..., (S-1)..., 0...]
        let lastTokReshaped = lastTok.reshaped([B, D])
        
        let lmHead = Linear(inFeatures: dim, outFeatures: vocabSize,
                            weight: weight, scale: nil,
                            castOutputToBF16: false)
        let logits = lmHead(Tensor(array: lastTokReshaped, dtype: .f32))
        return logits
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

        for (k, layer) in layers.enumerated() {
            weightLoader?.ensureLayer(k)
            x = layer(Tensor(array: x, dtype: .f32), startPos: startPos, inputIds: flatIds).array
            weightLoader?.releaseLayer(k)
        }

        let logits = head(Tensor(array: x, dtype: .f32), hcFn: hcHeadFn, hcScale: hcHeadScale, hcBase: hcHeadBase, norm: norm)
        
        return logits
    }

    public func releaseCache() {
        // MLX doesn't have explicit buffer commits, but we can clear things if needed.
    }

    @discardableResult
    public func rewindKVTo(pos: Int) -> Bool {
        // Just keeping the API signature.
        return true
    }
}
