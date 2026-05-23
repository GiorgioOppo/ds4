import Foundation
import MLX
import MLXNN
import MLXFast

public final class MLA {
    public let layerId: Int
    public let nHeads: Int
    public let headDim: Int
    public let ropeHeadDim: Int
    public let nopeHeadDim: Int
    public let nGroups: Int
    public let oLoraRank: Int
    public let qLoraRank: Int
    public let windowSize: Int
    public let compressRatio: Int
    public let eps: Float
    public let softmaxScale: Float

    public let wqA: Linear
    public let qNorm: RMSNorm
    public let wqB: Linear
    public let wkv: Linear
    public let kvNorm: RMSNorm
    public let woA: Linear
    public let woB: Linear
    public let attnSink: Tensor

    public let rope: RoPE
    
    public private(set) var kvCache: MLXArray?

    public init(config: ModelConfig, layerId: Int,
                wqA: Linear, qNorm: RMSNorm, wqB: Linear,
                wkv: Linear, kvNorm: RMSNorm,
                woA: Linear, woB: Linear,
                attnSink: Tensor,
                rope: RoPE,
                kvCache: Tensor?) {
        self.layerId = layerId
        self.nHeads = config.nHeads
        self.headDim = config.headDim
        self.ropeHeadDim = config.ropeHeadDim
        self.nopeHeadDim = config.headDim - config.ropeHeadDim
        self.nGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.qLoraRank = config.qLoraRank
        self.windowSize = config.windowSize
        self.compressRatio = config.compressRatios[layerId]
        self.eps = config.normEps
        self.softmaxScale = pow(Float(config.headDim), -0.5)
        
        self.wqA = wqA
        self.qNorm = qNorm
        self.wqB = wqB
        self.wkv = wkv
        self.kvNorm = kvNorm
        self.woA = woA
        self.woB = woB
        self.attnSink = attnSink
        self.rope = rope
    }

    public func releaseCache() {
        kvCache = nil
    }

    @discardableResult
    public func rewindKVTo(pos: Int) -> Bool {
        if let cache = kvCache, pos <= cache.shape[1] {
            if pos == 0 {
                kvCache = nil
            } else {
                kvCache = cache[0..., 0..<pos, 0...]
            }
            return true
        }
        return false
    }

    public func callAsFunction(_ xIn: Tensor, startPos: Int) -> Tensor {
        let x = xIn.array
        let B = x.shape[0]
        let S = x.shape[1]

        var qrFlat = qNorm(wqA(Tensor(array: x.reshaped([B * S, x.shape[2]]), dtype: .f32))).array
        qrFlat = qrFlat.reshaped([B, S, qLoraRank])
        var q = wqB(Tensor(array: qrFlat.reshaped([B * S, qLoraRank]), dtype: .f32)).array
        q = q.reshaped([B * S, nHeads, headDim])

        let qRsqrt = rsqrt(mean(square(q), axes: [-1], keepDims: true) + eps)
        q = q * qRsqrt
        
        q = rope.apply(Tensor(array: q.reshaped([B * S, nHeads, headDim]), dtype: .f32), startPos: startPos, inverse: false).array

        var kv = kvNorm(wkv(Tensor(array: x.reshaped([B * S, x.shape[2]]), dtype: .f32))).array
        kv = rope.apply(Tensor(array: kv.reshaped([B * S, 1, headDim]), dtype: .f32), startPos: startPos, inverse: false).array
        kv = kv.reshaped([B, S, headDim])

        if let cache = kvCache {
            kvCache = concatenated([cache, kv], axis: 1)
        } else {
            kvCache = kv
        }
        let currentKV = kvCache!

        let qPerToken = q.reshaped([B, S, nHeads, headDim]).transposed(0, 2, 1, 3) // [B, nHeads, S, headDim]
        let kPerToken = currentKV.expandedDimensions(axes: [1]) // [B, 1, S_total, headDim]
        let vPerToken = currentKV.expandedDimensions(axes: [1])

        var o: MLXArray
        if S > 1 {
            let mask = MLXNN.MultiHeadAttention.createAdditiveCausalMask(S, dtype: q.dtype)
            o = MLXFast.scaledDotProductAttention(queries: qPerToken, keys: kPerToken, values: vPerToken, scale: softmaxScale, mask: mask)
        } else {
            o = MLXFast.scaledDotProductAttention(queries: qPerToken, keys: kPerToken, values: vPerToken, scale: softmaxScale, mask: nil)
        }

        o = o.transposed(0, 2, 1, 3).reshaped([B * S, nHeads, headDim])

        o = rope.apply(Tensor(array: o, dtype: .f32), startPos: startPos, inverse: true).array

        let perGroupD = nHeads * headDim / nGroups
        let oView = o.reshaped([B, S, nGroups, perGroupD])
        let woAR = woA.weight.array.reshaped([nGroups, oLoraRank, perGroupD])
        
        let oViewExpanded = oView.expandedDimensions(axes: [3])
        let woARExpanded = woAR.expandedDimensions(axes: [0, 1])
        
        let oR = (oViewExpanded * woARExpanded).sum(axes: [-1])

        let oFlat = oR.reshaped([B * S, nGroups * oLoraRank])
        var result = woB(Tensor(array: oFlat, dtype: .f32)).array
        result = result.reshaped([B, S, wkv.inFeatures])

        return Tensor(array: result, dtype: .f32)
    }
}
