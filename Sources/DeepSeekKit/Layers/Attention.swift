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

    public func setKVCache(_ cache: MLXArray?) {
        self.kvCache = cache
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
        kv = kv.reshaped([B * S, headDim])
        kv = simulateFP8KV(kv, nopeHeadDim: nopeHeadDim, headDim: headDim)
        kv = kv.reshaped([B, S, headDim])
        // Store the KV cache in bf16: halves the cache footprint (the
        // projection at maxSeqLen above is the single largest live
        // allocation per layer) and lets the SDPA kernel use its bf16
        // fast path. The pre-store math (kvNorm + RoPE + FP8 simulate)
        // stays in f32 for precision; the cast to bf16 only happens at
        // the storage boundary.
        let kvBf16 = kv.asType(.bfloat16)

        if let cache = kvCache {
            kvCache = concatenated([cache, kvBf16], axis: 1)
        } else {
            kvCache = kvBf16
        }
        // For layers trained as pure sliding-window (compressRatio == 0)
        // the reference keeps only the most recent `windowSize` tokens in
        // KV. Without this cap the cache grows unbounded for long
        // contexts and re-uses positions outside training distribution.
        // Layers with compressRatio > 0 would normally keep
        // `windowSize + maxSeqLen/ratio` rows (compressed long-range +
        // raw window) — without the Compressor we can't reconstruct the
        // compressed half, so they currently grow unchecked. Tracked in
        // TODO: restore Compressor / sparse-attn path.
        if compressRatio == 0, let cache = kvCache, cache.shape[1] > windowSize {
            let s = cache.shape[1]
            kvCache = cache[0..., (s - windowSize)..<s, 0...]
        }
        MLX.eval(kvCache!)
        let currentKV = kvCache!   // bf16

        // Run SDPA in bf16 to match the cached K/V dtype and pick up the
        // bf16 kernel. q comes in f32 (rsqrt + RoPE done in f32 above) —
        // cast at the SDPA boundary so RoPE numerics stay full precision.
        let qPerToken = q.reshaped([B, S, nHeads, headDim])
            .transposed(0, 2, 1, 3)
            .asType(.bfloat16)                          // [B, nHeads, S, headDim] bf16
        let kPerToken = currentKV.expandedDimensions(axes: [1])  // [B, 1, S_total, headDim] bf16
        let vPerToken = currentKV.expandedDimensions(axes: [1])

        var o: MLXArray
        if S > 1 {
            let L = currentKV.shape[1]
            var mask = MLXNN.MultiHeadAttention.createAdditiveCausalMask(S, dtype: .bfloat16)
            if L > S {
                let historyMask = MLXArray.zeros([S, L - S]).asType(.bfloat16)
                mask = concatenated([historyMask, mask], axis: 1)
            }
            o = MLXFast.scaledDotProductAttention(queries: qPerToken, keys: kPerToken, values: vPerToken, scale: softmaxScale, mask: mask)
        } else {
            o = MLXFast.scaledDotProductAttention(queries: qPerToken, keys: kPerToken, values: vPerToken, scale: softmaxScale, mask: nil)
        }

        // Cast back to f32 for the inverse-RoPE + woA/woB tail. The
        // subsequent matmuls expect f32 weights as currently produced
        // by `Linear.getDequantizedWeight()`.
        o = o.asType(.float32).transposed(0, 2, 1, 3).reshaped([B * S, nHeads, headDim])

        o = rope.apply(Tensor(array: o, dtype: .f32), startPos: startPos, inverse: true).array

        let perGroupD = nHeads * headDim / nGroups
        let oView = o.reshaped([B, S, nGroups, perGroupD]) // [B, S, nGroups, perGroupD]
        let woAR = woA.getDequantizedWeight().reshaped([nGroups, oLoraRank, perGroupD]) // [nGroups, oLoraRank, perGroupD]
        
        var oGroupResults: [MLXArray] = []
        for g in 0..<nGroups {
            let oG = oView[0..., 0..., g, 0...] // [B, S, perGroupD]
            let wG = woAR[g, 0..., 0...] // [oLoraRank, perGroupD]
            let wG_T = wG.transposed() // [perGroupD, oLoraRank]
            let oR_G = matmul(oG, wG_T) // [B, S, oLoraRank]
            oGroupResults.append(oR_G.expandedDimensions(axes: [2])) // [B, S, 1, oLoraRank]
        }
        let oR = concatenated(oGroupResults, axis: 2) // [B, S, nGroups, oLoraRank]

        let oFlat = oR.reshaped([B * S, nGroups * oLoraRank])
        var result = woB(Tensor(array: oFlat, dtype: .f32)).array
        result = result.reshaped([B, S, wkv.inFeatures])

        return Tensor(array: result, dtype: .f32)
    }

    private func simulateFP8KV(_ x: MLXArray, nopeHeadDim: Int, headDim: Int) -> MLXArray {
        let B_S = x.shape[0]
        
        let nopeSlice = x[0..., 0..<nopeHeadDim]
        let ropeSlice = x[0..., nopeHeadDim..<headDim]
        
        let blocks = nopeHeadDim / 64
        let blocked = nopeSlice.reshaped([B_S, blocks, 64])
        
        let amax = maximum(abs(blocked).max(axes: [-1], keepDims: true), MLXArray(Float32(1e-4)))
        let ln2 = MLXArray(Float32(log(2.0)))
        let log2Amax = log(amax / MLXArray(Float32(448.0))) / ln2
        let scale = exp(ceil(log2Amax) * ln2)
        let scaled = clip(blocked / scale, min: MLXArray(Float32(-448.0)), max: MLXArray(Float32(448.0)))
        
        let absScaled = abs(scaled)
        let log2Abs = log(maximum(absScaled, MLXArray(Float32(1e-9)))) / ln2
        let p = exp(floor(log2Abs) * ln2)
        let m = scaled / p
        let roundedM = round(m * MLXArray(Float32(8.0))) / MLXArray(Float32(8.0))
        let yRounded = roundedM * p
        
        let quantized = MLX.where(absScaled .< MLXArray(Float32(0.001)), MLXArray.zeros(like: scaled), yRounded)
        let unscaled = (quantized * scale).reshaped([B_S, nopeHeadDim])
        
        return concatenated([unscaled, ropeSlice], axis: 1)
    }

    private var dim: Int { wkv.inFeatures }
}
