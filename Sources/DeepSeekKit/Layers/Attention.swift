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
    public let maxSeqLen: Int
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

    /// Optional KV-compressor for layers with `compressRatio > 0`.
    /// Aggregates `compressRatio` consecutive raw tokens into one
    /// "compressed memory" entry; the entries are concatenated to the
    /// raw sliding window as additional keys/values at attention time,
    /// extending effective context past the trained `windowSize` while
    /// staying in-distribution. Nil for `compressRatio == 0` layers.
    public var compressor: Compressor? = nil

    /// Raw sliding-window KV cache. Same semantics as the previous
    /// `kvCache` (per-token KV writes + sliding window mask at
    /// attention time). [B, S_window, headDim] bf16.
    public private(set) var kvCache: MLXArray?

    /// Compressed long-range KV cache. Populated by `compressor` —
    /// one entry per `compressRatio` raw tokens. [B, S_compressed,
    /// headDim] bf16. Nil until the first compressed entry is emitted.
    public private(set) var compressedCache: MLXArray?

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
        self.maxSeqLen = config.maxSeqLen
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
        compressedCache = nil
        compressor?.releaseState()
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
        // NOTE: previous revisions trimmed the cache here to enforce
        // a per-layer cap (windowSize for ratio==0, windowSize+maxSeq
        // /ratio for ratio>0). That broke prefill: when the prompt
        // exceeded the cap, the early queries lost their own positions
        // from the key set and the model collapsed to repeating a
        // single token. Restoring unbounded growth pending a proper
        // sliding-window *mask* (instead of cache truncation) for
        // ratio==0 layers. Memory cost: O(seq_len × head_dim × bf16)
        // per layer — bounded by maxSeqLen.
        MLX.eval(kvCache!)
        let currentKV = kvCache!   // bf16

        // Invoke the Compressor (if present) to emit / update compressed
        // long-range KV entries. Each entry aggregates `compressRatio`
        // consecutive raw tokens via softmax-weighted pooling. Wired
        // only for layers with `compressRatio > 0` (and the non-overlap
        // path, i.e. ratio != 4 for now). The compressed entries are
        // concatenated to the raw sliding window as additional keys
        // at SDPA time, restoring the architectural "memory" past the
        // trained window.
        if let comp = compressor {
            // Make sure the Compressor has its RoPE freqs (gather at
            // stride=ratio from MLA's full freqs table). Set once.
            if comp.freqs == nil { comp.freqs = rope.freqs }
            // xIn is [B, S, dim] — exactly what Compressor expects.
            if let newCompressed = comp(xIn, startPos: startPos) {
                let newCompressedBf16 = newCompressed.array.asType(.bfloat16)
                if let existing = compressedCache {
                    compressedCache = concatenated(
                        [existing, newCompressedBf16], axis: 1)
                } else {
                    compressedCache = newCompressedBf16
                }
                MLX.eval(compressedCache!)
            }
        }

        // Sliding-window attention.
        //
        // The model was trained with `windowSize` = 128 raw KV plus the
        // compressed long-range path on ratio>0 layers (Compressor +
        // Indexer + attention sink). The MLX port does not implement
        // the compressed path; we keep only the sliding window, which
        // is in-distribution for layers with `compressRatio == 0` and
        // a degradation (no long-range context) for ratio>0 layers —
        // strictly better than vanilla SDPA on the full unbounded KV,
        // which puts the model at positions it never saw at training
        // time and collapses generation onto regurgitated prompt
        // tokens.
        //
        // We slice the cache to the in-window range so SDPA only
        // computes over the keys that can actually contribute; an
        // additive mask enforces the per-query window + causal limits
        // within that slice.
        let cacheLen = currentKV.shape[1]
        let attendStart = max(0, startPos - (windowSize - 1))
        let rawSlice = attendStart > 0
            ? currentKV[0..., attendStart..<cacheLen, 0...]
            : currentKV
        let Lraw = rawSlice.shape[1]
        let Lcomp = compressedCache?.shape[1] ?? 0

        // Full attention KV is the compressed long-range entries
        // followed by the raw sliding-window slice. Order matters: the
        // mask construction below assumes compressed columns come
        // first.
        let fullKV: MLXArray
        if Lcomp > 0, let comp = compressedCache {
            fullKV = concatenated([comp, rawSlice], axis: 1)
        } else {
            fullKV = rawSlice
        }
        let L = fullKV.shape[1]

        // Run SDPA in bf16 to match the cached K/V dtype and pick up the
        // bf16 kernel. q comes in f32 (rsqrt + RoPE done in f32 above) —
        // cast at the SDPA boundary so RoPE numerics stay full precision.
        let qPerToken = q.reshaped([B, S, nHeads, headDim])
            .transposed(0, 2, 1, 3)
            .asType(.bfloat16)                          // [B, nHeads, S, headDim] bf16
        let kPerToken = fullKV.expandedDimensions(axes: [1])  // [B, 1, L, headDim] bf16
        let vPerToken = fullKV.expandedDimensions(axes: [1])

        // Build the additive mask of shape (S, L).
        //
        // Layout:
        //   columns [0..Lcomp)        → compressed entries (always
        //                                allowed for all queries — they
        //                                live in the older-than-window
        //                                past). Causality fudge of up
        //                                to `compressRatio - 1` tokens
        //                                is accepted as a quality
        //                                trade-off.
        //   columns [Lcomp..L)        → raw sliding-window KV. Standard
        //                                causal + window mask applies.
        let queryPos = MLXArray((0..<S).map { Int32(startPos + $0) })
            .reshaped([S, 1])
        let keyPosRaw = MLXArray((0..<Lraw).map { Int32(attendStart + $0) })
            .reshaped([1, Lraw])
        let diffRaw = queryPos - keyPosRaw                             // [S, Lraw]
        let causal = (diffRaw .>= 0).asType(.int8)
        let inWin  = (diffRaw .< Int32(windowSize)).asType(.int8)
        let allowedRawF = (causal * inWin).asType(.bfloat16)           // 1.0 / 0.0
        let maskRaw = (MLXArray(Float(1.0)) - allowedRawF).asType(.bfloat16)
            * MLXArray(Float(-1e9)).asType(.bfloat16)
        let mask: MLXArray
        if Lcomp > 0 {
            let maskComp = MLXArray.zeros([S, Lcomp]).asType(.bfloat16)
            mask = concatenated([maskComp, maskRaw], axis: 1)
        } else {
            mask = maskRaw
        }

        let o0 = MLXFast.scaledDotProductAttention(
            queries: qPerToken, keys: kPerToken, values: vPerToken,
            scale: softmaxScale, mask: mask)

        // Cast back to f32 for the inverse-RoPE + woA/woB tail. The
        // subsequent matmuls expect f32 weights as currently produced
        // by `Linear.getDequantizedWeight()`.
        var o = o0.asType(.float32).transposed(0, 2, 1, 3).reshaped([B * S, nHeads, headDim])

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
