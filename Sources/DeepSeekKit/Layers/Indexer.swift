import Foundation
import MLX
import MLXNN

/// Learned top-K compressed-position selector. Mirrors the `Indexer`
/// module in `Reference/inference/model.py:380-433`.
///
/// Used only on attention layers with `compressRatio == 4`. Given the
/// per-token query latent `qr`, computes a score per (query, compressed-
/// position) pair via its own small attention path:
///
///   q       = wq_b(qr)                  [B, S, H, D]   (+ RoPE on last `rd`)
///   weights = weights_proj(x) * scale   [B, S, H]
///   score   = einsum("bshd,btd->bsht", q, kv_cache)
///   score   = (relu(score) * weights[..., None]).sum(dim=-2)   [B, S, T]
///   topk    = topk(score, K=index_topk, dim=-1)               [B, S, K]
///
/// The Indexer maintains its OWN compressed KV cache (separate from
/// MLA's): an inner `Compressor` instance compresses x into a
/// `head_dim`-wide latent (index_head_dim, typically 128 — smaller
/// than MLA's 512) using the overlap path (rotate=true in the
/// reference, but Hadamard rotation is skipped here — affects QAT
/// noise only).
///
/// The returned top-K indices are absolute positions in the Indexer's
/// own kv_cache (range [0, T) where T is the current number of
/// compressed entries). MLA uses them to build the additive mask
/// for the compressed columns of its attention KV: allowed iff the
/// column index is in the top-K set.
///
/// Caveats vs trained:
/// - No Hadamard rotation on q (the `rotate=True` path).
/// - No FP4 activation-quant simulation on q.
/// Both are training-time QAT details; quality at inference is
/// preserved to a very good approximation without them.
public final class Indexer {
    public let dim: Int
    public let nHeads: Int               // index_n_heads, typically 64
    public let headDim: Int              // index_head_dim, typically 128
    public let ropeHeadDim: Int          // rope_head_dim, typically 64
    public let indexTopK: Int            // index_topk, typically 512
    public let qLoraRank: Int            // q_lora_rank, typically 1024
    public let compressRatio: Int        // 4
    public let softmaxScale: Float

    public let wqB: Linear              // q_lora_rank → n_heads * head_dim
    public let weightsProj: Linear      // dim → n_heads
    public let compressor: Compressor   // own — independent compressed KV

    /// The Indexer's OWN compressed KV cache (separate from MLA's
    /// `compressedCache`). Used for top-K scoring only — NOT for the
    /// final attention output. Width = `headDim` (index_head_dim,
    /// typically 128), narrower than MLA's compressed cache.
    public var kvCache: MLXArray?       // [B, T, headDim] bf16

    /// RoPE freqs for q's last `ropeHeadDim` channels. Assigned by
    /// parent MLA before first forward.
    public var freqs: Tensor?

    public init(config: ModelConfig,
                wqB: Linear,
                weightsProj: Linear,
                compressor: Compressor) {
        self.dim = config.dim
        self.nHeads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.ropeHeadDim = config.ropeHeadDim
        self.indexTopK = config.indexTopk
        self.qLoraRank = config.qLoraRank
        self.compressRatio = compressor.compressRatio
        self.softmaxScale = pow(Float(self.headDim), -0.5)
        self.wqB = wqB
        self.weightsProj = weightsProj
        self.compressor = compressor
    }

    public func releaseCache() {
        kvCache = nil
        compressor.releaseState()
    }

    /// Forward. Returns top-K compressed-position indices per query
    /// `[B, S, topK]` int32. Returns nil if no compressed entries yet.
    ///
    /// - Parameters:
    ///   - x: hidden state, [B, S, dim] f32
    ///   - qr: q low-rank projection (qNorm(wqA(x))), [B, S, qLoraRank] f32
    ///   - startPos: absolute starting position of this chunk
    public func callAsFunction(_ x: Tensor, qr: Tensor, startPos: Int) -> MLXArray? {
        // Update the inner compressor's KV cache.
        if compressor.freqs == nil { compressor.freqs = freqs }
        if compressor.kvCache == nil {
            // First call: seed the compressor's destination with an empty
            // placeholder. Compressor will write into its slice; we manage
            // the actual cache merging below.
            compressor.kvCache = MLXArray.zeros([1, 1, headDim]).asType(.float32)
        }
        if let newEntries = compressor(x, startPos: startPos) {
            let entries = newEntries.array.asType(.bfloat16)
            if let existing = kvCache {
                kvCache = concatenated([existing, entries], axis: 1)
            } else {
                kvCache = entries
            }
        }
        guard let cache = kvCache, cache.shape[1] > 0 else { return nil }

        let xArr = x.array
        let qrArr = qr.array
        let B = xArr.shape[0]
        let S = xArr.shape[1]
        let T = cache.shape[1]
        let H = nHeads
        let D = headDim
        let rd = ropeHeadDim

        // 1) Build q from qr: [B*S, q_lora_rank] → [B*S, H*D] → [B, S, H, D]
        var q = wqB(Tensor(array: qrArr.reshaped([B * S, qLoraRank]), dtype: .f32)).array
        q = q.reshaped([B, S, H, D])

        // 2) Apply RoPE to the last `rd` channels of q.
        q = applyRoPEOnQ(q, B: B, S: S, H: H, D: D, rd: rd, startPos: startPos)

        // 3) score = einsum("bshd,btd->bsht", q, cache)
        //    Implemented as matmul with broadcast:
        //      q: [B, S, H, D]
        //      cacheT: [B, 1, D, T]
        //      out: [B, S, H, T]
        let qBf = q.asType(.bfloat16)
        let cacheT = cache.expandedDimensions(axes: [1]).transposed(0, 1, 3, 2)
        var indexScore = matmul(qBf, cacheT)            // [B, S, H, T] bf16

        // 4) ReLU then weighted-sum over heads. weights from
        //    weights_proj(x) * (softmax_scale * H^-0.5).
        let weightsScale = MLXArray(Float(softmaxScale * powf(Float(H), -0.5)))
            .asType(.bfloat16)
        let weights = weightsProj(x).array.asType(.bfloat16)
            .reshaped([B, S, H])
            * weightsScale
        // ReLU
        indexScore = maximum(indexScore, MLXArray.zeros(like: indexScore))
        // weights expanded to [B, S, H, 1] then multiply + sum over H axis (=2).
        let weightsExpanded = weights.expandedDimensions(axes: [3])
        let scorePerQuery = (indexScore * weightsExpanded).sum(axes: [2])
        // scorePerQuery: [B, S, T] bf16

        // 5) Causal mask for prefill (startPos == 0).
        //    Compressed position t covers raw positions [t*ratio, (t+1)*ratio).
        //    Query at absolute position s sees t iff (t+1)*ratio ≤ s+1,
        //    i.e. t < (s+1)/ratio  (integer floor).
        let scoreMasked: MLXArray
        if startPos == 0 {
            // mask[s, t] = true (masked) iff t >= (s+1)/ratio
            // Build [S, T] mask as float32 (0 or -large).
            let sIdx = MLXArray((0..<S).map { Int32(($0 + 1) / compressRatio) })
                .reshaped([S, 1])                       // [S, 1]
            let tIdx = MLXArray((0..<T).map { Int32($0) })
                .reshaped([1, T])                       // [1, T]
            let isMasked = (tIdx .>= sIdx).asType(.bfloat16)  // 1 = masked, 0 = visible
            let maskAdd = isMasked * MLXArray(Float(-1e9)).asType(.bfloat16)
            scoreMasked = scorePerQuery + maskAdd
                .expandedDimensions(axes: [0])          // [1, S, T] broadcasts to B
        } else {
            scoreMasked = scorePerQuery
        }

        // 6) Top-K. mlx-swift exposes argSort but not topk directly;
        //    sort ascending then slice the last K and reverse.
        let topK = min(indexTopK, T)
        let sorted = argSort(scoreMasked, axis: -1)     // [B, S, T] int32 ascending
        let topkAsc = sorted[0..., 0..., (T - topK)..<T]
        let topkIdxs = topkAsc[0..., 0..., .stride(by: -1)]  // [B, S, topK]
        return topkIdxs
    }

    /// Apply rotary position embedding to the last `rd` channels of
    /// `q` (shape [B, S, H, D]). Mirrors `apply_rotary_emb(q[..., -rd:],
    /// freqs_cis)` in the Python reference.
    private func applyRoPEOnQ(_ q: MLXArray, B: Int, S: Int, H: Int,
                                D: Int, rd: Int, startPos: Int) -> MLXArray {
        guard let freqs = freqs else { return q }
        let freqSlice = freqs.array[startPos..<(startPos + S)]  // [S, rd/2, 2]

        let nopePart = q[0..., 0..., 0..., 0..<(D - rd)]
        let ropePart = q[0..., 0..., 0..., (D - rd)..<D]
            .reshaped([B, S, H, rd / 2, 2])

        let cosFreq = freqSlice[0..., 0..., 0]
            .reshaped([1, S, 1, rd / 2])
        let sinFreq = freqSlice[0..., 0..., 1]
            .reshaped([1, S, 1, rd / 2])

        let x0 = ropePart[0..., 0..., 0..., 0..., 0]
        let x1 = ropePart[0..., 0..., 0..., 0..., 1]
        let out0 = x0 * cosFreq - x1 * sinFreq
        let out1 = x0 * sinFreq + x1 * cosFreq
        let outRope = stacked([out0, out1], axis: -1)
            .reshaped([B, S, H, rd])

        return concatenated([nopePart, outRope], axis: -1)
    }
}
