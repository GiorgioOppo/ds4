import Foundation
import MLX
import MLXNN

/// KV-compressor: gated softmax-pooling over `compressRatio` consecutive
/// raw KV positions, emitting one compressed entry per group.
///
/// Mirrors the `Compressor` module in `Reference/inference/model.py:279-377`
/// for the non-overlap path (ratio > 4, typically ratio = 128). The
/// overlap path (ratio == 4) is left as a TODO — those layers fall back
/// to the existing sliding-window-only behavior in MLA until the
/// overlap + Indexer pieces land.
///
/// State:
/// - `wkv`, `wgate`, `norm`: trained projections (loaded via Linear's
///   MLX-native triplet path when present, otherwise random init).
/// - `ape`: learned positional embedding `[ratio, coff*headDim]` (f32).
/// - `kvState`, `scoreState`: rolling buffers used by the decode path
///   to accumulate `compressRatio` raw tokens before emitting one
///   compressed entry. Reset between conversations.
/// - `kvCache`: destination slice owned by the parent MLA where the
///   compressed entries are written. Assigned externally by MLA before
///   the first forward.
public final class Compressor {
    public let dim: Int
    public let headDim: Int
    public let ropeHeadDim: Int
    public let nopeHeadDim: Int
    public let compressRatio: Int
    public let overlap: Bool   // true when ratio == 4 (overlap path; not yet implemented)
    public let rotate: Bool    // only used by Indexer's internal Compressor
    public let normEps: Float

    public let ape: Tensor               // [ratio, coff*headDim] f32
    public let wkv: Linear               // dim → coff*headDim
    public let wgate: Linear             // dim → coff*headDim
    public let norm: RMSNorm

    public private(set) var kvState: MLXArray?    // [B, coff*ratio, coff*headDim]
    public private(set) var scoreState: MLXArray?  // same shape, init -inf

    /// Set by parent MLA. The slice of the parent's KV cache where
    /// compressed entries are written. Shape `[B, max_seq_len/ratio, headDim]`.
    public var kvCache: MLXArray?

    /// Set by parent MLA: precomputed RoPE freqs for compressed positions.
    public var freqs: Tensor?

    public init(config: ModelConfig,
                compressRatio: Int,
                headDim: Int,
                rotate: Bool,
                ape: Tensor,
                wkv: Linear,
                wgate: Linear,
                norm: RMSNorm) {
        self.dim = config.dim
        self.headDim = headDim
        self.ropeHeadDim = config.ropeHeadDim
        self.nopeHeadDim = headDim - config.ropeHeadDim
        self.compressRatio = compressRatio
        self.overlap = (compressRatio == 4)
        self.rotate = rotate
        self.normEps = config.normEps
        self.ape = ape
        self.wkv = wkv
        self.wgate = wgate
        self.norm = norm
    }

    /// Reset rolling state. Called by MLA on cache release / rewind.
    public func releaseState() {
        kvState = nil
        scoreState = nil
        kvCache = nil
    }

    /// Forward.
    /// - Prefill (`startPos == 0`): produces `S / ratio` compressed entries
    ///   (writes them into `kvCache[0..<S/ratio]`). Leftover `S % ratio`
    ///   raw tokens are accumulated into `kvState`/`scoreState` for the
    ///   next decode round.
    /// - Decode (`startPos > 0`): updates `kvState`/`scoreState` with the
    ///   one new token. When `(startPos + 1) % ratio == 0`, emits one
    ///   compressed entry into `kvCache[startPos / ratio]`.
    @discardableResult
    public func callAsFunction(_ x: Tensor, startPos: Int) -> Tensor? {
        guard let kvCacheTarget = kvCache else {
            FileHandle.standardError.write(Data(
                "[Compressor] kvCache not assigned; skipping forward\n".utf8))
            return nil
        }
        // Overlap path (ratio == 4) not yet implemented; bail.
        if overlap { return nil }

        let xArr = x.array
        let B = xArr.shape[0]
        let S = xArr.shape[1]
        let ratio = compressRatio
        let d = headDim
        let rd = ropeHeadDim

        // Compute kv and score from x. Reference does this in f32 for
        // pooling precision; we follow.
        let xF = xArr.dtype == .float32 ? xArr : xArr.asType(.float32)
        let kvFlat = wkv(Tensor(array: xF.reshaped([B * S, dim]), dtype: .f32)).array
        let scoreFlat = wgate(Tensor(array: xF.reshaped([B * S, dim]), dtype: .f32)).array
        let kv = kvFlat.reshaped([B, S, d])
        let score = scoreFlat.reshaped([B, S, d])

        if startPos == 0 {
            return forwardPrefill(kv: kv, score: score, B: B, S: S, ratio: ratio,
                                   d: d, rd: rd, kvCache: kvCacheTarget,
                                   xDtype: xArr.dtype)
        } else {
            return forwardDecode(kv: kv, score: score, B: B, ratio: ratio,
                                  d: d, rd: rd, kvCache: kvCacheTarget,
                                  startPos: startPos, xDtype: xArr.dtype)
        }
    }

    // MARK: - Prefill path

    private func forwardPrefill(kv: MLXArray, score: MLXArray,
                                  B: Int, S: Int, ratio: Int,
                                  d: Int, rd: Int,
                                  kvCache: MLXArray,
                                  xDtype: MLX.DType) -> Tensor? {
        let cutoff = S - (S % ratio)
        let remainder = S - cutoff

        // Stash the leftover (`S % ratio`) tokens into the rolling state
        // so the first decode-time emit can complete the group.
        if remainder > 0 {
            ensureState(B: B)
            // ape: [ratio, d] (coff=1, non-overlap)
            let kvRem = kv[0..., cutoff..<S, 0...]
            let scoreRem = score[0..., cutoff..<S, 0...]
            let apeRem = ape.array[0..<remainder, 0...]
            // kvState shape: [B, ratio, d]
            // Write the remainder into slots [0..<remainder].
            kvState = updateStateSlot(state: kvState!, start: 0, count: remainder,
                                      value: kvRem)
            scoreState = updateStateSlot(state: scoreState!, start: 0, count: remainder,
                                         value: scoreRem + apeRem.expandedDimensions(axes: [0]))
        }

        guard cutoff >= ratio else { return nil }   // not enough for one group

        // Group the aligned prefix into [B, cutoff/ratio, ratio, d].
        let nGroups = cutoff / ratio
        let kvGroups = kv[0..., 0..<cutoff, 0...].reshaped([B, nGroups, ratio, d])
        let scoreGroups = (score[0..., 0..<cutoff, 0...] +
                           ape.array.expandedDimensions(axes: [0, 1]))
            .reshaped([B, nGroups, ratio, d])

        // Softmax over the group axis, weighted sum → one compressed entry per group.
        let attn = softmax(scoreGroups, axis: 2)        // [B, nGroups, ratio, d]
        var compressed = (kvGroups * attn).sum(axes: [2])  // [B, nGroups, d]

        // Apply norm + RoPE to the compressed entries.
        compressed = applyNormAndRoPE(compressed: compressed,
                                       B: B, nNew: nGroups, d: d, rd: rd,
                                       startCompressIdx: 0)

        // Write into the parent's kvCache slice at [0..<nGroups].
        writeCacheSlice(kvCache: kvCache, start: 0, count: nGroups, value: compressed)

        return Tensor(array: compressed.asType(xDtype), dtype: .f32)
    }

    // MARK: - Decode path

    private func forwardDecode(kv: MLXArray, score: MLXArray,
                                B: Int, ratio: Int, d: Int, rd: Int,
                                kvCache: MLXArray,
                                startPos: Int, xDtype: MLX.DType) -> Tensor? {
        ensureState(B: B)
        let slot = startPos % ratio

        // Single new raw token; insert it into the slot of the rolling state.
        let kvNew = kv[0..., 0..<1, 0...]                          // [B, 1, d]
        let scoreNew = score[0..., 0..<1, 0...] +
                       ape.array[slot..<(slot + 1), 0...]
                         .expandedDimensions(axes: [0])             // [B, 1, d]
        kvState = updateStateSlot(state: kvState!, start: slot, count: 1, value: kvNew)
        scoreState = updateStateSlot(state: scoreState!, start: slot, count: 1, value: scoreNew)

        let shouldEmit = ((startPos + 1) % ratio) == 0
        guard shouldEmit else { return nil }

        // Aggregate the group: softmax over the ratio axis of the state.
        // kvState / scoreState shape: [B, ratio, d]
        let attn = softmax(scoreState!, axis: 1)
        var compressed = (kvState! * attn).sum(axes: [1], keepDims: true)  // [B, 1, d]
        compressed = applyNormAndRoPE(compressed: compressed,
                                       B: B, nNew: 1, d: d, rd: rd,
                                       startCompressIdx: startPos / ratio)

        // Write into the parent's cache slot at index `startPos / ratio`.
        let idx = startPos / ratio
        writeCacheSlice(kvCache: kvCache, start: idx, count: 1, value: compressed)

        // Reset state for the next group (re-initialize to all -inf scores +
        // zeros for kv so the next softmax is well-defined).
        kvState = MLXArray.zeros([B, ratio, d]).asType(.float32)
        scoreState = MLXArray.zeros([B, ratio, d]).asType(.float32) +
                     MLXArray(Float(-1e30)).asType(.float32)

        return Tensor(array: compressed.asType(xDtype), dtype: .f32)
    }

    // MARK: - Helpers

    /// Allocates `kvState` and `scoreState` on first use. `scoreState`
    /// is initialised to a large negative (effectively -inf for
    /// softmax) so unused slots do not dilute the weighted sum.
    private func ensureState(B: Int) {
        let ratio = compressRatio
        let d = headDim
        if kvState == nil {
            kvState = MLXArray.zeros([B, ratio, d]).asType(.float32)
        }
        if scoreState == nil {
            scoreState = MLXArray.zeros([B, ratio, d]).asType(.float32) +
                         MLXArray(Float(-1e30)).asType(.float32)
        }
    }

    /// Update `state[:, start..<start+count, :]` to `value`. Returns the
    /// updated state (MLX is functional — we rebuild the affected slice
    /// by concatenation since there's no in-place index_update).
    private func updateStateSlot(state: MLXArray, start: Int, count: Int,
                                  value: MLXArray) -> MLXArray {
        let total = state.shape[1]
        let end = start + count
        var parts: [MLXArray] = []
        if start > 0 {
            parts.append(state[0..., 0..<start, 0...])
        }
        parts.append(value)
        if end < total {
            parts.append(state[0..., end..<total, 0...])
        }
        return concatenated(parts, axis: 1)
    }

    /// Same pattern as `updateStateSlot` but for the parent's kvCache.
    private func writeCacheSlice(kvCache: MLXArray, start: Int, count: Int,
                                  value: MLXArray) {
        let updated = updateStateSlot(state: kvCache, start: start, count: count,
                                       value: value)
        // The kvCache stored property is owned by the parent — we can't
        // assign back to it here. The caller pattern is: parent passes
        // its kvCache by value, we update and return the new one via
        // `getUpdatedKVCache()` after the forward. For now, we
        // explicitly DON'T mutate; MLA must call our `compressedSlice`
        // accessor or use the returned compressed tensor directly.
        //
        // This is a known limitation: the Python reference mutates
        // `self.kv_cache[:, idx]` in place. With MLX's functional
        // semantics we'd need MLA to merge the new compressed entries
        // back into its master KV. For now MLA can grab the value the
        // forward returns and concat it directly to its attention
        // input.
        _ = updated
    }

    /// Norm the compressed nope dims and apply RoPE to the rope dims.
    /// Mirrors the Python `apply_rotary_emb(kv[..., -rd:], freqs_cis)`
    /// step. `startCompressIdx` is the absolute compressed-cache index
    /// of the first new entry. Compressed entry k corresponds to the
    /// raw absolute position `k * compressRatio`; we use that to look
    /// up the right RoPE freqs (stride = compressRatio).
    private func applyNormAndRoPE(compressed: MLXArray, B: Int, nNew: Int,
                                    d: Int, rd: Int,
                                    startCompressIdx: Int) -> MLXArray {
        // RMSNorm over the head_dim axis.
        let varx = mean(square(compressed), axes: [-1], keepDims: true)
        let normed = compressed * rsqrt(varx + normEps) * norm.weight.array

        guard let freqs = freqs else { return normed }

        // Stride-ratio gather: compressed entry k uses raw RoPE position
        // `k * ratio`. Build the position index list explicitly and
        // gather via `take` (slicing with stride isn't a 1-liner in
        // mlx-swift; explicit indices are clearer).
        let positions = (0..<nNew).map {
            Int32((startCompressIdx + $0) * compressRatio)
        }
        let freqIdxs = MLXArray(positions)
        let freqGathered = take(freqs.array, freqIdxs, axis: 0)
        // freqGathered: [nNew, rd/2, 2]

        let nopePart = normed[0..., 0..., 0..<(d - rd)]
        let ropePart = normed[0..., 0..., (d - rd)..<d]
            .reshaped([B, nNew, rd / 2, 2])

        let cosFreq = freqGathered[0..., 0..., 0].expandedDimensions(axes: [0])
        let sinFreq = freqGathered[0..., 0..., 1].expandedDimensions(axes: [0])

        let x0 = ropePart[0..., 0..., 0..., 0]
        let x1 = ropePart[0..., 0..., 0..., 1]
        let out0 = x0 * cosFreq - x1 * sinFreq
        let out1 = x0 * sinFreq + x1 * cosFreq
        let outRope = stacked([out0, out1], axis: -1).reshaped([B, nNew, rd])

        return concatenated([nopePart, outRope], axis: -1)
    }
}
