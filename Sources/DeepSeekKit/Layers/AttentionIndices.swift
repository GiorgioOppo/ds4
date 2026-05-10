import Foundation

/// Host-side helpers that produce the deterministic top-k index tables
/// consumed by the sparse attention kernel. Mirrors `get_window_topk_idxs`
/// and `get_compress_topk_idxs` from
/// Reference/inference/model.py:254-276.
///
/// Both functions return Int32 arrays of shape `[bsz, seqlen, K]` (decode
/// uses seqlen = 1) suitable for upload as a `Tensor` of dtype `.i32`.
public enum AttentionIndices {

    /// Sliding window of size `windowSize` over the KV ring buffer.
    /// Returns indices `[bsz, seqlen, windowSize]` into the ring buffer.
    /// `-1` marks padding (positions before the start of the sequence).
    public static func slidingWindow(windowSize: Int, batch: Int,
                                      seqlen: Int, startPos: Int) -> [Int32] {
        let total = batch * seqlen * windowSize
        var out = [Int32](repeating: 0, count: total)

        var matrix = [Int32](repeating: -1, count: seqlen * windowSize)

        if startPos >= windowSize - 1 {
            // Decode after the ring has filled. The reference produces a
            // single-row [windowSize] table that wraps around the ring.
            // We replicate that row across all `seqlen` slots (decode case
            // uses seqlen = 1).
            let sp = startPos % windowSize
            for j in 0..<(windowSize - sp - 1) { matrix[j] = Int32(sp + 1 + j) }
            for j in 0..<(sp + 1) { matrix[windowSize - sp - 1 + j] = Int32(j) }
            for s in 1..<seqlen {
                for j in 0..<windowSize { matrix[s * windowSize + j] = matrix[j] }
            }
        } else if startPos > 0 {
            // Mid-ring: indices 0..startPos+1 are valid, rest is -1.
            for j in 0...startPos { matrix[j] = Int32(j) }
            // remaining [-1] already set
            for s in 1..<seqlen {
                for j in 0..<windowSize { matrix[s * windowSize + j] = matrix[j] }
            }
        } else {
            // Prefill: matrix[s, j] = max(s - windowSize + 1, 0) + j, masked
            // to -1 when matrix > s.
            for s in 0..<seqlen {
                let base = max(s - windowSize + 1, 0)
                for j in 0..<min(seqlen, windowSize) {
                    let v = base + j
                    matrix[s * windowSize + j] = (v > s) ? -1 : Int32(v)
                }
            }
        }

        // Broadcast across batch.
        for b in 0..<batch {
            for s in 0..<seqlen {
                for j in 0..<windowSize {
                    out[(b * seqlen + s) * windowSize + j] = matrix[s * windowSize + j]
                }
            }
        }
        return out
    }

    /// Compressed-token positions for layers without an Indexer (compress_ratio == 128).
    /// Each query attends to all compressed tokens that come strictly before it.
    /// Returns `[bsz, seqlen, K]` where `K = seqlen / ratio` (prefill) or
    /// `(startPos + 1) / ratio` (decode).
    public static func compressed(ratio: Int, batch: Int, seqlen: Int,
                                   startPos: Int, offset: Int) -> (idxs: [Int32], K: Int) {
        if startPos > 0 {
            let K = (startPos + 1) / ratio
            var matrix = [Int32](repeating: 0, count: K)
            for k in 0..<K { matrix[k] = Int32(k + offset) }
            var out = [Int32](repeating: 0, count: batch * seqlen * K)
            for b in 0..<batch {
                for s in 0..<seqlen {
                    for k in 0..<K { out[(b * seqlen + s) * K + k] = matrix[k] }
                }
            }
            return (out, K)
        } else {
            let K = seqlen / ratio
            var matrix = [Int32](repeating: 0, count: seqlen * K)
            for s in 0..<seqlen {
                let limit = (s + 1) / ratio
                for k in 0..<K {
                    matrix[s * K + k] = (k >= limit) ? -1 : Int32(k + offset)
                }
            }
            var out = [Int32](repeating: 0, count: batch * seqlen * K)
            for b in 0..<batch {
                for s in 0..<seqlen {
                    for k in 0..<K { out[(b * seqlen + s) * K + k] = matrix[s * K + k] }
                }
            }
            return (out, K)
        }
    }
}
