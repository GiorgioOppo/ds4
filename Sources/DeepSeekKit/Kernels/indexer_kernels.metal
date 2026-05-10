#include <metal_stdlib>
using namespace metal;

// Composition kernels for Indexer.forward.
// Mirrors the reduction step in Reference/inference/model.py:421:
//
//   index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
//
// followed by an optional causal mask + the post-topk mask + offset.

// indexer_score_reduce: y[b, s, t] = Σ_h max(0, score[b, s, h, t]) * weights[b, s, h]
// Optionally applies the prefill causal mask: if `causal == 1`, then any cell
// with `t >= (s + 1) / ratio` is set to -inf to exclude future compressed
// tokens from the topk. (model.py:425-426)
kernel void indexer_score_reduce_f32(
    device const float* score    [[buffer(0)]],   // [B, S, H, T]
    device const float* weights  [[buffer(1)]],   // [B, S, H]
    device float*       y        [[buffer(2)]],   // [B, S, T]
    constant uint4&     dims     [[buffer(3)]],   // (B, S, H, T)
    constant uint2&     misc     [[buffer(4)]],   // (ratio, causal)
    uint3 gid [[thread_position_in_grid]]
) {
    uint t = gid.x, s = gid.y, b = gid.z;
    uint B = dims.x, S = dims.y, H = dims.z, T = dims.w;
    uint ratio = misc.x, causal = misc.y;
    if (b >= B || s >= S || t >= T) return;

    if (causal != 0u && t >= (s + 1u) / ratio) {
        y[(b * S + s) * T + t] = -INFINITY;
        return;
    }

    float acc = 0.0f;
    for (uint h = 0; h < H; h++) {
        float sc = score[((b * S + s) * H + h) * T + t];
        sc = max(sc, 0.0f);   // relu
        acc += sc * weights[(b * S + s) * H + h];
    }
    y[(b * S + s) * T + t] = acc;
}

// indexer_topk_postprocess: mask invalid topk indices and add offset.
// Mirrors model.py:428-432:
//   if start_pos == 0:
//       mask = topk_idxs >= (s + 1) / ratio
//       topk_idxs = where(mask, -1, topk_idxs + offset)
//   else:
//       topk_idxs += offset
//
// `causal == 1` only when start_pos == 0 (prefill). For decode we just add
// offset unconditionally.
kernel void indexer_topk_postprocess_i32(
    device int*         idxs   [[buffer(0)]],   // [B, S, K] — modified in place
    constant uint3&     dims   [[buffer(1)]],   // (B, S, K)
    constant uint3&     misc   [[buffer(2)]],   // (ratio, offset, causal)
    uint3 gid [[thread_position_in_grid]]
) {
    uint k = gid.x, s = gid.y, b = gid.z;
    uint B = dims.x, S = dims.y, K = dims.z;
    uint ratio = misc.x;
    int offset = (int)misc.y;
    uint causal = misc.z;
    if (b >= B || s >= S || k >= K) return;

    uint loc = (b * S + s) * K + k;
    int v = idxs[loc];
    if (v < 0) return;            // already masked
    if (causal != 0u && (uint)v >= (s + 1u) / ratio) {
        idxs[loc] = -1;
    } else {
        idxs[loc] = v + offset;
    }
}
