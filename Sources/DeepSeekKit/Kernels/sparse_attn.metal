#include <metal_stdlib>
using namespace metal;

// sparse_attn — sparse multi-head attention with FlashAttention-style online
// softmax + KV gather via topk_idxs + per-head learnable sink logit.
//
// Mirrors `sparse_attn_kernel` from
// Reference/inference/kernel.py lines 277–352.
//
// Layout:
//   q          [B, M, H, D] f32    queries
//   kv         [B, N, D]    f32    shared K=V (DeepSeek MLA: one KV head)
//   sink       [H]          f32    per-head sink logit (learnable)
//   topk_idxs  [B, M, K]    i32    KV positions to attend to; -1 = padding
//   o          [B, M, H, D] f32    output
//   scale      float
//
// Algorithm per (b, m, h):
//   acc_o = 0;  sum_exp = 0;  s_max = -inf
//   for k in 0..K:
//       idx = topk_idxs[b, m, k]
//       if idx == -1: continue
//       score = (q[b,m,h] · kv[b,idx]) * scale
//       new_max = max(s_max, score)
//       fac     = exp(s_max - new_max)
//       e       = exp(score - new_max)
//       acc_o  *= fac;   acc_o   += e * kv[b, idx]
//       sum_exp *= fac;  sum_exp += e
//       s_max = new_max
//   sum_exp += exp(sink[h] - s_max)            # absorb sink logit
//   o[b, m, h] = acc_o / sum_exp
//
// Parallelism: one thread per (b, m, h). The accumulator lives in device
// memory (we update `o` in place), so each step is a read-modify-write of D
// floats. Slow but correct — see README "Performance" for the planned
// FlashAttention tiling rewrite.

kernel void sparse_attn_f32(
    device const float* q       [[buffer(0)]],
    device const float* kv      [[buffer(1)]],
    device const float* sink    [[buffer(2)]],
    device const int*   tk      [[buffer(3)]],
    device float*       o       [[buffer(4)]],
    constant uint4&     dims    [[buffer(5)]],   // (B, M, N, D)
    constant uint2&     misc    [[buffer(6)]],   // (H, K)
    constant float&     scale   [[buffer(7)]],
    constant uint&      mOffset [[buffer(8)]],   // query-tile start (see SparseAttention.apply)
    uint3 gid [[thread_position_in_grid]]
) {
    // `m` is global: the grid height is one query *tile*, `mOffset` shifts
    // it back to the absolute position so q/tk/o index correctly while the
    // dispatch — and its command buffer — stays small enough to dodge the
    // macOS GPU interactivity watchdog.
    uint h = gid.x, m = gid.y + mOffset, b = gid.z;
    uint B = dims.x, M = dims.y, N = dims.z, D = dims.w;
    uint H = misc.x, K = misc.y;
    if (b >= B || m >= M || h >= H) return;

    uint qOff = ((b * M + m) * H + h) * D;
    uint oOff = qOff;
    uint tkOff = (b * M + m) * K;

    device const float* qrow = q + qOff;
    device const int*   tkrow = tk + tkOff;
    device float*       orow = o + oOff;

    // Init accumulator to 0.
    for (uint d = 0; d < D; d++) orow[d] = 0.0f;

    float s_max = -INFINITY;
    float sum_exp = 0.0f;

    for (uint k = 0; k < K; k++) {
        int idx = tkrow[k];
        if (idx < 0) continue;
        device const float* kvrow = kv + ((b * N) + (uint)idx) * D;

        // score = q · kv
        float score = 0.0f;
        for (uint d = 0; d < D; d++) score += qrow[d] * kvrow[d];
        score *= scale;

        // Online softmax
        float new_max = max(s_max, score);
        float fac = (s_max == -INFINITY) ? 0.0f : exp(s_max - new_max);
        float e   = exp(score - new_max);

        for (uint d = 0; d < D; d++) {
            orow[d] = orow[d] * fac + e * kvrow[d];
        }
        sum_exp = sum_exp * fac + e;
        s_max = new_max;
    }

    // Absorb the sink logit. If no valid index was hit (s_max still -inf),
    // the sink alone determines the denominator: o stays zero, sum_exp = 1
    // after the exp(sink - sink) → o ends up as 0 (division by 1) which is
    // a sensible degenerate output.
    if (s_max == -INFINITY) {
        sum_exp = 1.0f;   // exp(0) for the sink term
    } else {
        sum_exp += exp(sink[h] - s_max);
    }

    for (uint d = 0; d < D; d++) orow[d] /= sum_exp;
}
