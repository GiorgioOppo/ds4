#include <metal_stdlib>
using namespace metal;

// Standard scaled-dot-product attention with GQA + causal mask
// (TODO §10.2 / T2). One thread per (batch, query_head, query_pos)
// cell; each thread streams the full S_kv key/value sequence and
// accumulates softmax(Q · K^T / sqrt(D)) · V into the corresponding
// D-element output row.
//
// Streaming-softmax (FlashAttention-style numerics — without the
// tiling) keeps memory in registers:
//   m  ← running max of (score - constant)
//   l  ← running sum of exp(score - m)
//   o  ← running weighted V accumulator
// After the loop, normalize o ← o / l.
//
// "Naive" because no tiling, no simdgroup_matrix MMA, no
// threadgroup cooperation — one thread does the whole row's work.
// Good enough for decode (S_q = 1, B*Hq threads max) where the
// inner cost is O(S_kv * D); too slow for long prefill where
// FlashAttention tiling would batter this by an order of
// magnitude. Replacing this kernel is on the perf roadmap.
//
// Layouts (row-major, last dim contiguous):
//   Q   [B, S_q,  Hq,  D]  f32
//   K   [B, S_kv, Hkv, D]  f32   (S_kv used prefix of the cache)
//   V   [B, S_kv, Hkv, D]  f32
//   Out [B, S_q,  Hq,  D]  f32
//
// GQA: each query head h_q maps to the kv head
//   h_kv = (h_q * Hkv) / Hq
// — integer division. Requires Hq % Hkv == 0 (caller-checked).
//
// Causal mask: query at absolute position (startPos + s_q) attends
// to key positions [0, startPos + s_q]. Future positions
// (s_kv > startPos + s_q) are skipped.

constant uint SDPA_MAX_D = 256;   // largest head_dim we accommodate inline

kernel void sdpa_naive_causal_gqa_f32(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V       [[buffer(2)]],
    device float*       Out     [[buffer(3)]],
    constant uint4&     dims    [[buffer(4)]],  // (B, S_q, S_kv, D)
    constant uint2&     heads   [[buffer(5)]],  // (Hq, Hkv)
    constant uint&      startPos[[buffer(6)]],
    constant float&     scale   [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint b   = gid.x;
    uint hq  = gid.y;
    uint sq  = gid.z;
    uint B   = dims.x;
    uint S_q = dims.y;
    uint S_kv = dims.z;
    uint D   = dims.w;
    uint Hq  = heads.x;
    uint Hkv = heads.y;
    if (b >= B || hq >= Hq || sq >= S_q) return;

    uint h_kv = (hq * Hkv) / Hq;
    uint q_abs = startPos + sq;

    // Pointers to the Q row and the row stride for K/V at this batch
    // and kv-head.
    device const float* qrow =
        Q + (((b * S_q) + sq) * Hq + hq) * D;
    // K[b, s_kv, h_kv, :] = K + (((b * S_kv) + s_kv) * Hkv + h_kv) * D
    // pre-compute the base + per-s_kv stride for clarity.
    uint kv_stride = Hkv * D;
    device const float* k0 =
        K + ((b * S_kv) * Hkv + h_kv) * D;
    device const float* v0 =
        V + ((b * S_kv) * Hkv + h_kv) * D;
    device float* orow =
        Out + (((b * S_q) + sq) * Hq + hq) * D;

    // Streaming-softmax state.
    float m = -INFINITY;
    float l = 0.0f;
    float o[SDPA_MAX_D];
    for (uint d = 0; d < D; d++) o[d] = 0.0f;

    for (uint s = 0; s <= q_abs && s < S_kv; s++) {
        device const float* krow = k0 + s * kv_stride;
        // Dot product Q[..] · K[..] with scale baked in (saves a
        // multiply per step).
        float dot = 0.0f;
        for (uint d = 0; d < D; d++) {
            dot += qrow[d] * krow[d];
        }
        float score = dot * scale;

        // Streaming rescale.
        float m_new = max(m, score);
        float alpha = exp(m - m_new);   // safe: m starts at -INF, alpha=0 first iter
        float p     = exp(score - m_new);
        device const float* vrow = v0 + s * kv_stride;
        for (uint d = 0; d < D; d++) {
            o[d] = o[d] * alpha + p * vrow[d];
        }
        l = l * alpha + p;
        m = m_new;
    }

    // Normalize. Guard against `l == 0` (happens iff no s_kv was
    // attended to — shouldn't happen since s = 0 is always allowed
    // for q_abs ≥ 0, but be defensive).
    float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint d = 0; d < D; d++) {
        orow[d] = o[d] * inv_l;
    }
}

// --------------------------------------------------------------------
// KV cache append. Copies one turn's freshly-computed K and V (shape
// [B, S, Hkv, D]) into the per-layer cache buffers at row `startPos`
// of the second dim. Pure blit-style — could be replaced with an
// `MTLBlitCommandEncoder.copy` call from Swift, but a single kernel
// dispatch keeps the contract obvious in the layer code.
// --------------------------------------------------------------------

kernel void kv_cache_append_f32(
    device const float* src       [[buffer(0)]],  // [B, S, Hkv, D]
    device float*       cache     [[buffer(1)]],  // [B, MaxSeq, Hkv, D]
    constant uint4&     dims      [[buffer(2)]],  // (B, S, MaxSeq, Hkv * D)
    constant uint&      startPos  [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint b      = gid.x;
    uint s      = gid.y;
    uint d_inner= gid.z;  // index in flattened (Hkv * D) span
    uint B      = dims.x;
    uint S      = dims.y;
    uint MaxSeq = dims.z;
    uint HKVxD  = dims.w;
    if (b >= B || s >= S || d_inner >= HKVxD) return;
    uint srcIdx   = ((b * S) + s) * HKVxD + d_inner;
    uint dstS     = startPos + s;
    if (dstS >= MaxSeq) return;
    uint dstIdx   = ((b * MaxSeq) + dstS) * HKVxD + d_inner;
    cache[dstIdx] = src[srcIdx];
}
