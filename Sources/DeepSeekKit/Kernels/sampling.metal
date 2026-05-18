#include <metal_stdlib>
using namespace metal;

// Apply temperature in-place: logits[i] /= T.  T==0 leaves logits untouched
// (the host should switch to argmax in that case).
kernel void apply_temperature(
    device float*    logits   [[buffer(0)]],
    constant uint&   V        [[buffer(1)]],
    constant float&  T        [[buffer(2)]],
    uint             gid      [[thread_position_in_grid]]
) {
    if (gid >= V || T == 0.0f) return;
    logits[gid] /= T;
}

// Argmax over [V] producing a single uint at out[0].
//
// Single-threadgroup version: usato per V piccolo (V < 8192). Per
// V grande (es. vocab 130k), il host usa la variante multi-stage
// `argmax_f32_stage1` + `argmax_f32_stage2` che parallelizza fra
// più threadgroup.
kernel void argmax_f32(
    device const float* x   [[buffer(0)]],
    device uint*        out [[buffer(1)]],
    constant uint&      V   [[buffer(2)]],
    uint                tid [[thread_position_in_threadgroup]],
    uint                tcount [[threads_per_threadgroup]]
) {
    threadgroup float bestV[32];
    threadgroup uint  bestI[32];

    float lv = -INFINITY;
    uint  li = 0;
    for (uint i = tid; i < V; i += tcount) {
        float v = x[i];
        if (v > lv) { lv = v; li = i; }
    }
    uint lane = tid % 32, warp = tid / 32;
    bestV[warp] = lv; bestI[warp] = li;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp == 0) {
        float v = (tid < (tcount + 31)/32) ? bestV[lane] : -INFINITY;
        uint  ix = (tid < (tcount + 31)/32) ? bestI[lane] : 0u;
        // simple linear reduce within first warp
        for (uint off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(v, off);
            uint  oi = simd_shuffle_down(ix, off);
            if (ov > v) { v = ov; ix = oi; }
        }
        if (lane == 0) out[0] = ix;
    }
}

// Argmax multi-stage, fase 1 — tile parallel.
//
// Dispatch: M threadgroup × T thread. Ogni threadgroup gestisce il
// tile [tgId * tileSize, min((tgId+1) * tileSize, V)) e produce il
// suo (best_value, best_index) parziale in `partV[tgId]` / `partI[tgId]`.
// La riduzione fra i M parziali avviene in stage 2.
//
// Per V=130k e tileSize=2048: M=64 threadgroup paralleli — satura
// la maggior parte dei shader core sulla GPU. Prima il single-stage
// usava 1 solo threadgroup.
kernel void argmax_f32_stage1(
    device const float* x       [[buffer(0)]],
    device float*       partV   [[buffer(1)]],   // [M]
    device uint*        partI   [[buffer(2)]],   // [M]
    constant uint2&     dims    [[buffer(3)]],   // (V, tileSize)
    uint                tgId    [[threadgroup_position_in_grid]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tcount  [[threads_per_threadgroup]]
) {
    uint V = dims.x;
    uint tileSize = dims.y;
    uint start = tgId * tileSize;
    if (start >= V) return;
    uint end = min(start + tileSize, V);

    threadgroup float bestV[32];
    threadgroup uint  bestI[32];

    float lv = -INFINITY;
    uint  li = start;
    for (uint i = start + tid; i < end; i += tcount) {
        float v = x[i];
        if (v > lv) { lv = v; li = i; }
    }
    uint lane = tid % 32u, warp = tid / 32u;
    bestV[warp] = lv;
    bestI[warp] = li;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp == 0u) {
        uint nWarps = (tcount + 31u) / 32u;
        float v = (lane < nWarps) ? bestV[lane] : -INFINITY;
        uint  ix = (lane < nWarps) ? bestI[lane] : 0u;
        for (uint off = 16u; off > 0u; off >>= 1) {
            float ov = simd_shuffle_down(v, off);
            uint  oi = simd_shuffle_down(ix, off);
            if (ov > v) { v = ov; ix = oi; }
        }
        if (lane == 0u) {
            partV[tgId] = v;
            partI[tgId] = ix;
        }
    }
}

// Argmax multi-stage, fase 2 — final reduce su M parziali.
//
// Dispatch: 1 threadgroup × min(M, 256) thread. M è tipicamente
// piccolo (64-128 per V=130k), quindi 1 threadgroup va bene.
kernel void argmax_f32_stage2(
    device const float* partV   [[buffer(0)]],   // [M]
    device const uint*  partI   [[buffer(1)]],   // [M]
    device uint*        out     [[buffer(2)]],
    constant uint&      M       [[buffer(3)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tcount  [[threads_per_threadgroup]]
) {
    threadgroup float bestV[32];
    threadgroup uint  bestI[32];

    float lv = -INFINITY;
    uint  li = 0u;
    for (uint i = tid; i < M; i += tcount) {
        float v = partV[i];
        if (v > lv) { lv = v; li = partI[i]; }
    }
    uint lane = tid % 32u, warp = tid / 32u;
    bestV[warp] = lv;
    bestI[warp] = li;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp == 0u) {
        uint nWarps = (tcount + 31u) / 32u;
        float v = (lane < nWarps) ? bestV[lane] : -INFINITY;
        uint  ix = (lane < nWarps) ? bestI[lane] : 0u;
        for (uint off = 16u; off > 0u; off >>= 1) {
            float ov = simd_shuffle_down(v, off);
            uint  oi = simd_shuffle_down(ix, off);
            if (ov > v) { v = ov; ix = oi; }
        }
        if (lane == 0u) out[0] = ix;
    }
}
