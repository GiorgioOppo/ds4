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
