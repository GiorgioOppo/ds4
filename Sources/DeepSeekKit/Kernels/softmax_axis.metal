#include <metal_stdlib>
using namespace metal;

// Softmax along an arbitrary axis. Generalises softmax.metal which only
// covers the last dim. Used by Compressor for per-window softmax over
// `dim=2` after `score.unflatten(1, (-1, ratio))` (model.py:342).
//
// The tensor is interpreted as [outer, axis, inner] where:
//   outer = product of dims before the softmax axis
//   axis  = size of the softmax axis
//   inner = product of dims after the softmax axis
//
// Each (outer, inner) pair is one independent softmax row (length=axis).
// One threadgroup per (outer, inner). Reduction across `axis` via
// threadgroup memory.

// Reduction strategy: simdgroup ops (simd_max / simd_sum) + 1 stage
// cross-simdgroup. Sostituisce il vecchio reduction tree
// log2(tgsize) shared-memory.
//
// Benefits vs reduction tree:
//   - log2(256)=8 barrier sostituiti da 1 barrier
//   - simd_max/simd_sum sono hardware-accelerated (1 cycle SIMD-wide)
//   - shared memory: 32 float (1 slot per simdgroup) vs tgsize float
//
// Allinea il design a `softmax_f32` (one-row kernel) che già usa
// questa strategia, e ai pattern del kernel `rmsnorm_f32`.
kernel void softmax_axis_f32(
    device float*       x       [[buffer(0)]],
    constant uint3&     dims    [[buffer(1)]],   // (outer, axis, inner)
    threadgroup float*  shared_ [[threadgroup(0)]],
    uint2 tg     [[threadgroup_position_in_grid]],
    uint2 tidv   [[thread_position_in_threadgroup]],
    uint2 tgsv   [[threads_per_threadgroup]]
) {
    uint outer = tg.x, inner = tg.y;
    uint OUTER = dims.x, AXIS = dims.y, INNER = dims.z;
    uint tid = tidv.x;
    uint tgsize = tgsv.x;
    if (outer >= OUTER || inner >= INNER) return;

    uint lane = tid % 32u;
    uint warp = tid / 32u;
    uint nWarps = (tgsize + 31u) / 32u;

    // base offset for this (outer, inner) row; stride between successive
    // axis elements is INNER (axis is the middle dim).
    uint base = outer * AXIS * INNER + inner;

    // 1) max-reduction over the axis: each thread sweeps a slice of
    // the axis, then simd_max within the simdgroup, then a single
    // cross-simdgroup reduction in shared memory.
    float m = -INFINITY;
    for (uint i = tid; i < AXIS; i += tgsize) {
        m = max(m, x[base + i * INNER]);
    }
    m = simd_max(m);
    if (lane == 0u) shared_[warp] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0u) {
        float v = (lane < nWarps) ? shared_[lane] : -INFINITY;
        v = simd_max(v);
        if (lane == 0u) shared_[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = shared_[0];

    // 2) exp + sum (stesso pattern).
    float s = 0.0f;
    for (uint i = tid; i < AXIS; i += tgsize) {
        float e = exp(x[base + i * INNER] - gmax);
        x[base + i * INNER] = e;
        s += e;
    }
    s = simd_sum(s);
    if (lane == 0u) shared_[warp] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0u) {
        float v = (lane < nWarps) ? shared_[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) shared_[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gsum = shared_[0];

    // 3) normalize
    for (uint i = tid; i < AXIS; i += tgsize) {
        x[base + i * INNER] /= gsum;
    }
}
