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

kernel void softmax_axis_f32(
    device float*       x       [[buffer(0)]],
    constant uint3&     dims    [[buffer(1)]],   // (outer, axis, inner)
    threadgroup float*  shared_ [[threadgroup(0)]],
    uint2 tg     [[threadgroup_position_in_grid]],
    uint  tid    [[thread_position_in_threadgroup]],
    uint  tgsize [[threads_per_threadgroup]]
) {
    uint outer = tg.x, inner = tg.y;
    uint OUTER = dims.x, AXIS = dims.y, INNER = dims.z;
    if (outer >= OUTER || inner >= INNER) return;

    // base offset for this (outer, inner) row; stride between successive
    // axis elements is INNER (axis is the middle dim).
    uint base = outer * AXIS * INNER + inner;

    // 1) max-reduction over the axis
    float m = -INFINITY;
    for (uint i = tid; i < AXIS; i += tgsize) {
        m = max(m, x[base + i * INNER]);
    }
    shared_[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shared_[tid] = max(shared_[tid], shared_[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float gmax = shared_[0];

    // 2) exp + sum
    float s = 0.0f;
    for (uint i = tid; i < AXIS; i += tgsize) {
        float e = exp(x[base + i * INNER] - gmax);
        x[base + i * INNER] = e;
        s += e;
    }
    shared_[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shared_[tid] += shared_[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float gsum = shared_[0];

    // 3) normalize
    for (uint i = tid; i < AXIS; i += tgsize) {
        x[base + i * INNER] /= gsum;
    }
}
