#include <metal_stdlib>
using namespace metal;

// Numerically stable softmax over the last dim (length=N), one threadgroup per row.
kernel void softmax_f32(
    device float*    x        [[buffer(0)]],
    constant uint&   N        [[buffer(1)]],
    uint             row      [[threadgroup_position_in_grid]],
    uint             tid      [[thread_position_in_threadgroup]],
    uint             tcount   [[threads_per_threadgroup]]
) {
    threadgroup float partial[32];
    device float* row_ptr = x + row * N;

    // 1) max
    float m = -INFINITY;
    for (uint i = tid; i < N; i += tcount) m = max(m, row_ptr[i]);
    m = simd_max(m);
    uint lane = tid % 32, warp = tid / 32;
    if (lane == 0) partial[warp] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {
        float v = (tid < (tcount + 31)/32) ? partial[lane] : -INFINITY;
        v = simd_max(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = partial[0];

    // 2) exp + sum
    float s = 0.0f;
    for (uint i = tid; i < N; i += tcount) {
        float e = exp(row_ptr[i] - gmax);
        row_ptr[i] = e;
        s += e;
    }
    s = simd_sum(s);
    if (lane == 0) partial[warp] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {
        float v = (tid < (tcount + 31)/32) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gsum = partial[0];

    // 3) normalize
    for (uint i = tid; i < N; i += tcount) {
        row_ptr[i] /= gsum;
    }
}
