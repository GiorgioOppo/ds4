#include <metal_stdlib>
using namespace metal;

// y = x * rsqrt(mean(x^2) + eps) * weight
// One threadgroup per row. Threads cooperate on the reduction.
// x: [rows, dim] f32, weight: [dim], y: [rows, dim] f32
//
// Two kernels for weight dtype: f32 (post-conversion / inferred-defaults)
// and bf16 (HF-native — `torch_dtype: bfloat16` checkpoints store every
// RMSNorm gain in BF16). Earlier the f32 kernel was used regardless of
// the actual on-disk dtype, so a BF16 weight buffer was reinterpreted as
// F32: half the reads land inside the legit data (BF16 ≈1.0 values pair
// up into F32 ≈1.0, harmless) and the other half spill into the adjacent
// tensor bytes inside the streaming-pool rotating slot — which, for
// ffn_norm.weight in V4-Flash, happen to be FP8 quantization codes that
// read as F32 give ~1e35. That blows up the RMSNorm output to NaN and
// every downstream layer collapses.

template <typename W>
inline float load_weight(device const W* w, uint i);

template <> inline float load_weight<float>(device const float* w, uint i) {
    return w[i];
}
template <> inline float load_weight<bfloat>(device const bfloat* w, uint i) {
    return float(w[i]);
}

template <typename W>
inline void rmsnorm_impl(
    device const float* x,
    device const W*     weight,
    device float*       y,
    uint                dim,
    float               eps,
    uint                row,
    uint                tid,
    uint                tcount
) {
    threadgroup float partial[32];

    device const float* xrow = x + row * dim;
    device float*       yrow = y + row * dim;

    float sumsq = 0.0f;
    for (uint i = tid; i < dim; i += tcount) {
        float v = xrow[i];
        sumsq += v * v;
    }

    sumsq = simd_sum(sumsq);
    uint lane = tid % 32;
    uint warp = tid / 32;
    if (lane == 0) partial[warp] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp == 0) {
        float v = (tid < (tcount + 31) / 32) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = partial[0] / (float)dim;
    float scale = rsqrt(mean + eps);

    for (uint i = tid; i < dim; i += tcount) {
        yrow[i] = xrow[i] * scale * load_weight<W>(weight, i);
    }
}

kernel void rmsnorm_f32(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float*       y       [[buffer(2)]],
    constant uint&      dim     [[buffer(3)]],
    constant float&     eps     [[buffer(4)]],
    uint                row     [[threadgroup_position_in_grid]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tcount  [[threads_per_threadgroup]]
) {
    rmsnorm_impl<float>(x, weight, y, dim, eps, row, tid, tcount);
}

kernel void rmsnorm_bf16w_f32(
    device const float*  x      [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float*        y      [[buffer(2)]],
    constant uint&       dim    [[buffer(3)]],
    constant float&      eps    [[buffer(4)]],
    uint                 row    [[threadgroup_position_in_grid]],
    uint                 tid    [[thread_position_in_threadgroup]],
    uint                 tcount [[threads_per_threadgroup]]
) {
    rmsnorm_impl<bfloat>(x, weight, y, dim, eps, row, tid, tcount);
}
