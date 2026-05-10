#include <metal_stdlib>
using namespace metal;

// Composition kernels for HyperConnections.pre / .post.
// Used together with hc_split_sinkhorn (in hc_sinkhorn.metal) and the dense
// f32 matmul (in gemm_bf16.metal) to implement HC pre / post collapse and
// expand passes from Reference/inference/model.py:673-686.

// rsqrt_mean_square — out[row] = rsqrt(mean(x[row, :]²) + eps).
// One threadgroup per row, threads cooperate on the sum via shared memory.
kernel void rsqrt_mean_square_f32(
    device const float* x       [[buffer(0)]],
    device float*       out     [[buffer(1)]],
    constant uint&      dim     [[buffer(2)]],
    constant float&     eps     [[buffer(3)]],
    threadgroup float*  shared_ [[threadgroup(0)]],
    uint   row     [[threadgroup_position_in_grid]],
    uint   tid     [[thread_position_in_threadgroup]],
    uint   tgsize  [[threads_per_threadgroup]]
) {
    device const float* row_ptr = x + row * dim;
    float sumsq = 0.0f;
    for (uint i = tid; i < dim; i += tgsize) {
        float v = row_ptr[i];
        sumsq += v * v;
    }
    shared_[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shared_[tid] += shared_[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float mean = shared_[0] / float(dim);
        out[row] = rsqrt(mean + eps);
    }
}

// broadcast_row_mul — y[row, c] *= scale[row], in place.
kernel void broadcast_row_mul_f32(
    device float*       y       [[buffer(0)]],
    device const float* scale   [[buffer(1)]],
    constant uint2&     dims    [[buffer(2)]],   // (rows, cols)
    uint2 gid [[thread_position_in_grid]]
) {
    uint c = gid.x, r = gid.y;
    if (r >= dims.x || c >= dims.y) return;
    y[r * dims.y + c] *= scale[r];
}

// hc_collapse — y[n, d] = Σ_h pre[n, h] · x[n, h, d].
// Implements `torch.sum(pre.unsqueeze(-1) * x, dim=2)` from model.py:680.
kernel void hc_collapse_f32(
    device const float* x       [[buffer(0)]],   // [N, hc, d]
    device const float* pre     [[buffer(1)]],   // [N, hc]
    device float*       y       [[buffer(2)]],   // [N, d]
    constant uint3&     dims    [[buffer(3)]],   // (N, hc, d)
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, n = gid.y;
    uint N = dims.x, HC = dims.y, D = dims.z;
    if (d >= D || n >= N) return;
    float acc = 0.0f;
    for (uint h = 0; h < HC; h++) {
        acc += pre[n * HC + h] * x[(n * HC + h) * D + d];
    }
    y[n * D + d] = acc;
}

// hc_post_compose — y[n, j, d] = post[n, j] · xv[n, d] + Σ_k comb[n, k, j] · residual[n, k, d].
// Implements model.py:685
//     y = post.unsqueeze(-1) * x.unsqueeze(-2) +
//         torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
// where the contraction is over the FIRST hc dim of comb (call it k).
kernel void hc_post_compose_f32(
    device const float* xv       [[buffer(0)]],   // [N, d]
    device const float* residual [[buffer(1)]],   // [N, hc, d]
    device const float* post     [[buffer(2)]],   // [N, hc]
    device const float* comb     [[buffer(3)]],   // [N, hc, hc] — index [N, k, j]
    device float*       y        [[buffer(4)]],   // [N, hc, d]
    constant uint3&     dims     [[buffer(5)]],   // (N, hc, d)
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, j = gid.y, n = gid.z;
    uint N = dims.x, HC = dims.y, D = dims.z;
    if (d >= D || j >= HC || n >= N) return;
    float term1 = post[n * HC + j] * xv[n * D + d];
    float term2 = 0.0f;
    for (uint k = 0; k < HC; k++) {
        term2 += comb[(n * HC + k) * HC + j] * residual[(n * HC + k) * D + d];
    }
    y[(n * HC + j) * D + d] = term1 + term2;
}
