// Laguna S 2.1 — per-head RMSNorm + NeoX RoPE (ported from the reference
// laguna-s2.1 branch, head 448d569). Needs metal/deepseek/dsv4_rope.metal first.
// Part of the concatenated library: see MetalRuntime.kernelFiles for the order.

#include <metal_stdlib>
using namespace metal;

// Laguna-specific primitives. The C graph owns model semantics and scheduling;
// these kernels only cover operations that are not represented by the shared
// DeepSeek/GLM Metal API.

struct ds4_metal_args_laguna_norm_rope {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    float    eps;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t pad0;
};

static inline void laguna_head_rms_norm_rope_neox(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *row,
        device const float *weight,
        threadgroup float  *scratch [[threadgroup(0)]],
        uint tid,
        uint nth,
        uint token) {
    float ss = 0.0f;
    for (uint i = tid; i < args.head_dim; i += nth) {
        const float v = row[i];
        ss += v * v;
    }
    scratch[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1u; step != 0u; step >>= 1u) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(scratch[0] / (float)args.head_dim + args.eps);
    for (uint i = tid; i < args.head_dim; i += nth) {
        row[i] = row[i] * inv * weight[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    const uint half_rot = args.n_rot >> 1u;
    if (tid >= half_rot) return;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        rope_yarn_corr_dims((int)args.n_rot,
                            (int)args.n_ctx_orig,
                            args.freq_base,
                            args.beta_fast,
                            args.beta_slow,
                            corr_dims);
    }
    const int rel_i0 = (int)(tid * 2u);
    const float inv_ndims = -1.0f / (float)args.n_rot;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
    const float theta = (float)(args.pos0 + token) *
        exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
    const float theta = (float)(args.pos0 + token) *
        pow(args.freq_base, inv_ndims * (float)rel_i0);
#endif
    float cos_theta;
    float sin_theta;
    rope_yarn(theta,
              args.freq_scale,
              corr_dims,
              rel_i0,
              args.ext_factor,
              args.attn_factor,
              &cos_theta,
              &sin_theta);
    const float x0 = row[tid];
    const float x1 = row[tid + half_rot];
    row[tid] = x0 * cos_theta - x1 * sin_theta;
    row[tid + half_rot] = x0 * sin_theta + x1 * cos_theta;
}

// Laguna uses Qwen-style per-head RMSNorm and NeoX rotary pairs. Rotary
// dimensions occupy the prefix of each head; any remaining dimensions are
// normalized but left unrotated.
kernel void kernel_laguna_head_rms_norm_rope_neox(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *x,
        device const float *weight,
        threadgroup float  *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens ||
        args.head_dim == 0u || args.n_rot > args.head_dim ||
        (args.n_rot & 1u) != 0u) {
        return;
    }

    device float *row = x +
        ((uint64_t)token * args.n_head + head) * args.head_dim;
    laguna_head_rms_norm_rope_neox(
        args, row, weight, scratch, tid, ntg_u.x, token);
}

// Decode uses the same norm/RoPE arithmetic for Q and K. Keeping both tensors
// in one grid removes a small Metal dispatch without changing the per-head
// reduction order.
kernel void kernel_laguna_qk_head_rms_norm_rope_neox(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *q,
        device float       *k,
        device const float *q_weight,
        device const float *k_weight,
        constant uint      &n_q_head,
        threadgroup float  *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint combined_head = tgpig.x;
    const uint token = tgpig.y;
    if (combined_head >= args.n_head || token >= args.n_tokens ||
        n_q_head >= args.n_head || args.head_dim == 0u ||
        args.n_rot > args.head_dim || (args.n_rot & 1u) != 0u) {
        return;
    }

    const bool is_q = combined_head < n_q_head;
    const uint tensor_head = is_q ? combined_head : combined_head - n_q_head;
    const uint tensor_n_head = is_q ? n_q_head : args.n_head - n_q_head;
    device float *row = (is_q ? q : k) +
        ((uint64_t)token * tensor_n_head + tensor_head) * args.head_dim;
    device const float *weight = is_q ? q_weight : k_weight;
    laguna_head_rms_norm_rope_neox(
        args, row, weight, scratch, tid, ntg_u.x, token);
}
