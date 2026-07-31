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
    uint32_t cache_row;
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

// Laguna always uses 128-wide heads, with either 64 or 128 rotary
// dimensions. One SIMD lane can therefore own {lane, lane+32, lane+64,
// lane+96}: the complete RMS reduction is one simd_sum and both elements of
// each NeoX pair remain in registers. This is the portable part of the
// mlxfast branch's barrier-free path and works on pre-Metal-4 Apple GPUs.
static inline float4 laguna_head_rms_norm_rope_neox_simd(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *row,
        device const float *weight,
        ushort lane,
        uint token) {
    const uint d0 = lane;
    const uint d1 = lane + 32u;
    const uint d2 = lane + 64u;
    const uint d3 = lane + 96u;
    float4 values = float4(row[d0], row[d1], row[d2], row[d3]);
    const float square_sum = simd_sum(
        values.x * values.x + values.y * values.y
        + values.z * values.z + values.w * values.w);
    const float inv = rsqrt(square_sum / 128.0f + args.eps);
    values *= inv * float4(
        weight[d0], weight[d1], weight[d2], weight[d3]);

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        rope_yarn_corr_dims((int)args.n_rot,
                            (int)args.n_ctx_orig,
                            args.freq_base,
                            args.beta_fast,
                            args.beta_slow,
                            corr_dims);
    }
    const float inv_ndims = -1.0f / (float)args.n_rot;
    const float pos = (float)(args.pos0 + token);
#ifdef DS4_METAL_ROPE_EXP2_LOG2
    const float log2_base = log2(args.freq_base);
#define DS4_LAGUNA_ROPE_THETA(rel) \
        (pos * exp2(inv_ndims * (float)(rel) * log2_base))
#else
#define DS4_LAGUNA_ROPE_THETA(rel) \
        (pos * pow(args.freq_base, inv_ndims * (float)(rel)))
#endif
    if (args.n_rot == 128u) {
        float cos_theta;
        float sin_theta;
        int rel_i0 = (int)(2u * lane);
        rope_yarn(DS4_LAGUNA_ROPE_THETA(rel_i0),
                  args.freq_scale, corr_dims, rel_i0,
                  args.ext_factor, args.attn_factor,
                  &cos_theta, &sin_theta);
        const float x0 = values.x;
        const float x1 = values.z;
        values.x = x0 * cos_theta - x1 * sin_theta;
        values.z = x0 * sin_theta + x1 * cos_theta;

        rel_i0 = (int)(2u * (lane + 32u));
        rope_yarn(DS4_LAGUNA_ROPE_THETA(rel_i0),
                  args.freq_scale, corr_dims, rel_i0,
                  args.ext_factor, args.attn_factor,
                  &cos_theta, &sin_theta);
        const float y0 = values.y;
        const float y1 = values.w;
        values.y = y0 * cos_theta - y1 * sin_theta;
        values.w = y0 * sin_theta + y1 * cos_theta;
    } else {
        // n_rot == 64: the upper half of the head remains unrotated.
        float cos_theta;
        float sin_theta;
        const int rel_i0 = (int)(2u * lane);
        rope_yarn(DS4_LAGUNA_ROPE_THETA(rel_i0),
                  args.freq_scale, corr_dims, rel_i0,
                  args.ext_factor, args.attn_factor,
                  &cos_theta, &sin_theta);
        const float x0 = values.x;
        const float x1 = values.y;
        values.x = x0 * cos_theta - x1 * sin_theta;
        values.y = x0 * sin_theta + x1 * cos_theta;
    }
#undef DS4_LAGUNA_ROPE_THETA

    row[d0] = values.x;
    row[d1] = values.y;
    row[d2] = values.z;
    row[d3] = values.w;
    return values;
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

// Decode fusion retaining the legacy reduction order. This is the preferred
// M1 path: it removes the standalone KV-store dispatch without forcing the
// four-head SIMD scheduling that is beneficial only on newer GPUs.
kernel void kernel_laguna_qk_head_rms_norm_rope_store_neox(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *q,
        device float       *k,
        device const float *q_weight,
        device const float *k_weight,
        constant uint      &n_q_head,
        device const float *v,
        device half        *key_cache,
        device half        *value_cache,
        threadgroup float  *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 threads_per_group [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint combined_head = tgpig.x;
    const uint thread_count = threads_per_group.x;
    if (combined_head >= args.n_head || args.n_tokens != 1u ||
        n_q_head >= args.n_head || args.head_dim == 0u ||
        args.n_rot > args.head_dim || (args.n_rot & 1u) != 0u) {
        return;
    }

    const bool is_q = combined_head < n_q_head;
    const uint tensor_head = is_q ? combined_head : combined_head - n_q_head;
    const uint tensor_n_head = is_q ? n_q_head : args.n_head - n_q_head;
    device float *row = (is_q ? q : k) +
        (uint64_t)tensor_head * args.head_dim;
    device const float *weight = is_q ? q_weight : k_weight;

    float square_sum = 0.0f;
    for (uint i = tid; i < args.head_dim; i += thread_count) {
        const float value = row[i];
        square_sum += value * value;
    }
    scratch[tid] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = thread_count >> 1u; step != 0u; step >>= 1u) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(
        scratch[0] / (float)args.head_dim + args.eps);
    for (uint i = tid; i < args.head_dim; i += thread_count) {
        row[i] = row[i] * inv * weight[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    const uint half_rot = args.n_rot >> 1u;
    if (tid < half_rot) {
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
        const float theta = (float)args.pos0 *
            exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
        const float theta = (float)args.pos0 *
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
    if (is_q) return;

    threadgroup_barrier(mem_flags::mem_device);
    const uint width = tensor_n_head * args.head_dim;
    const uint64_t dst =
        (uint64_t)args.cache_row * width
        + (uint64_t)tensor_head * args.head_dim;
    device const float *v_row =
        v + (uint64_t)tensor_head * args.head_dim;
    for (uint i = tid; i < args.head_dim; i += thread_count) {
        key_cache[dst + i] = (half)row[i];
        value_cache[dst + i] = (half)v_row[i];
    }
}

// Four heads per threadgroup, one independent SIMD group per head. Keeping
// the legacy kernels above provides an A/B and correctness escape hatch.
kernel void kernel_laguna_qk_head_rms_norm_rope_simd(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *q,
        device float       *k,
        device const float *q_weight,
        device const float *k_weight,
        constant uint      &n_q_head,
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint combined_head = tgpig.x * 4u + (uint)simd_group;
    const uint token = tgpig.y;
    if (combined_head >= args.n_head || token >= args.n_tokens ||
        n_q_head >= args.n_head || args.head_dim != 128u ||
        (args.n_rot != 64u && args.n_rot != 128u)) {
        return;
    }

    const bool is_q = combined_head < n_q_head;
    const uint tensor_head = is_q ? combined_head : combined_head - n_q_head;
    const uint tensor_n_head = is_q ? n_q_head : args.n_head - n_q_head;
    device float *row = (is_q ? q : k) +
        ((uint64_t)token * tensor_n_head + tensor_head) * args.head_dim;
    laguna_head_rms_norm_rope_neox_simd(
        args, row, is_q ? q_weight : k_weight, lane, token);
}

// Decode-only fusion: K is converted directly from its freshly roped
// registers to the F16 ring, while the same K row remains available in float
// for diagnostics. V is committed by the K-head SIMD groups in the same
// dispatch. No inter-group barrier is required.
kernel void kernel_laguna_qk_head_rms_norm_rope_store_simd(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *q,
        device float       *k,
        device const float *q_weight,
        device const float *k_weight,
        constant uint      &n_q_head,
        device const float *v,
        device half        *key_cache,
        device half        *value_cache,
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint combined_head = tgpig.x * 4u + (uint)simd_group;
    if (combined_head >= args.n_head || args.n_tokens != 1u ||
        n_q_head >= args.n_head || args.head_dim != 128u ||
        (args.n_rot != 64u && args.n_rot != 128u)) {
        return;
    }

    const bool is_q = combined_head < n_q_head;
    const uint tensor_head = is_q ? combined_head : combined_head - n_q_head;
    const uint tensor_n_head = is_q ? n_q_head : args.n_head - n_q_head;
    device float *row = (is_q ? q : k) +
        (uint64_t)tensor_head * args.head_dim;
    const float4 roped = laguna_head_rms_norm_rope_neox_simd(
        args, row, is_q ? q_weight : k_weight, lane, 0u);
    if (is_q) return;

    const uint width = tensor_n_head * args.head_dim;
    const uint64_t dst =
        (uint64_t)args.cache_row * width
        + (uint64_t)tensor_head * args.head_dim;
    device const float *v_row =
        v + (uint64_t)tensor_head * args.head_dim;
    key_cache[dst + lane] = (half)roped.x;
    key_cache[dst + lane + 32u] = (half)roped.y;
    key_cache[dst + lane + 64u] = (half)roped.z;
    key_cache[dst + lane + 96u] = (half)roped.w;
    value_cache[dst + lane] = (half)v_row[lane];
    value_cache[dst + lane + 32u] = (half)v_row[lane + 32u];
    value_cache[dst + lane + 64u] = (half)v_row[lane + 64u];
    value_cache[dst + lane + 96u] = (half)v_row[lane + 96u];
}
