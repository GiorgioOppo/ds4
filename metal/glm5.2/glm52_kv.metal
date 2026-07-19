// GLM 5.2 — primitive della cache compatta DSA e store dei piani KV/indexer
// Parte della libreria concatenata: vedi MetalRuntime.kernelFiles per l'ordine.

#include <metal_stdlib>
using namespace metal;

// MARK: - Compact DSA cache primitives

// Cache-ready GLM KV rows contain a normalized 512-wide KV-LoRA prefix and the
// untouched 64-wide K-RoPE tail. Normalization and RoPE are intentionally
// outside this atomic primitive so each operation can be checked independently
// against its scalar oracle.
struct ds4_metal_args_glm52_store_compact_kv {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t pad0;
};

// One (token, part) threadgroup copies either the 512-wide KV-LoRA prefix or
// the 64-wide K-RoPE tail. Both destinations are always IEEE binary16. The
// subtraction form of the capacity check avoids wrapping pos0 + token.
kernel void kernel_glm52_store_compact_kv_f16(
        constant ds4_metal_args_glm52_store_compact_kv &args,
        device const float *kv_rows,
        device half        *kv_lora_cache,
        device half        *k_rope_cache,
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 threads [[threads_per_threadgroup]]) {
    constexpr uint KV_RAW_WIDTH = 576u;
    constexpr uint KV_LORA_WIDTH = 512u;
    constexpr uint K_ROPE_WIDTH = 64u;
    const uint token = group.x;
    const uint part = group.y;
    const uint nth = threads.x;

    if (token >= args.n_tokens || part > 1u ||
        args.pos0 >= args.cache_cap ||
        token >= args.cache_cap - args.pos0) {
        return;
    }

    const uint pos = args.pos0 + token;
    device const float *src = kv_rows + (uint64_t)token * KV_RAW_WIDTH;
    if (part == 0u) {
        device half *dst = kv_lora_cache + (uint64_t)pos * KV_LORA_WIDTH;
        for (uint i = tid; i < KV_LORA_WIDTH; i += nth) {
            dst[i] = (half)src[i];
        }
    } else {
        device half *dst = k_rope_cache + (uint64_t)pos * K_ROPE_WIDTH;
        for (uint i = tid; i < K_ROPE_WIDTH; i += nth) {
            dst[i] = (half)src[KV_LORA_WIDTH + i];
        }
    }
}

// MARK: - GLM KV-LoRA normalization

struct ds4_metal_args_glm52_kv_lora_norm {
    uint32_t n_tokens;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

// Produce a cache-ready 576-wide F32 row. Only the 512-wide KV-LoRA prefix is
// RMS-normalized and weighted; the 64-wide K-RoPE section is copied bit-for-bit
// as F32 so its positional rotation remains a later, independent operation.
kernel void kernel_glm52_kv_lora_norm_cache_ready_f32(
        constant ds4_metal_args_glm52_kv_lora_norm &args,
        device const float *raw_rows,
        device const float *weight,
        device float       *cache_ready_rows,
        threadgroup float  *scratch [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 threads [[threads_per_threadgroup]]) {
    constexpr uint RAW_WIDTH = 576u;
    constexpr uint KV_LORA_WIDTH = 512u;
    constexpr float EPSILON = 1.0e-5f;

    const uint token = group.x;
    const uint nth = threads.x;
    if (token >= args.n_tokens || nth != 128u) return;

    device const float *src = raw_rows + (uint64_t)token * RAW_WIDTH;
    device float *dst = cache_ready_rows + (uint64_t)token * RAW_WIDTH;

    float sum_squares = 0.0f;
    for (uint i = tid; i < KV_LORA_WIDTH; i += nth) {
        const float value = src[i];
        sum_squares += value * value;
    }
    scratch[tid] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1u; step > 0u; step >>= 1u) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inverse_rms =
        rsqrt(scratch[0] / (float)KV_LORA_WIDTH + EPSILON);
    for (uint i = tid; i < KV_LORA_WIDTH; i += nth) {
        dst[i] = (src[i] * inverse_rms) * weight[i];
    }
    for (uint i = KV_LORA_WIDTH + tid; i < RAW_WIDTH; i += nth) {
        dst[i] = src[i];
    }
}

// MARK: - GLM indexer-key normalization and store

struct ds4_metal_args_glm52_store_indexer_k {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t pad0;
};

// The GGUF tensors call this k_norm, but the upstream arithmetic is centered
// LayerNorm rather than RMSNorm: mean, centered variance, affine weight+bias,
// then RoPE. GLM 5.2 rotates indices 0..<64 of the 128-wide indexer key. Keep
// that prefix layout explicit: the query-side upstream helper uses the same
// offset even though its generic function name contains "rope_tail".
kernel void kernel_glm52_store_indexer_k_f16(
        constant ds4_metal_args_glm52_store_indexer_k &args,
        device const float *raw_keys,
        device const float *weight,
        device const float *bias,
        device half        *indexer_key_cache,
        threadgroup float  *scratch [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 threads [[threads_per_threadgroup]]) {
    constexpr uint HEAD_DIM = 128u;
    constexpr uint ROT_DIM = 64u;
    constexpr float EPSILON = 1.0e-6f;
    constexpr float FREQUENCY_BASE = 8.0e6f;

    const uint token = group.x;
    const uint nth = threads.x;
    if (token >= args.n_tokens || nth != 32u ||
        args.pos0 >= args.cache_cap ||
        token >= args.cache_cap - args.pos0) {
        return;
    }

    const uint position = args.pos0 + token;
    device const float *src = raw_keys + (uint64_t)token * HEAD_DIM;
    device half *dst = indexer_key_cache + (uint64_t)position * HEAD_DIM;

    float sum = 0.0f;
    for (uint i = tid; i < HEAD_DIM; i += nth) sum += src[i];
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1u; step > 0u; step >>= 1u) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = scratch[0] / (float)HEAD_DIM;

    float sum_squares = 0.0f;
    for (uint i = tid; i < HEAD_DIM; i += nth) {
        const float centered = src[i] - mean;
        sum_squares += centered * centered;
    }
    scratch[tid] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1u; step > 0u; step >>= 1u) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_deviation =
        rsqrt(scratch[0] / (float)HEAD_DIM + EPSILON);

    const float theta_base = (float)position;
    const float inverse_rotation_dimension = -1.0f / (float)ROT_DIM;
    for (uint i = tid; i < HEAD_DIM; i += nth) {
        if (i < ROT_DIM) {
            if ((i & 1u) != 0u) continue;
            const float theta = theta_base *
                pow(FREQUENCY_BASE, inverse_rotation_dimension * (float)i);
            const float cosine = cos(theta);
            const float sine = sin(theta);
            const float x0 = ((src[i] - mean) * inverse_deviation) * weight[i]
                + bias[i];
            const uint j = i + 1u;
            const float x1 = ((src[j] - mean) * inverse_deviation) * weight[j]
                + bias[j];
            dst[i] = (half)(x0 * cosine - x1 * sine);
            dst[j] = (half)(x0 * sine + x1 * cosine);
        } else {
            const float value = ((src[i] - mean) * inverse_deviation) * weight[i]
                + bias[i];
            dst[i] = (half)value;
        }
    }
}

kernel void kernel_glm52_store_compact_row_f16(
        constant ds4_metal_args_glm52_store_compact_kv &args,
        device const float *rows,
        device half        *compact_cache,
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 threads [[threads_per_threadgroup]]) {
    constexpr uint ROW_WIDTH = 576u;
    const uint token = group.x;
    const uint nth = threads.x;
    if (token >= args.n_tokens ||
        args.pos0 >= args.cache_cap ||
        token >= args.cache_cap - args.pos0) {
        return;
    }
    device const float *src = rows + (uint64_t)token * ROW_WIDTH;
    device half *dst = compact_cache +
        (uint64_t)(args.pos0 + token) * ROW_WIDTH;
    for (uint i = tid; i < ROW_WIDTH; i += nth) {
        dst[i] = (half)src[i];
    }
}

// Rotate the n_rot-wide PREFIX of every head in place — the indexer-side
// convention: upstream forces rot_offset = 0 for the 128-wide indexer
// queries and keys, while the MLA query rotates its tail. Same adjacent-pair
// linear GLM RoPE as the tail kernel; only the span origin differs.
