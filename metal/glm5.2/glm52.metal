// GLM 5.2 (`glm-dsa`) kernels.
//
// Keep these kernels architecture-owned instead of adding GLM branches to the
// DeepSeek V4 source. The first correctness tranche contains the exact GLM
// router; compact DSA attention and MoE kernels are added beside it only after
// their CPU/reference fixtures are available.

#include <metal_stdlib>
using namespace metal;

struct ds4_metal_args_glm52_router_select {
    uint32_t n_expert;
    uint32_t n_expert_used;
    float    expert_weight_scale;
    uint32_t pad0;
};

static inline float ds4_glm52_router_sigmoid(float x) {
    // Stable form matching the reference implementation for large logits.
    if (x >= 0.0f) {
        const float e = exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = exp(x);
    return e / (1.0f + e);
}

static inline bool ds4_glm52_router_better(
        threadgroup const float *scores,
        int32_t                  a,
        int32_t                  b) {
    const float sa = scores[(uint)a];
    const float sb = scores[(uint)b];
    // The lower expert id wins exact ties. This is part of deterministic greedy
    // inference and must stay identical across CPU, Metal and cache budgets.
    return sa > sb || (sa == sb && a < b);
}

// GLM 5.2 router for one or more token rows. Selection uses
// sigmoid(logit)+bias, while route weights use the unbiased sigmoid values.
// One 256-lane threadgroup owns one token row.
kernel void kernel_glm52_router_select(
        constant ds4_metal_args_glm52_router_select &args,
        device const float   *logits,
        device const float   *bias,
        device int32_t       *selected,
        device float         *weights,
        device float         *probs,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float *selection_scores = scratch;
    threadgroup int32_t *indices =
        (threadgroup int32_t *)(scratch + 256u);

    device const float *token_logits =
        logits + (uint64_t)token * args.n_expert;
    device int32_t *token_selected =
        selected + (uint64_t)token * args.n_expert_used;
    device float *token_weights =
        weights + (uint64_t)token * args.n_expert_used;
    device float *token_probs =
        probs + (uint64_t)token * args.n_expert;

    const uint n_expert = min(args.n_expert, 256u);
    const bool active = tid < n_expert;
    const float probability = active
        ? ds4_glm52_router_sigmoid(token_logits[tid])
        : 0.0f;
    if (active) token_probs[tid] = probability;
    selection_scores[tid] = active ? probability + bias[tid] : -INFINITY;
    indices[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Power-of-two bitonic network. All 256 lanes participate in every barrier.
    for (uint width = 2u; width <= 256u; width <<= 1u) {
        for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
            const uint other = tid ^ stride;
            if (other > tid) {
                const int32_t a = indices[tid];
                const int32_t b = indices[other];
                const bool descending = (tid & width) == 0u;
                const bool should_swap = descending
                    ? ds4_glm52_router_better(selection_scores, b, a)
                    : ds4_glm52_router_better(selection_scores, a, b);
                if (should_swap) {
                    indices[tid] = b;
                    indices[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint used = min(args.n_expert_used, n_expert);
    if (tid < used) token_selected[tid] = indices[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < used) {
        float sum = 0.0f;
        for (uint i = 0u; i < used; i++) {
            sum += token_probs[(uint)token_selected[i]];
        }
        sum = max(sum, 6.103515625e-5f);
        token_weights[tid] =
            token_probs[(uint)token_selected[tid]] / sum *
            args.expert_weight_scale;
    }
}

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

// MARK: - DSA indexer scoring

struct ds4_metal_args_glm52_indexer_scores {
    uint32_t n_rows;
    uint32_t n_tokens;
    uint32_t pos0;
    float    scale;
};

// Exact GLM 5.2 indexer geometry: 32 query heads of width 128 against one
// shared F16 key row. A 128-lane group consists of four SIMD-groups; each SIMD
// group evaluates one head at a time. Output is token-major [n_tokens, n_rows].
// Rows after pos0 + token are causal future rows and are written as -INFINITY.
kernel void kernel_glm52_indexer_scores_f16(
        constant ds4_metal_args_glm52_indexer_scores &args,
        device const float *queries,
        device const float *head_weights,
        device const half  *indexer_key_cache,
        device float       *scores,
        threadgroup float  *shared [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint N_HEAD = 32u;
    constexpr uint HEAD_DIM = 128u;
    constexpr uint HEADS_PER_WAVE = 4u;
    const uint row = group.x;
    const uint token = group.y;

    if (row >= args.n_rows || token >= args.n_tokens) return;

    device float *dst = scores + (uint64_t)token * args.n_rows + row;
    if (args.pos0 >= args.n_rows ||
        token >= args.n_rows - args.pos0 ||
        row > args.pos0 + token) {
        if (tid == 0u) *dst = -INFINITY;
        return;
    }

    threadgroup float *key = shared;
    threadgroup float *partial_scores = shared + HEAD_DIM;
    if (tid < HEAD_DIM) {
        key[tid] = (float)indexer_key_cache[(uint64_t)row * HEAD_DIM + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float score = 0.0f;
    for (uint head0 = 0u; head0 < N_HEAD; head0 += HEADS_PER_WAVE) {
        const uint head = head0 + (uint)simd_group;
        device const float4 *q4 = (device const float4 *)(queries +
            ((uint64_t)token * N_HEAD + head) * HEAD_DIM);
        threadgroup const float4 *k4 = (threadgroup const float4 *)key;

        float dot_product = dot(q4[lane], k4[lane]);
        dot_product = simd_sum(dot_product);
        if (lane == 0u) {
            const float weight = head_weights[(uint64_t)token * N_HEAD + head];
            partial_scores[simd_group] =
                max(dot_product * args.scale, 0.0f) * weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            score += partial_scores[0];
            score += partial_scores[1];
            score += partial_scores[2];
            score += partial_scores[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) *dst = score;
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
