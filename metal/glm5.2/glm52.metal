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

// MARK: - Compact DSA attention core

// The three kernels below are the decode-side attention over the compact
// KV-LoRA cache, split by stage exactly like the CPU oracle
// (GLM52AttentionCPUReference.absorbed): absorb the query into attn_k_b once,
// score the selected rows, accumulate the softmax in the KV-LoRA domain, then
// project through attn_v_b once. Geometry is architecture-exact GLM 5.2:
// 64 heads, per-head QK 192 nope + 64 rope, KV-LoRA 512, value 256, cache row
// 512 + 64 = 576 F16 values.

struct ds4_metal_args_glm52_attention {
    uint32_t n_rows;
    uint32_t n_selected;
    float    scale;
    // Decode semantics: the compact cache keeps RAW K-RoPE tails and each
    // selected row is rotated at attention time with its own absolute
    // position (pos = row index). Zero reads the tail as stored — the
    // pre-rotated fixture path of the isolated kernel tests.
    uint32_t rotate_k_tail;
};

// q_low[h][j] = dot(q_nope[h][0..192), attn_k_b[h][j][0..192)).
// One 32-lane SIMD group per output element; four elements per threadgroup.
// Grid: (512/4, 64 heads). queries are [64][256] rows (nope prefix + rope
// tail); key_b is [64][512][192] row-major.
kernel void kernel_glm52_qk_lowrank_f32(
        constant ds4_metal_args_glm52_attention &args,
        device const float *queries,
        device const float *key_b,
        device float       *q_low,
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint N_HEAD = 64u;
    constexpr uint N_NOPE = 192u;
    constexpr uint N_LORA = 512u;
    constexpr uint QK_DIM = 256u;
    const uint j = group.x * 4u + (uint)simd_group;
    const uint head = group.y;
    if (j >= N_LORA || head >= N_HEAD) return;

    device const float *q = queries + (uint64_t)head * QK_DIM;
    device const float *w = key_b + ((uint64_t)head * N_LORA + j) * N_NOPE;
    float partial = 0.0f;
    for (uint i = lane; i < N_NOPE; i += 32u) {
        partial += q[i] * w[i];
    }
    partial = simd_sum(partial);
    if (lane == 0u) {
        q_low[(uint64_t)head * N_LORA + j] = partial;
    }
}

// Softmax attention of ONE token over the selected compact-cache rows,
// accumulated in the KV-LoRA domain. One 128-thread threadgroup per head;
// scores live in threadgroup memory, so n_selected is capped at 2048 (the
// architecture's indexer top-k). Output attn_lora is [64][512], already
// divided by the softmax denominator (clamped at 1e-20 like upstream).
// shared layout: [n_selected scores | 4 reduction slots | 1 broadcast slot].
kernel void kernel_glm52_attention_indexed_f16(
        constant ds4_metal_args_glm52_attention &args,
        device const float    *q_low,
        device const float    *queries,
        device const half     *compact_cache,
        device const uint32_t *selected,
        device float          *attn_lora,
        threadgroup float     *shared [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint N_HEAD = 64u;
    constexpr uint N_LORA = 512u;
    constexpr uint ROT_DIM = 64u;
    constexpr uint QK_DIM = 256u;
    constexpr uint N_NOPE = QK_DIM - ROT_DIM;
    constexpr uint ROW_WIDTH = N_LORA + ROT_DIM;
    constexpr uint N_THREADS = 128u;
    const uint head = group.x;
    if (head >= N_HEAD || args.n_selected == 0u) return;

    threadgroup float *scores = shared;
    threadgroup float *reduction = shared + args.n_selected;
    threadgroup float *broadcast = reduction + 4u;
    device const float *q_low_head = q_low + (uint64_t)head * N_LORA;
    device const float *q_rope = queries + (uint64_t)head * QK_DIM + N_NOPE;

    // Phase A: raw scaled scores of every selected row.
    for (uint s = tid; s < args.n_selected; s += N_THREADS) {
        const uint row_index = selected[s];
        device const half *row = compact_cache +
            (uint64_t)row_index * ROW_WIDTH;
        float dot_product = 0.0f;
        for (uint j = 0u; j < N_LORA; j++) {
            dot_product += q_low_head[j] * (float)row[j];
        }
        if (args.rotate_k_tail != 0u) {
            // Upstream decode: theta = row_pos * 8e6^(-2t/64) on adjacent
            // pairs of the raw tail, recomputed per row from the F16 cache.
            const float row_position = (float)row_index;
            for (uint pair = 0u; pair < ROT_DIM / 2u; pair++) {
                const float theta = row_position *
                    pow(8000000.0f, -2.0f * (float)pair / (float)ROT_DIM);
                const float c = cos(theta);
                const float sn = sin(theta);
                const float x0 = (float)row[N_LORA + pair * 2u];
                const float x1 = (float)row[N_LORA + pair * 2u + 1u];
                dot_product += q_rope[pair * 2u] * (x0 * c - x1 * sn)
                    + q_rope[pair * 2u + 1u] * (x0 * sn + x1 * c);
            }
        } else {
            for (uint i = 0u; i < ROT_DIM; i++) {
                dot_product += q_rope[i] * (float)row[N_LORA + i];
            }
        }
        scores[s] = dot_product * args.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase B: stable softmax — shared maximum, then exp and denominator.
    float local_max = -INFINITY;
    for (uint s = tid; s < args.n_selected; s += N_THREADS) {
        local_max = max(local_max, scores[s]);
    }
    local_max = simd_max(local_max);
    if (lane == 0u) reduction[simd_group] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        broadcast[0] = max(max(reduction[0], reduction[1]),
                           max(reduction[2], reduction[3]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float max_score = broadcast[0];

    float local_sum = 0.0f;
    for (uint s = tid; s < args.n_selected; s += N_THREADS) {
        const float weight = exp(scores[s] - max_score);
        scores[s] = weight;
        local_sum += weight;
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0u) reduction[simd_group] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        const float denominator =
            reduction[0] + reduction[1] + reduction[2] + reduction[3];
        broadcast[0] = max(denominator, 1.0e-20f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denominator = broadcast[0];

    // Phase C: weighted accumulation of the KV-LoRA prefixes.
    for (uint j = tid; j < N_LORA; j += N_THREADS) {
        float accumulator = 0.0f;
        for (uint s = 0u; s < args.n_selected; s++) {
            device const half *row = compact_cache +
                (uint64_t)selected[s] * ROW_WIDTH;
            accumulator += scores[s] * (float)row[j];
        }
        attn_lora[(uint64_t)head * N_LORA + j] = accumulator / denominator;
    }
}

// out[h][d] = dot(attn_v_b[h][d][0..512), attn_lora[h][0..512)).
// One 32-lane SIMD group per output element; four elements per threadgroup.
// Grid: (256/4, 64 heads). value_b is [64][256][512] row-major.
kernel void kernel_glm52_value_project_f32(
        constant ds4_metal_args_glm52_attention &args,
        device const float *attn_lora,
        device const float *value_b,
        device float       *attn_out,
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint N_HEAD = 64u;
    constexpr uint N_LORA = 512u;
    constexpr uint N_VALUE = 256u;
    const uint d = group.x * 4u + (uint)simd_group;
    const uint head = group.y;
    if (d >= N_VALUE || head >= N_HEAD) return;

    device const float *lora = attn_lora + (uint64_t)head * N_LORA;
    device const float *w = value_b +
        ((uint64_t)head * N_VALUE + d) * N_LORA;
    float partial = 0.0f;
    for (uint j = lane; j < N_LORA; j += 32u) {
        partial += w[j] * lora[j];
    }
    partial = simd_sum(partial);
    if (lane == 0u) {
        attn_out[(uint64_t)head * N_VALUE + d] = partial;
    }
}

// MARK: - Compact DSA attention core (Q8_0 weights)

// Q8_0 variants of the two projection stages. Row layouts match the F32
// kernels exactly — [head][kvLora] rows of 192 for attn_k_b, [head][value]
// rows of 512 for attn_v_b — with each row stored as consecutive block_q8_0
// blocks (2-byte f16 scale + 32 int8 = 34 bytes, prelude struct). Each of the
// 32 lanes owns position `lane` of every block, so the block scale multiplies
// OUTSIDE the int8 product exactly like upstream's dot_q8_0_row_f32_ref; the
// activation stays F32 (the bit-faithful reference path, not the requantized
// matvec_q8_0 fast path).

kernel void kernel_glm52_qk_lowrank_q8_0(
        constant ds4_metal_args_glm52_attention &args,
        device const float *queries,
        device const char  *key_b,
        device float       *q_low,
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint N_HEAD = 64u;
    constexpr uint N_NOPE = 192u;
    constexpr uint N_LORA = 512u;
    constexpr uint QK_DIM = 256u;
    constexpr uint N_BLOCKS = N_NOPE / QK8_0;             // 6
    constexpr uint ROW_BYTES = N_BLOCKS * 34u;            // 204
    const uint j = group.x * 4u + (uint)simd_group;
    const uint head = group.y;
    if (j >= N_LORA || head >= N_HEAD) return;

    device const float *q = queries + (uint64_t)head * QK_DIM;
    device const block_q8_0 *row = (device const block_q8_0 *)
        (key_b + ((uint64_t)head * N_LORA + j) * ROW_BYTES);
    float partial = 0.0f;
    for (uint b = 0u; b < N_BLOCKS; b++) {
        partial += (float)row[b].d * (float)row[b].qs[lane]
            * q[b * QK8_0 + lane];
    }
    partial = simd_sum(partial);
    if (lane == 0u) {
        q_low[(uint64_t)head * N_LORA + j] = partial;
    }
}

kernel void kernel_glm52_value_project_q8_0(
        constant ds4_metal_args_glm52_attention &args,
        device const float *attn_lora,
        device const char  *value_b,
        device float       *attn_out,
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint N_HEAD = 64u;
    constexpr uint N_LORA = 512u;
    constexpr uint N_VALUE = 256u;
    constexpr uint N_BLOCKS = N_LORA / QK8_0;             // 16
    constexpr uint ROW_BYTES = N_BLOCKS * 34u;            // 544
    const uint d = group.x * 4u + (uint)simd_group;
    const uint head = group.y;
    if (d >= N_VALUE || head >= N_HEAD) return;

    device const float *lora = attn_lora + (uint64_t)head * N_LORA;
    device const block_q8_0 *row = (device const block_q8_0 *)
        (value_b + ((uint64_t)head * N_VALUE + d) * ROW_BYTES);
    float partial = 0.0f;
    for (uint b = 0u; b < N_BLOCKS; b++) {
        partial += (float)row[b].d * (float)row[b].qs[lane]
            * lora[b * QK8_0 + lane];
    }
    partial = simd_sum(partial);
    if (lane == 0u) {
        attn_out[(uint64_t)head * N_VALUE + d] = partial;
    }
}

// MARK: - Routed expert FFN (K-quant weights)

// Validation kernels for the quantized FFN matvec stages: one thread owns one
// output row and dequantizes its row serially with the exact reference
// pairing (same element order as the CPU dequant references), so the GPU path
// is comparable to the FFN oracle on dequantized weights. Q8_0 covers the
// dense blocks, the shared expert and the output head; the K-quants cover the
// routed experts. Correctness only — the tuned per-quant families come later,
// beside these fixtures.

struct ds4_metal_args_glm52_moe {
    uint32_t weight_type;    // GGUF id: 10=q2_K, 12=q4_K, 13=q5_K, 14=q6_K
    uint32_t row_count;
    uint32_t input_width;    // dot width, multiple of 256
    float    route_weight;   // pair kernel: multiplies the SwiGLU mid
};

static inline float glm52_half_at(device const uchar *p) {
    return (float)(*(device const half *)p);
}

static inline uchar2 glm52_scale_min_k4(uint j, device const uchar *q) {
    if (j < 4u) {
        return uchar2(q[j] & 63u, q[j + 4u] & 63u);
    }
    return uchar2((q[j + 4u] & 0x0Fu) | ((q[j - 4u] >> 6u) << 4u),
                  (q[j + 4u] >> 4u) | ((q[j] >> 6u) << 4u));
}

static inline float glm52_dot_q2_K_row(device const uchar *row,
                                       device const float *x,
                                       uint width) {
    float acc = 0.0f;
    for (uint sb = 0u; sb < width / 256u; sb++) {
        device const uchar *base = row + sb * 84u;
        device const uchar *qs = base + 16u;
        const float d = glm52_half_at(base + 80u);
        const float dmin = glm52_half_at(base + 82u);
        for (uint half128 = 0u; half128 < 256u; half128 += 128u) {
            for (uint plane = 0u; plane < 4u; plane++) {
                for (uint l = 0u; l < 32u; l++) {
                    const uchar sc = base[half128 / 16u + plane * 2u + l / 16u];
                    const uint q = (qs[half128 / 4u + l] >> (plane * 2u)) & 3u;
                    const float w = d * (float)(sc & 0x0Fu) * (float)q -
                        dmin * (float)(sc >> 4u);
                    acc += w * x[sb * 256u + half128 + plane * 32u + l];
                }
            }
        }
    }
    return acc;
}

static inline float glm52_dot_q4_K_row(device const uchar *row,
                                       device const float *x,
                                       uint width) {
    float acc = 0.0f;
    for (uint sb = 0u; sb < width / 256u; sb++) {
        device const uchar *base = row + sb * 144u;
        const float d = glm52_half_at(base);
        const float dmin = glm52_half_at(base + 2u);
        device const uchar *scales = base + 4u;
        device const uchar *qs = base + 16u;
        for (uint j = 0u; j < 8u; j++) {
            const uchar2 sm = glm52_scale_min_k4(j, scales);
            const float dj = d * (float)sm.x;
            const float mj = dmin * (float)sm.y;
            const uint chunk = (j / 2u) * 32u;
            const uint shift = (j & 1u) * 4u;
            for (uint l = 0u; l < 32u; l++) {
                const uint q = (qs[chunk + l] >> shift) & 0x0Fu;
                acc += (dj * (float)q - mj) * x[sb * 256u + j * 32u + l];
            }
        }
    }
    return acc;
}

static inline float glm52_dot_q5_K_row(device const uchar *row,
                                       device const float *x,
                                       uint width) {
    float acc = 0.0f;
    for (uint sb = 0u; sb < width / 256u; sb++) {
        device const uchar *base = row + sb * 176u;
        const float d = glm52_half_at(base);
        const float dmin = glm52_half_at(base + 2u);
        device const uchar *scales = base + 4u;
        device const uchar *qh = base + 16u;
        device const uchar *qs = base + 48u;
        for (uint j = 0u; j < 8u; j++) {
            const uchar2 sm = glm52_scale_min_k4(j, scales);
            const float dj = d * (float)sm.x;
            const float mj = dmin * (float)sm.y;
            const uint chunk = (j / 2u) * 32u;
            const uint shift = (j & 1u) * 4u;
            for (uint l = 0u; l < 32u; l++) {
                const uint q = ((qs[chunk + l] >> shift) & 0x0Fu) +
                    ((qh[l] >> j) & 1u) * 16u;
                acc += (dj * (float)q - mj) * x[sb * 256u + j * 32u + l];
            }
        }
    }
    return acc;
}

static inline float glm52_dot_q6_K_row(device const uchar *row,
                                       device const float *x,
                                       uint width) {
    float acc = 0.0f;
    for (uint sb = 0u; sb < width / 256u; sb++) {
        device const uchar *base = row + sb * 210u;
        device const uchar *ql = base;
        device const uchar *qh = base + 128u;
        device const char *scales = (device const char *)(base + 192u);
        const float d = glm52_half_at(base + 208u);
        for (uint half128 = 0u; half128 < 256u; half128 += 128u) {
            const uint qlHalf = half128 / 2u;
            const uint qhHalf = half128 / 4u;
            const uint scHalf = half128 / 16u;
            for (uint l = 0u; l < 32u; l++) {
                const uint sub = l / 16u;
                const uchar high = qh[qhHalf + l];
                const int q1 = (int)((ql[qlHalf + l] & 0x0Fu) | ((high << 4u) & 0x30u)) - 32;
                const int q2 = (int)((ql[qlHalf + l + 32u] & 0x0Fu) | ((high << 2u) & 0x30u)) - 32;
                const int q3 = (int)((ql[qlHalf + l] >> 4u) | (high & 0x30u)) - 32;
                const int q4 = (int)((ql[qlHalf + l + 32u] >> 4u) | ((high >> 2u) & 0x30u)) - 32;
                const uint out0 = sb * 256u + half128 + l;
                acc += d * (float)scales[scHalf + sub] * (float)q1 * x[out0];
                acc += d * (float)scales[scHalf + sub + 2u] * (float)q2 * x[out0 + 32u];
                acc += d * (float)scales[scHalf + sub + 4u] * (float)q3 * x[out0 + 64u];
                acc += d * (float)scales[scHalf + sub + 6u] * (float)q4 * x[out0 + 96u];
            }
        }
    }
    return acc;
}

static inline float glm52_dot_q8_0_row(device const uchar *row,
                                       device const float *x,
                                       uint width) {
    float acc = 0.0f;
    for (uint b = 0u; b < width / 32u; b++) {
        device const uchar *base = row + b * 34u;
        const float d = glm52_half_at(base);
        device const char *qs = (device const char *)(base + 2u);
        for (uint i = 0u; i < 32u; i++) {
            acc += d * (float)qs[i] * x[b * 32u + i];
        }
    }
    return acc;
}

static inline uint glm52_kquant_row_bytes(uint type, uint width) {
    switch (type) {
        case 8u:  return (width / 32u) * 34u;
        case 10u: return (width / 256u) * 84u;
        case 12u: return (width / 256u) * 144u;
        case 13u: return (width / 256u) * 176u;
        case 14u: return (width / 256u) * 210u;
        default:  return 0u;
    }
}

static inline float glm52_dot_kquant_row(uint type,
                                         device const uchar *row,
                                         device const float *x,
                                         uint width) {
    switch (type) {
        case 8u:  return glm52_dot_q8_0_row(row, x, width);
        case 10u: return glm52_dot_q2_K_row(row, x, width);
        case 12u: return glm52_dot_q4_K_row(row, x, width);
        case 13u: return glm52_dot_q5_K_row(row, x, width);
        case 14u: return glm52_dot_q6_K_row(row, x, width);
        default:  return 0.0f;
    }
}

static inline float glm52_silu(float value) {
    // Stable sigmoid, matching upstream sigmoid_stable.
    const float s = value >= 0.0f
        ? 1.0f / (1.0f + exp(-value))
        : exp(value) / (1.0f + exp(value));
    return value * s;
}

// mid[r] = silu(gate_r · x) * (up_r · x) * route_weight — GLM SwiGLU has no
// clamp and the route weight multiplies the mid BEFORE the down projection.
kernel void kernel_glm52_moe_pair_swiglu(
        constant ds4_metal_args_glm52_moe &args,
        device const float *x,
        device const uchar *gate_rows,
        device const uchar *up_rows,
        device float       *mid,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.row_count) return;
    const uint row_bytes = glm52_kquant_row_bytes(args.weight_type,
                                                  args.input_width);
    if (row_bytes == 0u) return;
    const uint64_t offset = (uint64_t)tid * row_bytes;
    const float g = glm52_dot_kquant_row(args.weight_type, gate_rows + offset,
                                         x, args.input_width);
    const float u = glm52_dot_kquant_row(args.weight_type, up_rows + offset,
                                         x, args.input_width);
    mid[tid] = glm52_silu(g) * u * args.route_weight;
}

// out[r] = down_r · mid for ONE expert; the chained validation wrapper sums
// expert contributions host-side.
kernel void kernel_glm52_moe_down(
        constant ds4_metal_args_glm52_moe &args,
        device const float *mid,
        device const uchar *down_rows,
        device float       *out,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.row_count) return;
    const uint row_bytes = glm52_kquant_row_bytes(args.weight_type,
                                                  args.input_width);
    if (row_bytes == 0u) return;
    out[tid] = glm52_dot_kquant_row(args.weight_type,
                                    down_rows + (uint64_t)tid * row_bytes,
                                    mid, args.input_width);
}

// MARK: - Query/K-tail RoPE

struct ds4_metal_args_glm52_rope_tail {
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t pos;
};

// Rotate the n_rot-wide tail of every head in place: adjacent pairs, plain
// linear GLM RoPE (theta_k = pos * 8e6^(-2k/n_rot); ext_factor 0, freq_scale
// 1, no YaRN — upstream rope_tail_ext_inplace with GLM's shape constants).
// Serves both the 64-per-head query tails and the single 64-wide K row
// before its compact-cache store (head_dim == n_rot, nope 0).
kernel void kernel_glm52_rope_tail_f32(
        constant ds4_metal_args_glm52_rope_tail &args,
        device float *values,
        uint tid [[thread_position_in_grid]]) {
    constexpr float FREQ_BASE = 8000000.0f;
    const uint pairs = args.n_rot / 2u;
    if (tid >= args.n_head * pairs) return;
    const uint head = tid / pairs;
    const uint pair = tid % pairs;
    const uint nope = args.head_dim - args.n_rot;
    device float *tail = values + (uint64_t)head * args.head_dim
        + nope + pair * 2u;
    const float theta = (float)args.pos
        * pow(FREQ_BASE, -2.0f * (float)pair / (float)args.n_rot);
    const float c = cos(theta);
    const float s = sin(theta);
    const float x0 = tail[0];
    const float x1 = tail[1];
    tail[0] = x0 * c - x1 * s;
    tail[1] = x0 * s + x1 * c;
}

// Rotate the n_rot-wide PREFIX of every head in place — the indexer-side
// convention: upstream forces rot_offset = 0 for the 128-wide indexer
// queries and keys, while the MLA query rotates its tail. Same adjacent-pair
// linear GLM RoPE as the tail kernel; only the span origin differs.
kernel void kernel_glm52_rope_prefix_f32(
        constant ds4_metal_args_glm52_rope_tail &args,
        device float *values,
        uint tid [[thread_position_in_grid]]) {
    constexpr float FREQ_BASE = 8000000.0f;
    const uint pairs = args.n_rot / 2u;
    if (tid >= args.n_head * pairs) return;
    const uint head = tid / pairs;
    const uint pair = tid % pairs;
    device float *span = values + (uint64_t)head * args.head_dim + pair * 2u;
    const float theta = (float)args.pos
        * pow(FREQ_BASE, -2.0f * (float)pair / (float)args.n_rot);
    const float c = cos(theta);
    const float s = sin(theta);
    const float x0 = span[0];
    const float x1 = span[1];
    span[0] = x0 * c - x1 * s;
    span[1] = x0 * s + x1 * c;
}
