// GLM 5.2 — nucleo dell'\''attenzione compatta DSA (F32 e Q8_0)
// Parte della libreria concatenata: vedi MetalRuntime.kernelFiles per l'ordine.

#include <metal_stdlib>
using namespace metal;

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

