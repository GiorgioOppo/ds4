// GLM 5.2 — scoring dell'\''indexer DSA
// Parte della libreria concatenata: vedi MetalRuntime.kernelFiles per l'ordine.

#include <metal_stdlib>
using namespace metal;

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

