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

// MARK: - Compact top-2048 merge

// The packed key gives the merge one deterministic integer comparison:
//   high 32 bits: monotonically ordered IEEE-754 score
//   low  32 bits: reversed row id (lower id wins an exact score tie)
inline uint glm52_topk_ordered_float(float value) {
    // Indexer scores are finite in production; keep malformed NaNs
    // deterministic and below every valid value.
    if (isnan(value)) value = -INFINITY;
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0u ? ~bits : bits ^ 0x80000000u;
}

inline ulong glm52_topk_pack(float value, uint row) {
    return ((ulong)glm52_topk_ordered_float(value) << 32)
        | (ulong)(0xffffffffu - row);
}

struct ds4_metal_args_glm52_topk_merge_compact {
    uint32_t n_scores;
    uint32_t n_tokens;
    uint32_t input_count;
    uint32_t input_set_count;
    uint32_t input_stride;
    uint32_t input_row_stride;
    uint32_t output_stride;
    uint32_t output_row_stride;
};

// Merge pairs of descending index runs while retaining only output_stride
// entries from every pair. Unlike the generic argsort merge, outputs are
// compacted after each level, so a 32K context carries 32K, 16K, 8K, 4K and
// finally 2K indices instead of carrying 32K through every level.
kernel void kernel_glm52_indexer_topk_merge_compact(
        constant ds4_metal_args_glm52_topk_merge_compact &args,
        device const float *scores,
        device const uint *input,
        device uint *output,
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 threads [[threads_per_threadgroup]]) {
    const uint output_set_count = (args.input_set_count + 1u) / 2u;
    const uint linear_set = group.x;
    const uint token = linear_set / output_set_count;
    const uint output_set = linear_set - token * output_set_count;
    if (token >= args.n_tokens || output_set >= output_set_count) return;

    const uint input_set0 = output_set * 2u;
    const uint logical0 = input_set0 * args.input_stride;
    const uint logical1 = logical0 + args.input_stride;
    const uint len0 = logical0 < args.input_count
        ? min(args.input_stride, args.input_count - logical0) : 0u;
    const uint len1 = logical1 < args.input_count
        ? min(args.input_stride, args.input_count - logical1) : 0u;
    const uint output_count = min(args.output_stride, len0 + len1);
    if (output_count == 0u) return;

    device const uint *left = input
        + (uint64_t)token * args.input_row_stride
        + logical0;
    device const uint *right = left + args.input_stride;
    device uint *destination = output
        + (uint64_t)token * args.output_row_stride
        + output_set * args.output_stride;
    device const float *row = scores + (uint64_t)token * args.n_scores;

    const uint chunk = (output_count + threads.x - 1u) / threads.x;
    const uint rank0 = tid * chunk;
    const uint rank1 = min(rank0 + chunk, output_count);
    if (rank0 >= output_count) return;

    uint low = rank0 > len1 ? rank0 - len1 : 0u;
    uint high = min(rank0, len0);
    while (low < high) {
        const uint middle = (low + high) >> 1u;
        const uint left_index = left[middle];
        const uint right_index = right[rank0 - middle - 1u];
        const bool take_left =
            glm52_topk_pack(row[left_index], left_index)
            >= glm52_topk_pack(row[right_index], right_index);
        if (take_left) low = middle + 1u;
        else high = middle;
    }

    uint i = low;
    uint j = rank0 - i;
    for (uint rank = rank0; rank < rank1; rank++) {
        if (i >= len0) {
            destination[rank] = right[j++];
        } else if (j >= len1) {
            destination[rank] = left[i++];
        } else {
            const uint left_index = left[i];
            const uint right_index = right[j];
            const bool take_left =
                glm52_topk_pack(row[left_index], left_index)
                >= glm52_topk_pack(row[right_index], right_index);
            if (take_left) {
                destination[rank] = left_index;
                i++;
            } else {
                destination[rank] = right_index;
                j++;
            }
        }
    }
}
