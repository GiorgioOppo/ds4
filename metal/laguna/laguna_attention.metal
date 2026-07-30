// Laguna S 2.1 — gated GQA attention, prefill and decode, plus the fused
// flash-attention reduce with the per-head softplus gate (ported from the
// Part of the concatenated library: see MetalRuntime.kernelFiles for the order.

#include <metal_stdlib>
using namespace metal;

// reference laguna-s2.1 branch, head 448d569). Needs laguna_kv.metal (the
// prefill args struct) and metal/deepseek/flash_attn.metal before it.

struct ds4_metal_args_laguna_long_index {
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t cache_cap;
    uint32_t block_size;
    uint32_t block_count;
    uint32_t top_k;
    uint32_t raw_start;
    uint32_t raw_count;
    uint32_t score_stride;
    float    scale;
    uint32_t _pad;
};

// One F16 centroid per completed raw-key block. These rows are a search index
// only: indexed attention below consumes the original K/V of selected blocks.
kernel void kernel_laguna_long_index_compress_f16(
        constant uint32_t *args,
        device const half *key_cache,
        device half       *index_keys,
        uint gid [[thread_position_in_grid]]) {
    const uint row_width = args[0];
    const uint block_size = args[1];
    const uint first_block = args[2];
    const uint block_count = args[3];
    const uint total = block_count * row_width;
    if (gid >= total || row_width == 0u || block_size == 0u) return;
    const uint local_block = gid / row_width;
    const uint column = gid - local_block * row_width;
    const uint block = first_block + local_block;
    float sum = 0.0f;
    const uint first_row = block * block_size;
    for (uint r = 0; r < block_size; ++r) {
        sum += (float)key_cache[
            (uint64_t)(first_row + r) * row_width + column];
    }
    index_keys[(uint64_t)block * row_width + column] =
        half(sum / (float)block_size);
}

// Score every compressed block for every query head. GQA heads share index
// rows but retain independent selections, which avoids degrading one head to
// the aggregate preference of its siblings.
kernel void kernel_laguna_long_index_scores_f16(
        constant ds4_metal_args_laguna_long_index &args,
        device const float *query,
        device const half  *index_keys,
        device float       *scores,
        ushort lane [[thread_index_in_simdgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint block = tgpig.x;
    const uint head = tgpig.y;
    if (block >= args.block_count || head >= args.n_head ||
        args.n_head_kv == 0u || args.head_dim != 128u) return;
    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head / heads_per_kv;
    const uint row_width = args.n_head_kv * args.head_dim;
    device const float *qh = query + (uint64_t)head * args.head_dim;
    device const half *kh = index_keys +
        (uint64_t)block * row_width +
        (uint64_t)kv_head * args.head_dim;
    float dot = 0.0f;
    for (uint d = lane; d < args.head_dim; d += 32u) {
        dot += qh[d] * (float)kh[d];
    }
    dot = simd_sum(dot);
    if (lane == 0u) {
        scores[(uint64_t)head * args.score_stride + block] =
            dot * args.scale;
    }
}

static inline bool laguna_long_index_better(
        device const float *scores, int a, int b) {
    const float sa = scores[a];
    const float sb = scores[b];
    return sa != sb ? sa > sb : a < b;
}

// Heap selection is identical in spirit to DeepSeek's indexer path. The
// selected ids are sorted in cache order before attention, so row visitation
// stays deterministic.
kernel void kernel_laguna_long_index_topk(
        constant ds4_metal_args_laguna_long_index &args,
        device const float *all_scores,
        device int         *selected,
        threadgroup int    *heap [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]],
        uint head [[threadgroup_position_in_grid]]) {
    if (head >= args.n_head) return;
    const uint keep = min(args.top_k, args.block_count);
    device const float *scores =
        all_scores + (uint64_t)head * args.score_stride;
    if (tid == 0u) {
        uint heap_count = 0u;
        for (uint candidate = 0u;
             candidate < args.block_count; ++candidate) {
            if (heap_count < keep) {
                uint i = heap_count++;
                heap[i] = (int)candidate;
                while (i > 0u) {
                    const uint parent = (i - 1u) >> 1u;
                    if (!laguna_long_index_better(
                            scores, heap[parent], heap[i])) break;
                    const int swap = heap[parent];
                    heap[parent] = heap[i];
                    heap[i] = swap;
                    i = parent;
                }
            } else if (laguna_long_index_better(
                           scores, (int)candidate, heap[0])) {
                heap[0] = (int)candidate;
                uint i = 0u;
                while (true) {
                    const uint left = 2u * i + 1u;
                    const uint right = left + 1u;
                    uint worst = i;
                    if (left < heap_count &&
                        laguna_long_index_better(
                            scores, heap[worst], heap[left])) {
                        worst = left;
                    }
                    if (right < heap_count &&
                        laguna_long_index_better(
                            scores, heap[worst], heap[right])) {
                        worst = right;
                    }
                    if (worst == i) break;
                    const int swap = heap[i];
                    heap[i] = heap[worst];
                    heap[worst] = swap;
                    i = worst;
                }
            }
        }
        // Stable cache-order traversal of the selected blocks.
        for (uint i = 1u; i < keep; ++i) {
            const int value = heap[i];
            uint j = i;
            while (j > 0u && heap[j - 1u] > value) {
                heap[j] = heap[j - 1u];
                --j;
            }
            heap[j] = value;
        }
        device int *out =
            selected + (uint64_t)head * args.top_k;
        for (uint i = 0u; i < args.top_k; ++i) {
            out[i] = i < keep ? heap[i] : -1;
        }
    }
}

static inline void laguna_long_attention_row(
        device const float *qh,
        device const half *key_cache,
        device const half *value_cache,
        uint64_t base,
        uint lane,
        float scale,
        thread float4 &acc,
        thread float &max_score,
        thread float &score_sum) {
    float partial = 0.0f;
    for (uint d = lane; d < 128u; d += 32u) {
        partial += qh[d] * (float)key_cache[base + d];
    }
    const float score = simd_sum(partial) * scale;
    const float next_max = max(max_score, score);
    const float old_scale = max_score == -INFINITY ?
        0.0f : exp(max_score - next_max);
    const float value_scale = exp(score - next_max);
    score_sum = score_sum * old_scale + value_scale;
    const float4 value = float4(
        (float)value_cache[base + lane],
        (float)value_cache[base + lane + 32u],
        (float)value_cache[base + lane + 64u],
        (float)value_cache[base + lane + 96u]);
    acc = acc * old_scale + value * value_scale;
    max_score = next_max;
}

// Sparse global attention: original K/V rows from the selected historic
// blocks, followed by the dense recent tail. Cost is bounded by
// top_k*block_size + raw_count rather than the complete context.
kernel void kernel_laguna_attention_decode_indexed_f16(
        constant ds4_metal_args_laguna_long_index &args,
        device const float *query,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device const int   *selected,
        device float       *out,
        ushort lane [[thread_index_in_simdgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    if (head >= args.n_head || args.n_head_kv == 0u ||
        args.head_dim != 128u) return;
    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head / heads_per_kv;
    const uint row_width = args.n_head_kv * args.head_dim;
    device const float *qh = query + (uint64_t)head * args.head_dim;
    device const int *ids =
        selected + (uint64_t)head * args.top_k;
    float4 acc = float4(0.0f);
    float max_score = -INFINITY;
    float score_sum = 0.0f;

    for (uint pick = 0u; pick < args.top_k; ++pick) {
        const int block = ids[pick];
        if (block < 0 || (uint)block >= args.block_count) continue;
        const uint first = (uint)block * args.block_size;
        for (uint r = 0u; r < args.block_size; ++r) {
            const uint64_t base =
                (uint64_t)(first + r) * row_width +
                (uint64_t)kv_head * args.head_dim;
            laguna_long_attention_row(
                qh, key_cache, value_cache, base, lane, args.scale,
                acc, max_score, score_sum);
        }
    }
    for (uint r = 0u; r < args.raw_count; ++r) {
        const uint row = args.raw_start + r;
        const uint64_t base =
            (uint64_t)row * row_width +
            (uint64_t)kv_head * args.head_dim;
        laguna_long_attention_row(
            qh, key_cache, value_cache, base, lane, args.scale,
            acc, max_score, score_sum);
    }

    const float inv_sum = score_sum > 0.0f ? 1.0f / score_sum : 0.0f;
    const float gate_value = gate[head];
    const float gate_scale = gate_value > 20.0f ?
        gate_value : log(1.0f + exp(gate_value));
    device float *oh = out + (uint64_t)head * args.head_dim;
    oh[lane]       = acc.x * inv_sum * gate_scale;
    oh[lane + 32u] = acc.y * inv_sum * gate_scale;
    oh[lane + 64u] = acc.z * inv_sum * gate_scale;
    oh[lane + 96u] = acc.w * inv_sum * gate_scale;
}

// One SIMD group owns one query head. Queries in a prefill chunk execute in
// parallel, while each query visits keys in causal order. Keys from the current
// chunk come from the staging buffer; older keys come from the persistent ring.
kernel void kernel_laguna_attention_prefill_gqa_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device const half  *staged_key,
        device const half  *staged_value,
        device float       *out,
        ushort lane [[thread_index_in_simdgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u ||
        args.cache_cap == 0u) {
        return;
    }

    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head / heads_per_kv;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint query_pos = args.pos0 + token;
    const uint key_count = min(query_pos + 1u, args.cache_cap);
    const uint key_start = query_pos + 1u - key_count;
    device const float *qh = q +
        ((uint64_t)token * args.n_head + head) * args.head_dim;

    float4 acc = float4(0.0f);
    float max_score = -INFINITY;
    float score_sum = 0.0f;
    for (uint key_pos = key_start; key_pos <= query_pos; key_pos++) {
        const bool current = key_pos >= args.pos0;
        const uint source_row = current ?
            key_pos - args.pos0 : key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)source_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            const float key_value = current ?
                (float)staged_key[kv_base + d] :
                (float)key_cache[kv_base + d];
            partial += qh[d] * key_value;
        }
        const float score = simd_sum(partial) * args.scale;
        const float next_max = max(max_score, score);
        const float old_scale = max_score == -INFINITY ?
            0.0f : exp(max_score - next_max);
        const float value_scale = exp(score - next_max);
        score_sum = score_sum * old_scale + value_scale;
        const uint d0 = lane;
        const float4 value = current ?
            float4((float)staged_value[kv_base + d0],
                   (float)staged_value[kv_base + d0 + 32u],
                   (float)staged_value[kv_base + d0 + 64u],
                   (float)staged_value[kv_base + d0 + 96u]) :
            float4((float)value_cache[kv_base + d0],
                   (float)value_cache[kv_base + d0 + 32u],
                   (float)value_cache[kv_base + d0 + 64u],
                   (float)value_cache[kv_base + d0 + 96u]);
        acc = acc * old_scale + value * value_scale;
        max_score = next_max;
    }

    const float inv_sum = score_sum > 0.0f ? 1.0f / score_sum : 0.0f;
    const float gate_value = gate[(uint64_t)token * args.n_head + head];
    const float gate_scale = gate_value > 20.0f ?
        gate_value : log(1.0f + exp(gate_value));
    device float *oh = out +
        ((uint64_t)token * args.n_head + head) * args.head_dim;
    oh[lane]       = acc.x * inv_sum * gate_scale;
    oh[lane + 32u] = acc.y * inv_sum * gate_scale;
    oh[lane + 64u] = acc.z * inv_sum * gate_scale;
    oh[lane + 96u] = acc.w * inv_sum * gate_scale;
}

// Three adjacent query heads sharing one KV head execute in one SIMD group.
// The per-head reduction and causal key order are unchanged, while each K/V
// value is fetched only once for the group.
kernel void kernel_laguna_attention_prefill_gqa3_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device const half  *staged_key,
        device const half  *staged_value,
        device float       *out,
        ushort lane [[thread_index_in_simdgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head0 = tgpig.x * 3u;
    const uint head1 = head0 + 1u;
    const uint head2 = head0 + 2u;
    const uint token = tgpig.y;
    if (head2 >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u ||
        args.cache_cap == 0u) {
        return;
    }

    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head0 / heads_per_kv;
    if (head2 / heads_per_kv != kv_head) return;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint query_pos = args.pos0 + token;
    const uint key_count = min(query_pos + 1u, args.cache_cap);
    const uint key_start = query_pos + 1u - key_count;
    device const float *qh0 = q +
        ((uint64_t)token * args.n_head + head0) * args.head_dim;
    device const float *qh1 = qh0 + args.head_dim;
    device const float *qh2 = qh1 + args.head_dim;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float max0 = -INFINITY;
    float max1 = -INFINITY;
    float max2 = -INFINITY;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (uint key_pos = key_start; key_pos <= query_pos; key_pos++) {
        const bool current = key_pos >= args.pos0;
        const uint source_row = current ?
            key_pos - args.pos0 : key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)source_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            const float key_value = current ?
                (float)staged_key[kv_base + d] :
                (float)key_cache[kv_base + d];
            partial0 += qh0[d] * key_value;
            partial1 += qh1[d] * key_value;
            partial2 += qh2[d] * key_value;
        }
        const float score0 = simd_sum(partial0) * args.scale;
        const float score1 = simd_sum(partial1) * args.scale;
        const float score2 = simd_sum(partial2) * args.scale;
        const float next_max0 = max(max0, score0);
        const float next_max1 = max(max1, score1);
        const float next_max2 = max(max2, score2);
        const float old_scale0 = max0 == -INFINITY ?
            0.0f : exp(max0 - next_max0);
        const float old_scale1 = max1 == -INFINITY ?
            0.0f : exp(max1 - next_max1);
        const float old_scale2 = max2 == -INFINITY ?
            0.0f : exp(max2 - next_max2);
        const float value_scale0 = exp(score0 - next_max0);
        const float value_scale1 = exp(score1 - next_max1);
        const float value_scale2 = exp(score2 - next_max2);
        sum0 = sum0 * old_scale0 + value_scale0;
        sum1 = sum1 * old_scale1 + value_scale1;
        sum2 = sum2 * old_scale2 + value_scale2;
        const uint d0 = lane;
        const float4 value = current ?
            float4((float)staged_value[kv_base + d0],
                   (float)staged_value[kv_base + d0 + 32u],
                   (float)staged_value[kv_base + d0 + 64u],
                   (float)staged_value[kv_base + d0 + 96u]) :
            float4((float)value_cache[kv_base + d0],
                   (float)value_cache[kv_base + d0 + 32u],
                   (float)value_cache[kv_base + d0 + 64u],
                   (float)value_cache[kv_base + d0 + 96u]);
        acc0 = acc0 * old_scale0 + value * value_scale0;
        acc1 = acc1 * old_scale1 + value * value_scale1;
        acc2 = acc2 * old_scale2 + value * value_scale2;
        max0 = next_max0;
        max1 = next_max1;
        max2 = next_max2;
    }

    const float inv_sum0 = sum0 > 0.0f ? 1.0f / sum0 : 0.0f;
    const float inv_sum1 = sum1 > 0.0f ? 1.0f / sum1 : 0.0f;
    const float inv_sum2 = sum2 > 0.0f ? 1.0f / sum2 : 0.0f;
    const float gate_value0 = gate[(uint64_t)token * args.n_head + head0];
    const float gate_value1 = gate[(uint64_t)token * args.n_head + head1];
    const float gate_value2 = gate[(uint64_t)token * args.n_head + head2];
    const float gate_scale0 = gate_value0 > 20.0f ?
        gate_value0 : log(1.0f + exp(gate_value0));
    const float gate_scale1 = gate_value1 > 20.0f ?
        gate_value1 : log(1.0f + exp(gate_value1));
    const float gate_scale2 = gate_value2 > 20.0f ?
        gate_value2 : log(1.0f + exp(gate_value2));
    device float *oh0 = out +
        ((uint64_t)token * args.n_head + head0) * args.head_dim;
    device float *oh1 = oh0 + args.head_dim;
    device float *oh2 = oh1 + args.head_dim;
    oh0[lane]       = acc0.x * inv_sum0 * gate_scale0;
    oh0[lane + 32u] = acc0.y * inv_sum0 * gate_scale0;
    oh0[lane + 64u] = acc0.z * inv_sum0 * gate_scale0;
    oh0[lane + 96u] = acc0.w * inv_sum0 * gate_scale0;
    oh1[lane]       = acc1.x * inv_sum1 * gate_scale1;
    oh1[lane + 32u] = acc1.y * inv_sum1 * gate_scale1;
    oh1[lane + 64u] = acc1.z * inv_sum1 * gate_scale1;
    oh1[lane + 96u] = acc1.w * inv_sum1 * gate_scale1;
    oh2[lane]       = acc2.x * inv_sum2 * gate_scale2;
    oh2[lane + 32u] = acc2.y * inv_sum2 * gate_scale2;
    oh2[lane + 64u] = acc2.z * inv_sum2 * gate_scale2;
    oh2[lane + 96u] = acc2.w * inv_sum2 * gate_scale2;
}

// Global Laguna layers have six query heads per KV head. Keeping all six in
// one SIMD group preserves each head's reduction order while halving K/V
// traffic relative to the three-head kernel.
kernel void kernel_laguna_attention_prefill_gqa6_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device const half  *staged_key,
        device const half  *staged_value,
        device float       *out,
        ushort lane [[thread_index_in_simdgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head0 = tgpig.x * 6u;
    const uint head5 = head0 + 5u;
    const uint token = tgpig.y;
    if (head5 >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u ||
        args.cache_cap == 0u) {
        return;
    }

    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head0 / heads_per_kv;
    if (head5 / heads_per_kv != kv_head) return;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint query_pos = args.pos0 + token;
    const uint key_count = min(query_pos + 1u, args.cache_cap);
    const uint key_start = query_pos + 1u - key_count;
    device const float *qh0 = q +
        ((uint64_t)token * args.n_head + head0) * args.head_dim;
    device const float *qh1 = qh0 + args.head_dim;
    device const float *qh2 = qh1 + args.head_dim;
    device const float *qh3 = qh2 + args.head_dim;
    device const float *qh4 = qh3 + args.head_dim;
    device const float *qh5 = qh4 + args.head_dim;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    float4 acc4 = float4(0.0f);
    float4 acc5 = float4(0.0f);
    float max0 = -INFINITY;
    float max1 = -INFINITY;
    float max2 = -INFINITY;
    float max3 = -INFINITY;
    float max4 = -INFINITY;
    float max5 = -INFINITY;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    for (uint key_pos = key_start; key_pos <= query_pos; key_pos++) {
        const bool current = key_pos >= args.pos0;
        const uint source_row = current ?
            key_pos - args.pos0 : key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)source_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        float partial3 = 0.0f;
        float partial4 = 0.0f;
        float partial5 = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            const float key_value = current ?
                (float)staged_key[kv_base + d] :
                (float)key_cache[kv_base + d];
            partial0 += qh0[d] * key_value;
            partial1 += qh1[d] * key_value;
            partial2 += qh2[d] * key_value;
            partial3 += qh3[d] * key_value;
            partial4 += qh4[d] * key_value;
            partial5 += qh5[d] * key_value;
        }
        const float score0 = simd_sum(partial0) * args.scale;
        const float score1 = simd_sum(partial1) * args.scale;
        const float score2 = simd_sum(partial2) * args.scale;
        const float score3 = simd_sum(partial3) * args.scale;
        const float score4 = simd_sum(partial4) * args.scale;
        const float score5 = simd_sum(partial5) * args.scale;
        const float next_max0 = max(max0, score0);
        const float next_max1 = max(max1, score1);
        const float next_max2 = max(max2, score2);
        const float next_max3 = max(max3, score3);
        const float next_max4 = max(max4, score4);
        const float next_max5 = max(max5, score5);
        const float old_scale0 = max0 == -INFINITY ?
            0.0f : exp(max0 - next_max0);
        const float old_scale1 = max1 == -INFINITY ?
            0.0f : exp(max1 - next_max1);
        const float old_scale2 = max2 == -INFINITY ?
            0.0f : exp(max2 - next_max2);
        const float old_scale3 = max3 == -INFINITY ?
            0.0f : exp(max3 - next_max3);
        const float old_scale4 = max4 == -INFINITY ?
            0.0f : exp(max4 - next_max4);
        const float old_scale5 = max5 == -INFINITY ?
            0.0f : exp(max5 - next_max5);
        const float value_scale0 = exp(score0 - next_max0);
        const float value_scale1 = exp(score1 - next_max1);
        const float value_scale2 = exp(score2 - next_max2);
        const float value_scale3 = exp(score3 - next_max3);
        const float value_scale4 = exp(score4 - next_max4);
        const float value_scale5 = exp(score5 - next_max5);
        sum0 = sum0 * old_scale0 + value_scale0;
        sum1 = sum1 * old_scale1 + value_scale1;
        sum2 = sum2 * old_scale2 + value_scale2;
        sum3 = sum3 * old_scale3 + value_scale3;
        sum4 = sum4 * old_scale4 + value_scale4;
        sum5 = sum5 * old_scale5 + value_scale5;
        const uint d0 = lane;
        const float4 value = current ?
            float4((float)staged_value[kv_base + d0],
                   (float)staged_value[kv_base + d0 + 32u],
                   (float)staged_value[kv_base + d0 + 64u],
                   (float)staged_value[kv_base + d0 + 96u]) :
            float4((float)value_cache[kv_base + d0],
                   (float)value_cache[kv_base + d0 + 32u],
                   (float)value_cache[kv_base + d0 + 64u],
                   (float)value_cache[kv_base + d0 + 96u]);
        acc0 = acc0 * old_scale0 + value * value_scale0;
        acc1 = acc1 * old_scale1 + value * value_scale1;
        acc2 = acc2 * old_scale2 + value * value_scale2;
        acc3 = acc3 * old_scale3 + value * value_scale3;
        acc4 = acc4 * old_scale4 + value * value_scale4;
        acc5 = acc5 * old_scale5 + value * value_scale5;
        max0 = next_max0;
        max1 = next_max1;
        max2 = next_max2;
        max3 = next_max3;
        max4 = next_max4;
        max5 = next_max5;
    }

    const float inv_sum0 = sum0 > 0.0f ? 1.0f / sum0 : 0.0f;
    const float inv_sum1 = sum1 > 0.0f ? 1.0f / sum1 : 0.0f;
    const float inv_sum2 = sum2 > 0.0f ? 1.0f / sum2 : 0.0f;
    const float inv_sum3 = sum3 > 0.0f ? 1.0f / sum3 : 0.0f;
    const float inv_sum4 = sum4 > 0.0f ? 1.0f / sum4 : 0.0f;
    const float inv_sum5 = sum5 > 0.0f ? 1.0f / sum5 : 0.0f;
    const uint64_t gate_base = (uint64_t)token * args.n_head + head0;
    const float gate_value0 = gate[gate_base];
    const float gate_value1 = gate[gate_base + 1u];
    const float gate_value2 = gate[gate_base + 2u];
    const float gate_value3 = gate[gate_base + 3u];
    const float gate_value4 = gate[gate_base + 4u];
    const float gate_value5 = gate[gate_base + 5u];
    const float gate_scale0 = gate_value0 > 20.0f ?
        gate_value0 : log(1.0f + exp(gate_value0));
    const float gate_scale1 = gate_value1 > 20.0f ?
        gate_value1 : log(1.0f + exp(gate_value1));
    const float gate_scale2 = gate_value2 > 20.0f ?
        gate_value2 : log(1.0f + exp(gate_value2));
    const float gate_scale3 = gate_value3 > 20.0f ?
        gate_value3 : log(1.0f + exp(gate_value3));
    const float gate_scale4 = gate_value4 > 20.0f ?
        gate_value4 : log(1.0f + exp(gate_value4));
    const float gate_scale5 = gate_value5 > 20.0f ?
        gate_value5 : log(1.0f + exp(gate_value5));
    device float *oh0 = out + gate_base * args.head_dim;
    device float *oh1 = oh0 + args.head_dim;
    device float *oh2 = oh1 + args.head_dim;
    device float *oh3 = oh2 + args.head_dim;
    device float *oh4 = oh3 + args.head_dim;
    device float *oh5 = oh4 + args.head_dim;
    oh0[lane]       = acc0.x * inv_sum0 * gate_scale0;
    oh0[lane + 32u] = acc0.y * inv_sum0 * gate_scale0;
    oh0[lane + 64u] = acc0.z * inv_sum0 * gate_scale0;
    oh0[lane + 96u] = acc0.w * inv_sum0 * gate_scale0;
    oh1[lane]       = acc1.x * inv_sum1 * gate_scale1;
    oh1[lane + 32u] = acc1.y * inv_sum1 * gate_scale1;
    oh1[lane + 64u] = acc1.z * inv_sum1 * gate_scale1;
    oh1[lane + 96u] = acc1.w * inv_sum1 * gate_scale1;
    oh2[lane]       = acc2.x * inv_sum2 * gate_scale2;
    oh2[lane + 32u] = acc2.y * inv_sum2 * gate_scale2;
    oh2[lane + 64u] = acc2.z * inv_sum2 * gate_scale2;
    oh2[lane + 96u] = acc2.w * inv_sum2 * gate_scale2;
    oh3[lane]       = acc3.x * inv_sum3 * gate_scale3;
    oh3[lane + 32u] = acc3.y * inv_sum3 * gate_scale3;
    oh3[lane + 64u] = acc3.z * inv_sum3 * gate_scale3;
    oh3[lane + 96u] = acc3.w * inv_sum3 * gate_scale3;
    oh4[lane]       = acc4.x * inv_sum4 * gate_scale4;
    oh4[lane + 32u] = acc4.y * inv_sum4 * gate_scale4;
    oh4[lane + 64u] = acc4.z * inv_sum4 * gate_scale4;
    oh4[lane + 96u] = acc4.w * inv_sum4 * gate_scale4;
    oh5[lane]       = acc5.x * inv_sum5 * gate_scale5;
    oh5[lane + 32u] = acc5.y * inv_sum5 * gate_scale5;
    oh5[lane + 64u] = acc5.z * inv_sum5 * gate_scale5;
    oh5[lane + 96u] = acc5.w * inv_sum5 * gate_scale5;
}

struct ds4_metal_args_laguna_attention {
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t cache_cap;
    uint32_t key_start;
    uint32_t key_count;
    float    scale;
    uint32_t pad0;
};

struct ds4_metal_args_laguna_gqa3_decode {
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t key_count0;
    uint32_t n_tokens;
    uint32_t nsg;
    uint32_t nwg;
    float    scale;
};

// Global layers split keys across 32 workgroups. Evaluate three query heads
// sharing one KV head together so each K/V row is loaded once instead of three
// times. DFlash verifier rows use the second grid dimension while retaining
// the exact one-row arithmetic and established gated reduction.
kernel void kernel_laguna_attention_decode_gqa3_split_f16(
        constant ds4_metal_args_laguna_gqa3_decode &args,
        device const float *q,
        device const half  *key_cache,
        device const half  *value_cache,
        device float       *tmp,
        threadgroup float  *scratch [[threadgroup(0)]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group_u [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head0 = tgpig.x * 3u;
    const uint head2 = head0 + 2u;
    const uint token = tgpig.y;
    const uint iwg = tgpig.z;
    const uint simd_group = (uint)simd_group_u;
    const uint key_count = args.key_count0 + token;
    if (head2 >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u ||
        key_count == 0u ||
        args.nsg == 0u || simd_group >= args.nsg || iwg >= args.nwg) {
        return;
    }

    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head0 / heads_per_kv;
    if (head2 / heads_per_kv != kv_head) return;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint64_t query_base =
        (uint64_t)token * args.n_head * args.head_dim;
    device const float *qh0 = q + query_base +
                              (uint64_t)head0 * args.head_dim;
    device const float *qh1 = qh0 + args.head_dim;
    device const float *qh2 = qh1 + args.head_dim;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float max0 = -INFINITY;
    float max1 = -INFINITY;
    float max2 = -INFINITY;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    const uint first = iwg * args.nsg + simd_group;
    const uint stride = args.nwg * args.nsg;
    for (uint i = first; i < key_count; i += stride) {
        const uint64_t kv_base =
            (uint64_t)i * cache_width +
            (uint64_t)kv_head * args.head_dim;
        const uint d0 = lane;
        const float4 key = float4((float)key_cache[kv_base + d0],
                                  (float)key_cache[kv_base + d0 + 32u],
                                  (float)key_cache[kv_base + d0 + 64u],
                                  (float)key_cache[kv_base + d0 + 96u]);
        const float score0 = simd_sum(dot(float4(qh0[d0],
                                                   qh0[d0 + 32u],
                                                   qh0[d0 + 64u],
                                                   qh0[d0 + 96u]), key)) * args.scale;
        const float score1 = simd_sum(dot(float4(qh1[d0],
                                                   qh1[d0 + 32u],
                                                   qh1[d0 + 64u],
                                                   qh1[d0 + 96u]), key)) * args.scale;
        const float score2 = simd_sum(dot(float4(qh2[d0],
                                                   qh2[d0 + 32u],
                                                   qh2[d0 + 64u],
                                                   qh2[d0 + 96u]), key)) * args.scale;
        const float next_max0 = max(max0, score0);
        const float next_max1 = max(max1, score1);
        const float next_max2 = max(max2, score2);
        const float old_scale0 = max0 == -INFINITY ? 0.0f : exp(max0 - next_max0);
        const float old_scale1 = max1 == -INFINITY ? 0.0f : exp(max1 - next_max1);
        const float old_scale2 = max2 == -INFINITY ? 0.0f : exp(max2 - next_max2);
        const float value_scale0 = exp(score0 - next_max0);
        const float value_scale1 = exp(score1 - next_max1);
        const float value_scale2 = exp(score2 - next_max2);
        sum0 = sum0 * old_scale0 + value_scale0;
        sum1 = sum1 * old_scale1 + value_scale1;
        sum2 = sum2 * old_scale2 + value_scale2;
        const float4 value = float4((float)value_cache[kv_base + d0],
                                    (float)value_cache[kv_base + d0 + 32u],
                                    (float)value_cache[kv_base + d0 + 64u],
                                    (float)value_cache[kv_base + d0 + 96u]);
        acc0 = acc0 * old_scale0 + value * value_scale0;
        acc1 = acc1 * old_scale1 + value * value_scale1;
        acc2 = acc2 * old_scale2 + value * value_scale2;
        max0 = next_max0;
        max1 = next_max1;
        max2 = next_max2;
    }

    threadgroup float *partial_max = scratch;
    threadgroup float *partial_sum = partial_max + 3u * args.nsg;
    threadgroup float *partial_value = partial_sum + 3u * args.nsg;
    const uint slots[3] = {simd_group,
                           args.nsg + simd_group,
                           2u * args.nsg + simd_group};
    const float maxima[3] = {max0, max1, max2};
    const float sums[3] = {sum0, sum1, sum2};
    const float4 values[3] = {acc0, acc1, acc2};
    for (uint h = 0u; h < 3u; h++) {
        const uint slot = slots[h];
        if (lane == 0u) {
            partial_max[slot] = maxima[h];
            partial_sum[slot] = sums[h];
        }
        const uint base = slot * args.head_dim + lane;
        partial_value[base] = values[h].x;
        partial_value[base + 32u] = values[h].y;
        partial_value[base + 64u] = values[h].z;
        partial_value[base + 96u] = values[h].w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group != 0u) return;
    const uint nrows = args.n_tokens * args.n_head;
    device float *stats =
        tmp + (uint64_t)nrows * args.head_dim * args.nwg;
    for (uint h = 0u; h < 3u; h++) {
        const uint slot_base = h * args.nsg;
        float global_max = partial_max[slot_base];
        for (uint sg = 1u; sg < args.nsg; sg++) {
            global_max = max(global_max, partial_max[slot_base + sg]);
        }
        float merged_sum = 0.0f;
        float4 merged = float4(0.0f);
        for (uint sg = 0u; sg < args.nsg; sg++) {
            const uint slot = slot_base + sg;
            const float weight = partial_sum[slot] > 0.0f ?
                exp(partial_max[slot] - global_max) : 0.0f;
            merged_sum += partial_sum[slot] * weight;
            const uint base = slot * args.head_dim + lane;
            merged.x += partial_value[base] * weight;
            merged.y += partial_value[base + 32u] * weight;
            merged.z += partial_value[base + 64u] * weight;
            merged.w += partial_value[base + 96u] * weight;
        }

        const uint row = token * args.n_head + head0 + h;
        const uint64_t row_base =
            (uint64_t)row * args.head_dim * args.nwg;
        const uint dims[4] = {lane, lane + 32u, lane + 64u, lane + 96u};
        const float outputs[4] = {merged.x, merged.y, merged.z, merged.w};
        for (uint j = 0u; j < 4u; j++) {
            const uint d = dims[j];
            tmp[row_base + (d / 4u) * args.nwg * 4u + iwg * 4u + d % 4u] =
                outputs[j];
        }
        if (lane == 0u) {
            const uint64_t stat = (uint64_t)row * 2u * args.nwg + 2u * iwg;
            stats[stat] = merged_sum;
            stats[stat + 1u] = global_max;
        }
    }
}

// One threadgroup owns one query head. Short histories retain the original
// single-SIMD online reduction. Longer histories are striped over eight SIMD
// groups, then their independently normalized partials are merged in
// threadgroup memory. This keeps decode latency from growing serially with the
// absolute context position while preserving the short-context arithmetic.
kernel void kernel_laguna_attention_decode_gqa_f16(
        constant ds4_metal_args_laguna_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device float       *out,
        threadgroup float  *scratch [[threadgroup(0)]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group_u [[simdgroup_index_in_threadgroup]],
        uint head [[threadgroup_position_in_grid]]) {
    constexpr uint split_simd_groups = 8u;
    constexpr uint split_threshold = 256u;
    if (head >= args.n_head || args.n_head_kv == 0u ||
        args.head_dim != 128u || args.key_count == 0u) {
        return;
    }
    const uint simd_group = (uint)simd_group_u;
    const bool split = args.key_count > split_threshold;
    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head / heads_per_kv;
    const uint cache_width = args.n_head_kv * args.head_dim;
    device const float *qh = q + (uint64_t)head * args.head_dim;

    float4 acc = float4(0.0f);
    float max_score = -INFINITY;
    float score_sum = 0.0f;
    const uint key_first = split ? simd_group : 0u;
    const uint key_stride = split ? split_simd_groups : 1u;
    for (uint i = key_first; i < args.key_count; i += key_stride) {
        const uint key_pos = args.key_start + i;
        const uint cache_row = key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)cache_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            partial += qh[d] * (float)key_cache[kv_base + d];
        }
        const float score = simd_sum(partial) * args.scale;
        const float next_max = max(max_score, score);
        const float old_scale = max_score == -INFINITY ?
            0.0f : exp(max_score - next_max);
        const float value_scale = exp(score - next_max);
        score_sum = score_sum * old_scale + value_scale;
        const uint d0 = lane;
        acc.x = acc.x * old_scale +
            value_scale * (float)value_cache[kv_base + d0];
        acc.y = acc.y * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 32u];
        acc.z = acc.z * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 64u];
        acc.w = acc.w * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 96u];
        max_score = next_max;
    }

    threadgroup float *partial_max = scratch;
    threadgroup float *partial_sum = partial_max + split_simd_groups;
    threadgroup float *partial_value = partial_sum + split_simd_groups;
    if (lane == 0u) {
        partial_max[simd_group] = max_score;
        partial_sum[simd_group] = score_sum;
    }
    const uint value_base = simd_group * args.head_dim;
    partial_value[value_base + lane] = acc.x;
    partial_value[value_base + lane + 32u] = acc.y;
    partial_value[value_base + lane + 64u] = acc.z;
    partial_value[value_base + lane + 96u] = acc.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group != 0u) return;

    float4 merged = acc;
    float merged_sum = score_sum;
    if (split) {
        float global_max = partial_max[0];
        for (uint sg = 1u; sg < split_simd_groups; sg++) {
            global_max = max(global_max, partial_max[sg]);
        }
        merged = float4(0.0f);
        merged_sum = 0.0f;
        for (uint sg = 0u; sg < split_simd_groups; sg++) {
            const float weight = partial_sum[sg] > 0.0f ?
                exp(partial_max[sg] - global_max) : 0.0f;
            merged_sum += partial_sum[sg] * weight;
            const uint base = sg * args.head_dim + lane;
            merged.x += partial_value[base] * weight;
            merged.y += partial_value[base + 32u] * weight;
            merged.z += partial_value[base + 64u] * weight;
            merged.w += partial_value[base + 96u] * weight;
        }
    }

    const float inv_sum = merged_sum > 0.0f ? 1.0f / merged_sum : 0.0f;
    const float gate_value = gate[head];
    const float gate_scale = gate_value > 20.0f ?
        gate_value : log(1.0f + exp(gate_value));
    device float *oh = out + (uint64_t)head * args.head_dim;
    oh[lane]       = merged.x * inv_sum * gate_scale;
    oh[lane + 32u] = merged.y * inv_sum * gate_scale;
    oh[lane + 64u] = merged.z * inv_sum * gate_scale;
    oh[lane + 96u] = merged.w * inv_sum * gate_scale;
}

// Batch independent pre-wrap verifier rows in the grid's Y dimension. This is
// deliberately the same per-head reduction as decode; only command encoding
// and dispatch are shared across rows.
kernel void kernel_laguna_attention_decode_rows_gqa_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device float       *out,
        threadgroup float  *scratch [[threadgroup(0)]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group_u [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint split_simd_groups = 8u;
    constexpr uint split_threshold = 256u;
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u) {
        return;
    }
    const uint key_count = min(args.pos0 + token + 1u, args.cache_cap);
    if (key_count == 0u) return;
    const uint key_start = args.pos0 + token + 1u - key_count;
    const uint simd_group = (uint)simd_group_u;
    const bool split = key_count > split_threshold;
    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head / heads_per_kv;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint64_t row = (uint64_t)token * args.n_head + head;
    device const float *qh = q + row * args.head_dim;

    float4 acc = float4(0.0f);
    float max_score = -INFINITY;
    float score_sum = 0.0f;
    const uint key_first = split ? simd_group : 0u;
    const uint key_stride = split ? split_simd_groups : 1u;
    for (uint i = key_first; i < key_count; i += key_stride) {
        const uint key_pos = key_start + i;
        const uint cache_row = key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)cache_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            partial += qh[d] * (float)key_cache[kv_base + d];
        }
        const float score = simd_sum(partial) * args.scale;
        const float next_max = max(max_score, score);
        const float old_scale = max_score == -INFINITY ?
            0.0f : exp(max_score - next_max);
        const float value_scale = exp(score - next_max);
        score_sum = score_sum * old_scale + value_scale;
        const uint d0 = lane;
        acc.x = acc.x * old_scale +
            value_scale * (float)value_cache[kv_base + d0];
        acc.y = acc.y * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 32u];
        acc.z = acc.z * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 64u];
        acc.w = acc.w * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 96u];
        max_score = next_max;
    }

    threadgroup float *partial_max = scratch;
    threadgroup float *partial_sum = partial_max + split_simd_groups;
    threadgroup float *partial_value = partial_sum + split_simd_groups;
    if (lane == 0u) {
        partial_max[simd_group] = max_score;
        partial_sum[simd_group] = score_sum;
    }
    const uint value_base = simd_group * args.head_dim;
    partial_value[value_base + lane] = acc.x;
    partial_value[value_base + lane + 32u] = acc.y;
    partial_value[value_base + lane + 64u] = acc.z;
    partial_value[value_base + lane + 96u] = acc.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group != 0u) return;

    float4 merged = acc;
    float merged_sum = score_sum;
    if (split) {
        float global_max = partial_max[0];
        for (uint sg = 1u; sg < split_simd_groups; sg++) {
            global_max = max(global_max, partial_max[sg]);
        }
        merged = float4(0.0f);
        merged_sum = 0.0f;
        for (uint sg = 0u; sg < split_simd_groups; sg++) {
            const float weight = partial_sum[sg] > 0.0f ?
                exp(partial_max[sg] - global_max) : 0.0f;
            merged_sum += partial_sum[sg] * weight;
            const uint base = sg * args.head_dim + lane;
            merged.x += partial_value[base] * weight;
            merged.y += partial_value[base + 32u] * weight;
            merged.z += partial_value[base + 64u] * weight;
            merged.w += partial_value[base + 96u] * weight;
        }
    }

    const float inv_sum = merged_sum > 0.0f ? 1.0f / merged_sum : 0.0f;
    const float gate_value = gate[row];
    const float gate_scale = gate_value > 20.0f ?
        gate_value : log(1.0f + exp(gate_value));
    device float *oh = out + row * args.head_dim;
    oh[lane]       = merged.x * inv_sum * gate_scale;
    oh[lane + 32u] = merged.y * inv_sum * gate_scale;
    oh[lane + 64u] = merged.z * inv_sum * gate_scale;
    oh[lane + 96u] = merged.w * inv_sum * gate_scale;
}

// Laguna's split-K decode attention consumes the generic FlashAttention
// partial layout, but every reduced head is immediately multiplied by its
// learned gate. Apply that epilogue before the final store so decode does not
// write and reread the complete head buffer in a second dispatch.
kernel void kernel_laguna_flash_attn_reduce_gate_f32(
        constant ds4_metal_args_flash_attn_ext_vec_reduce &args,
        device const char  *htmp,
        device       char  *dst,
        device const float *gate,
        uint tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t row = tgpig;
    device const float *stats = (device const float *)htmp +
        (uint64_t)args.nrows * DV * NWG;

    float sum = stats[row * (2 * NWG) + 2 * lane];
    float max_value = stats[row * (2 * NWG) + 2 * lane + 1];
    const float global_max = simd_max(max_value);
    const float scale = exp(max_value - global_max);
    sum = simd_sum(sum * scale);
    const float inv_sum = sum == 0.0f ? 0.0f : 1.0f / sum;

    const float gate_value = gate[row];
    const float gate_scale = gate_value > 20.0f ?
        gate_value : log(1.0f + exp(gate_value));
    const short DV4 = DV / 4;
    device const float4 *partials =
        (device const float4 *)htmp + row * DV4 * NWG;
    device volatile float4 *out =
        (device volatile float4 *)dst + row * DV4;

    for (short i = simd_group; i < DV4; i += NWG) {
        const float4 value = simd_sum(partials[i * NWG + lane] * scale);
        if (lane == 0) {
            // Keep the same F32 materialization boundary as the unfused
            // reducer followed by the gate kernel. The volatile round trip
            // prevents fast-math from reassociating the two multiplies.
            out[i] = value * inv_sum;
            out[i] = out[i] * gate_scale;
        }
    }

#undef NWG
#undef DV
}
