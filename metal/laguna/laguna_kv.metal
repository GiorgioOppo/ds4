// Laguna S 2.1 — F16 ring KV store (ported from the reference laguna-s2.1
// branch, head 448d569). Defines the shared prefill-attention args struct;
// Part of the concatenated library: see MetalRuntime.kernelFiles for the order.

#include <metal_stdlib>
using namespace metal;

// laguna_attention.metal relies on it and must follow this file.

struct ds4_metal_args_laguna_kv_store {
    uint32_t cache_cap;
    uint32_t cache_row;
    uint32_t n_head_kv;
    uint32_t head_dim;
};

kernel void kernel_laguna_store_kv_f16(
        constant ds4_metal_args_laguna_kv_store &args,
        device const float *k,
        device const float *v,
        device half *key_cache,
        device half *value_cache,
        uint gid [[thread_position_in_grid]]) {
    const uint width = args.n_head_kv * args.head_dim;
    if (gid >= width || args.cache_row >= args.cache_cap) return;
    const uint64_t dst = (uint64_t)args.cache_row * width + gid;
    key_cache[dst] = (half)k[gid];
    value_cache[dst] = (half)v[gid];
}

struct ds4_metal_args_laguna_prefill_attention {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    float    scale;
    uint32_t pad0;
};

// Before the sliding window wraps, verifier rows occupy distinct cache slots.
// Store the complete speculative block at once; each query still limits its
// key count, so later rows cannot become visible to earlier queries.
kernel void kernel_laguna_store_kv_rows_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *k,
        device const float *v,
        device half *key_cache,
        device half *value_cache,
        uint gid [[thread_position_in_grid]]) {
    const uint width = args.n_head_kv * args.head_dim;
    const uint values = args.n_tokens * width;
    if (gid >= values) return;
    const uint token = gid / width;
    const uint col = gid - token * width;
    const uint cache_row = (args.pos0 + token) % args.cache_cap;
    const uint64_t dst = (uint64_t)cache_row * width + col;
    key_cache[dst] = (half)k[gid];
    value_cache[dst] = (half)v[gid];
}

// Stage the current chunk as f16 before attention. This preserves the same KV
// precision as decode without overwriting sliding-window rows that early
// queries in the chunk still need.
kernel void kernel_laguna_stage_kv_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *k,
        device const float *v,
        device half *staged_key,
        device half *staged_value,
        uint gid [[thread_position_in_grid]]) {
    const uint width = args.n_head_kv * args.head_dim;
    const uint values = args.n_tokens * width;
    if (gid >= values) return;
    staged_key[gid] = (half)k[gid];
    staged_value[gid] = (half)v[gid];
}

kernel void kernel_laguna_commit_kv_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const half *staged_key,
        device const half *staged_value,
        device half *key_cache,
        device half *value_cache,
        uint gid [[thread_position_in_grid]]) {
    const uint width = args.n_head_kv * args.head_dim;
    const uint values = args.n_tokens * width;
    if (gid >= values) return;
    const uint token = gid / width;
    const uint col = gid - token * width;
    const uint cache_row = (args.pos0 + token) % args.cache_cap;
    const uint64_t dst = (uint64_t)cache_row * width + col;
    key_cache[dst] = staged_key[gid];
    value_cache[dst] = staged_value[gid];
}
