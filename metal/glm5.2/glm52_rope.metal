// GLM 5.2 — RoPE della coda query/K e del prefisso indexer
// Parte della libreria concatenata: vedi MetalRuntime.kernelFiles per l'ordine.

#include <metal_stdlib>
using namespace metal;

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
