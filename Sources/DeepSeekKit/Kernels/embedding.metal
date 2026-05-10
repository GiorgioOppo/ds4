#include <metal_stdlib>
using namespace metal;

// Embedding lookup + HC expand. Used by Transformer.forward.
//
// embed_lookup_f32: out[n, d] = weight[ids[n], d]
// One thread per (n, d) cell.
kernel void embed_lookup_f32(
    device const float* weight  [[buffer(0)]],   // [vocab, D]
    device const int*   ids     [[buffer(1)]],   // [N]
    device float*       out_    [[buffer(2)]],   // [N, D]
    constant uint2&     dims    [[buffer(3)]],   // (N, D)
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, n = gid.y;
    uint N = dims.x, D = dims.y;
    if (n >= N || d >= D) return;
    int id = ids[n];
    if (id < 0) {
        out_[n * D + d] = 0.0f;
        return;
    }
    out_[n * D + d] = weight[(uint)id * D + d];
}

// hc_expand: y[n, h, d] = x[n, d] for each h in 0..hc.
// Mirrors model.py:805 `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`.
kernel void hc_expand_f32(
    device const float* x     [[buffer(0)]],   // [N, D]
    device float*       y     [[buffer(1)]],   // [N, HC, D]
    constant uint3&     dims  [[buffer(2)]],   // (N, HC, D)
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, h = gid.y, n = gid.z;
    uint N = dims.x, HC = dims.y, D = dims.z;
    if (n >= N || h >= HC || d >= D) return;
    y[(n * HC + h) * D + d] = x[n * D + d];
}
