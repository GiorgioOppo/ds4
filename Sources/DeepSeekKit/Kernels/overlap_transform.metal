#include <metal_stdlib>
using namespace metal;

// Compressor.overlap_transform from Reference/inference/model.py:307-314.
//
// Input  shape: [B, S, R, 2D]   (R == compress_ratio, D == head_dim)
// Output shape: [B, S, 2R, D]
//
// For each (b, s, j, di):
//   if j >= R:                          (second R block)
//     out[b, s, j, di] = in[b, s, j - R, D + di]
//   else if s > 0:                      (first R block — from PREVIOUS s)
//     out[b, s, j, di] = in[b, s - 1, j, di]
//   else:                               (s == 0, j < R — fill value)
//     out[b, 0, j, di] = pad_value

kernel void overlap_transform_f32(
    device const float* in_       [[buffer(0)]],
    device float*       out_      [[buffer(1)]],
    constant uint4&     dims      [[buffer(2)]],   // (B, S, R, D)
    constant float&     padValue  [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint di = gid.x, j = gid.y, ns = gid.z;
    uint B = dims.x, S = dims.y, R = dims.z, D = dims.w;
    uint b = ns / S;
    uint s = ns - b * S;
    if (b >= B || s >= S || j >= 2u * R || di >= D) return;

    uint outIdx = ((b * S + s) * (2u * R) + j) * D + di;
    if (j >= R) {
        uint inIdx = ((b * S + s) * R + (j - R)) * (2u * D) + (D + di);
        out_[outIdx] = in_[inIdx];
    } else if (s > 0u) {
        uint inIdx = ((b * S + (s - 1u)) * R + j) * (2u * D) + di;
        out_[outIdx] = in_[inIdx];
    } else {
        out_[outIdx] = padValue;
    }
}
