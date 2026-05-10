#include <metal_stdlib>
using namespace metal;

// Two small composition kernels used by Compressor.forward.
//
// Both are straight ports of PyTorch broadcast / reduction ops from
// Reference/inference/model.py:316-376.

// broadcast_add_4d_2d: y[b, ns, r, c] += w[r, c]
// Used for `score += self.ape` (model.py:338) where ape: [ratio, coff*head_dim]
// and score: [b, num_blocks, ratio, coff*head_dim].
kernel void broadcast_add_4d_2d_f32(
    device float*       y     [[buffer(0)]],
    device const float* w     [[buffer(1)]],
    constant uint4&     dims  [[buffer(2)]],   // (B, NS, R, C)
    uint3 gid [[thread_position_in_grid]]
) {
    uint c = gid.x, r = gid.y, bn = gid.z;
    uint B = dims.x, NS = dims.y, R = dims.z, C = dims.w;
    uint b = bn / NS;
    uint ns = bn - b * NS;
    if (b >= B || ns >= NS || r >= R || c >= C) return;
    y[((b * NS + ns) * R + r) * C + c] += w[r * C + c];
}

// weighted_sum_axis2: y[b, ns, c] = Σ_r kv[b, ns, r, c] * score[b, ns, r, c]
// This is the combined element-wise mul + reduce-sum over the ratio axis,
// matching `(kv * score.softmax(dim=2)).sum(dim=2)` (model.py:342).
// Caller must apply the softmax to `score` beforehand.
kernel void weighted_sum_axis2_f32(
    device const float* kv     [[buffer(0)]],
    device const float* score  [[buffer(1)]],
    device float*       y      [[buffer(2)]],
    constant uint4&     dims   [[buffer(3)]],   // (B, NS, R, C)
    uint3 gid [[thread_position_in_grid]]
) {
    uint c = gid.x, ns = gid.y, b = gid.z;
    uint B = dims.x, NS = dims.y, R = dims.z, C = dims.w;
    if (b >= B || ns >= NS || c >= C) return;
    float acc = 0.0f;
    uint base = (b * NS + ns) * R * C + c;
    for (uint r = 0; r < R; r++) {
        acc += kv[base + r * C] * score[base + r * C];
    }
    y[(b * NS + ns) * C + c] = acc;
}
