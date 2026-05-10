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

// compressor_overlap_concat: gathers a [B, 2R, D] view from a state buffer
// laid out [B, 2R, 2D]. For row index `row < R` we keep the FIRST half of
// the last dim (low D); for `row >= R` we keep the SECOND half (high D).
// Mirrors the inline `torch.cat([state[:, :R, :D], state[:, R:, D:]], dim=1)`
// in Reference/inference/model.py:350-351.
kernel void compressor_overlap_concat_f32(
    device const float* state [[buffer(0)]],   // [B, 2R, 2D]
    device float*       out_  [[buffer(1)]],   // [B, 2R, D]
    constant uint3&     dims  [[buffer(2)]],   // (B, R, D)
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, row = gid.y, b = gid.z;
    uint B = dims.x, R = dims.y, D = dims.z;
    if (b >= B || row >= 2u * R || d >= D) return;
    uint inRowBase = (b * (2u * R) + row) * (2u * D);
    uint inColOff = row < R ? d : (D + d);
    out_[(b * (2u * R) + row) * D + d] = state[inRowBase + inColOff];
}

// compressor_state_shift: state[:, :R, :] = state[:, R:, :].
// Used after an overlap emit to slide the second half down.
// Allocate a temporary out [B, R, 2D] then blit-copy back to avoid aliased
// reads during writes — handled host-side. This kernel is the pure copy.
kernel void compressor_state_shift_copy_f32(
    device const float* state [[buffer(0)]],
    device float*       tmp   [[buffer(1)]],
    constant uint3&     dims  [[buffer(2)]],   // (B, R, twoD)
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, row = gid.y, b = gid.z;
    uint B = dims.x, R = dims.y, TWOD = dims.z;
    if (b >= B || row >= R || d >= TWOD) return;
    // Source: state[b, R + row, d]    →    Dest: tmp[b, row, d]
    uint srcOff = (b * (2u * R) + (R + row)) * TWOD + d;
    uint dstOff = (b * R + row) * TWOD + d;
    tmp[dstOff] = state[srcOff];
}
