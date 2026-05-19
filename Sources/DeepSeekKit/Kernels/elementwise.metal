#include <metal_stdlib>
using namespace metal;

// y = silu(g_clipped) * u_clipped
// Mirrors `Expert.forward` in Reference/inference/model.py:596-606:
//   if swiglu_limit > 0:
//       up = clamp(up,   min=-limit, max=limit)
//       gate = clamp(gate,           max=limit)   ← only upper bound
//   x = silu(gate) * up
//
// With `limit <= 0` the clipping is disabled and we fall back to the
// original SwiGLU.
kernel void silu_mul_f32(
    device const float* g [[buffer(0)]],
    device const float* u [[buffer(1)]],
    device float*       y [[buffer(2)]],
    constant uint&      N [[buffer(3)]],
    constant float&     limit [[buffer(4)]],
    uint                gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    float v = g[gid];
    float uv = u[gid];
    if (limit > 0.0f) {
        v = min(v, limit);                  // gate.clamp(max=limit)
        uv = clamp(uv, -limit, limit);      // up.clamp(min=-limit, max=limit)
    }
    float silu = v / (1.0f + exp(-v));
    y[gid] = silu * uv;
}

// out[i] = in[i] * scale[i % cols]
//
// Per-channel multiply on a row-major 2-D tensor flattened to 1-D
// at dispatch time. `cols` is the channel dimension stride (= the
// last axis of the logical tensor). Used by `Linear`'s AWQ /
// SmoothQuant inverseChannelScale path: when the calibrated quant
// migrated per-channel difficulty to the weight via
// `weight *= s[c]`, the activation must be pre-multiplied by
// `1/s[c]` for the GEMM to recover the original output.
kernel void channel_scale_f32(
    device const float* in    [[buffer(0)]],
    device const float* scale [[buffer(1)]],   // [cols]
    device float*       out   [[buffer(2)]],
    constant uint2&     dims  [[buffer(3)]],   // (N_total_elements, cols)
    uint gid [[thread_position_in_grid]]
) {
    uint N    = dims.x;
    uint cols = dims.y;
    if (gid >= N) return;
    out[gid] = in[gid] * scale[gid % cols];
}

// y += alpha * x
kernel void axpy_f32(
    device float*       y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    constant float&     alpha [[buffer(2)]],
    constant uint&      N [[buffer(3)]],
    uint                gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    y[gid] += alpha * x[gid];
}

// y = alpha * x
kernel void scale_f32(
    device float*       y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    constant float&     alpha [[buffer(2)]],
    constant uint&      N [[buffer(3)]],
    uint                gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    y[gid] = alpha * x[gid];
}

// in-place residual add: y += x
kernel void add_inplace_f32(
    device float*       y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    constant uint&      N [[buffer(2)]],
    uint                gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    y[gid] += x[gid];
}

// In-place F32 → BF16 → F32 round-trip. Truncates the F32 mantissa down to
// 7 bits (BF16 mantissa width), then writes the dequantized value back into
// the same buffer. Used to mimic the reference's BF16 activation propagation:
// the V4 model is trained with every sub-layer output cast to BF16 before
// feeding the next block, so each step accumulates ~0.8% per-element noise
// that the network learned to tolerate. Our forward propagates in F32
// throughout, which is technically more precise but moves the activations
// off the training distribution. This kernel restores that quantization
// noise at the residual stream boundary.
//
// Uses Metal's native `bfloat` for proper RTNE rounding (compiler matches
// PyTorch's `.to(torch.bfloat16)`).
kernel void bf16_round_trip_f32_inplace(
    device float*  x [[buffer(0)]],
    constant uint& N [[buffer(1)]],
    uint           gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    bfloat b = (bfloat)x[gid];
    x[gid] = (float)b;
}
