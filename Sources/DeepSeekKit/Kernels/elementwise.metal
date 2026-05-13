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
