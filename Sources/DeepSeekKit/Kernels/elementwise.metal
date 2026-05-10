#include <metal_stdlib>
using namespace metal;

// y = silu(g) * u   (SwiGLU body, with g and u already projected from x)
kernel void silu_mul_f32(
    device const float* g [[buffer(0)]],
    device const float* u [[buffer(1)]],
    device float*       y [[buffer(2)]],
    constant uint&      N [[buffer(3)]],
    uint                gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    float v = g[gid];
    float silu = v / (1.0f + exp(-v));
    y[gid] = silu * u[gid];
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
