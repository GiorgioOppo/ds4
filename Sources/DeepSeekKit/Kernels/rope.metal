#include <metal_stdlib>
using namespace metal;

// Rotary Position Embedding, applied to the head dimension (interleaved pair layout).
// x: [tokens, heads, headDim] f32
kernel void rope_f32(
    device float*    x        [[buffer(0)]],
    constant uint4&  dims     [[buffer(1)]],   // (tokens, heads, headDim, posOffset)
    constant float&  theta    [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint t = gid.x, h = gid.y, i = gid.z;
    uint T = dims.x, H = dims.y, D = dims.z;
    uint posOffset = dims.w;

    if (t >= T || h >= H || i >= D / 2) return;

    uint pos = t + posOffset;
    float freq = 1.0f / pow(theta, (2.0f * float(i)) / float(D));
    float angle = float(pos) * freq;
    float c = cos(angle);
    float s = sin(angle);

    uint base = (t * H + h) * D;
    float a = x[base + 2*i];
    float b = x[base + 2*i + 1];
    x[base + 2*i]     = a * c - b * s;
    x[base + 2*i + 1] = a * s + b * c;
}
