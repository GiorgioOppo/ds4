#include <metal_stdlib>
using namespace metal;

inline float deq_e4m3(uchar b) {
    uint sign = (uint)(b >> 7) & 1u;
    uint exp = (uint)(b >> 3) & 0xFu;
    uint mant = (uint)b & 0x7u;
    if (exp == 0u && mant == 0u) return sign != 0u ? -0.0f : 0.0f;
    if (exp == 0xFu && mant == 0x7u) return NAN;
    if (exp == 0u) {
        float v = (float)mant * 0x1p-9f;
        return sign != 0u ? -v : v;
    }
    uint bits = (sign << 31) | ((exp + 120u) << 23) | (mant << 20);
    return as_type<float>(bits);
}

inline float deq_e2m1(uchar nib) {
    uchar mag = nib & 7u;
    float v;
    switch (mag) {
        case 0u: v = 0.0f; break;
        case 1u: v = 0.5f; break;
        case 2u: v = 1.0f; break;
        case 3u: v = 1.5f; break;
        case 4u: v = 2.0f; break;
        case 5u: v = 3.0f; break;
        case 6u: v = 4.0f; break;
        default: v = 6.0f; break;
    }
    return ((nib & 8u) != 0u) ? -v : v;
}

// fp4_gemm — FP8 activation × FP4 weight GEMM with mixed scaling.
//
// Mirrors `fp4_gemm_kernel` from
// Reference/inference/kernel.py lines 441–536.
//
// Layout:
//   A:        [M, K]    fp8_e4m3
//   A_sc:     [M, K/128] f32       per-128 act scale along K
//   B:        [N, K/2]  fp4_e2m1   packed two nibbles per byte along K
//   B_sc:     [N, K/32] f32        per-32 weight scale along K
//   C:        [M, N]    f32        output
//
// FP4 nibble layout per byte (matches the upstream `float4_e2m1fn_x2` ordering):
//   - the LOW nibble (`byte & 0xF`) is the value at K-index 2*i
//   - the HIGH nibble (`byte >> 4`) is the value at K-index 2*i + 1
//
// Naive: one thread per output cell, K reduction unrolled by 32-element
// weight blocks (4 per activation block). Scales applied at block boundary.

constant uint BLOCK_K_ACT = 128;
constant uint BLOCK_K_W   = 32;

kernel void gemm_fp8_fp4_to_f32(
    device const uchar*  A      [[buffer(0)]],
    device const float*  A_sc   [[buffer(1)]],
    device const uchar*  B      [[buffer(2)]],
    device const float*  B_sc   [[buffer(3)]],
    device float*        C      [[buffer(4)]],
    constant uint3&      dims   [[buffer(5)]],   // (M, N, K)
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gid.y, col = gid.x;
    if (row >= M || col >= N) return;

    uint blocksKAct = K / BLOCK_K_ACT;
    uint blocksKW   = K / BLOCK_K_W;
    float acc = 0.0f;

    for (uint wb = 0; wb < blocksKW; wb++) {
        float a_scale = A_sc[row * blocksKAct + wb / (BLOCK_K_ACT / BLOCK_K_W)];
        float b_scale = B_sc[col * blocksKW + wb];

        float block_acc = 0.0f;
        uint k0 = wb * BLOCK_K_W;
        for (uint i = 0; i < BLOCK_K_W / 2; i++) {
            uchar packed = B[col * (K / 2u) + k0 / 2u + i];
            float blo = deq_e2m1(packed & 0xFu);
            float bhi = deq_e2m1(packed >> 4);
            float alo = deq_e4m3(A[row * K + k0 + 2u*i]);
            float ahi = deq_e4m3(A[row * K + k0 + 2u*i + 1u]);
            block_acc += alo * blo + ahi * bhi;
        }
        acc += block_acc * a_scale * b_scale;
    }
    C[row * N + col] = acc;
}
