#include <metal_stdlib>
using namespace metal;

// FP8-E4M3FN dequant — duplicated from act_quant.metal (each .metal is its own
// compilation unit). See dtype semantics in Quantization.swift.
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

// E8M0 (unsigned-exponent-only) dequant: value = 2^(b - 127). Encoded as a
// Float32 whose mantissa is zero and whose biased exponent equals `b`, i.e.
// the byte goes straight into the 8 exponent bits of an IEEE 754 single-
// precision word. Mirrors `deqE8M0` in DTypePacking.swift.
inline float deq_e8m0(uchar b) {
    if (b == 0xFFu) return NAN;
    return as_type<float>(((uint)b) << 23);
}

// fp8_gemm — FP8 × FP8 matrix multiply with per-128 block scaling.
//
// Mirrors `fp8_gemm_kernel` from
// Reference/inference/kernel.py lines 203–273.
//
// Layout:
//   A:        [M, K] fp8_e4m3
//   A_scale:  [M, K/128] f32   — produced by ActQuant (dynamic, runtime)
//   B:        [N, K] fp8_e4m3
//   B_scale:  [N/128, K/128] uchar — UE8M0 weight scale from disk
//   C:        [M, N] f32 output
//
// Naive tiled implementation: one thread per output cell, scalar loop over
// K-blocks of 128 elements. Apple Silicon has no native FP8 GEMM so each
// FP8 byte is dequantized through `deq_e4m3` in-shader before multiplying.
//
// Weight scales come from disk as UE8M0 (1 byte each — `scale_fmt: ue8m0`
// in DeepSeek-V4-HF). Reading them as `device const float*` would mis-
// interpret 4 consecutive bytes as one f32, producing silent garbage and
// out-of-bounds reads near the end of the buffer. Dequant inline via
// `deq_e8m0`.

constant uint BLOCK_K = 128;
constant uint BLOCK_N_FP8 = 128;

kernel void gemm_fp8_to_f32(
    device const uchar*  A      [[buffer(0)]],
    device const float*  A_sc   [[buffer(1)]],
    device const uchar*  B      [[buffer(2)]],
    device const uchar*  B_sc   [[buffer(3)]],
    device float*        C      [[buffer(4)]],
    constant uint3&      dims   [[buffer(5)]],   // (M, N, K)
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gid.y, col = gid.x;
    if (row >= M || col >= N) return;

    uint blocksK = K / BLOCK_K;
    float acc = 0.0f;

    for (uint kb = 0; kb < blocksK; kb++) {
        float a_scale = A_sc[row * blocksK + kb];
        float b_scale = deq_e8m0(B_sc[(col / BLOCK_N_FP8) * blocksK + kb]);

        float block_acc = 0.0f;
        uint k0 = kb * BLOCK_K;
        for (uint k = 0; k < BLOCK_K; k++) {
            float a = deq_e4m3(A[row * K + k0 + k]);
            float b = deq_e4m3(B[col * K + k0 + k]);
            block_acc += a * b;
        }
        acc += block_acc * a_scale * b_scale;
    }
    C[row * N + col] = acc;
}
