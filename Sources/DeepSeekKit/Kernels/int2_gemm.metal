#include <metal_stdlib>
using namespace metal;

// INT2 W2A16 GEMM: symmetric per-row × per-128-group quantized weights
// packed four-per-byte (LSB-first: index 4k in bits [1:0], 4k+1 in
// [3:2], 4k+2 in [5:4], 4k+3 in [7:6]). Two's-complement signed 2-bit
// values, range [-2, 1]. Mirrors gemm_int4_w4a16_to_f32 with denser
// packing and 2→8 sign extension instead of 4→8.
//
// Layout:
//   A:        [M, K]            f32 or bfloat — activations
//   W:        [N, K/4]          uchar         — packed signed 2-bit
//   W_sc:     [N, K/128]        half          — per-row × per-128 scales
//   C:        [M, N]            f32           — output
//
// Quantization range [-2, 1] is two's complement: 00=0, 01=+1, 10=-2,
// 11=-1. The kernel sign-extends via (value<<30)>>30, a single
// arithmetic-right pair on Apple Silicon.
//
// K%128 (and therefore K%4) is enforced at quant time by
// Int2Quant.shouldQuantizeToInt2 and at Linear-dispatch time.

constant uint INT2_BLOCK_K = 128;

inline int sext_i2(uint twoBits) {
    int x = int(twoBits);
    return (x << 30) >> 30;
}

kernel void gemm_int2_w2a16_to_f32(
    device const float* A      [[buffer(0)]],
    device const uchar* W      [[buffer(1)]],   // packed 2-bit, [N, K/4]
    device const half*  W_sc   [[buffer(2)]],
    device float*       C      [[buffer(3)]],
    constant uint3&     dims   [[buffer(4)]],   // (M, N, K)
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gid.y, col = gid.x;
    if (row >= M || col >= N) return;

    uint blocksK = K / INT2_BLOCK_K;
    uint rowBytes = K / 4;
    float acc = 0.0f;

    for (uint kb = 0; kb < blocksK; kb++) {
        float w_scale = float(W_sc[col * blocksK + kb]);
        float block_acc = 0.0f;
        uint k0 = kb * INT2_BLOCK_K;
        // Each byte yields 4 values; step in groups of 4.
        for (uint k = 0; k < INT2_BLOCK_K; k += 4) {
            uint kAbs = k0 + k;
            uchar byte = W[col * rowBytes + kAbs / 4];
            float w0 = float(sext_i2( byte        & 0x3));
            float w1 = float(sext_i2((byte >> 2)  & 0x3));
            float w2 = float(sext_i2((byte >> 4)  & 0x3));
            float w3 = float(sext_i2((byte >> 6)  & 0x3));
            float a0 = A[row * K + kAbs    ];
            float a1 = A[row * K + kAbs + 1];
            float a2 = A[row * K + kAbs + 2];
            float a3 = A[row * K + kAbs + 3];
            block_acc += a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3;
        }
        acc += block_acc * w_scale;
    }
    C[row * N + col] = acc;
}

kernel void gemm_int2_w2a16_bf16_to_f32(
    device const bfloat* A      [[buffer(0)]],
    device const uchar*  W      [[buffer(1)]],
    device const half*   W_sc   [[buffer(2)]],
    device float*        C      [[buffer(3)]],
    constant uint3&      dims   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gid.y, col = gid.x;
    if (row >= M || col >= N) return;

    uint blocksK = K / INT2_BLOCK_K;
    uint rowBytes = K / 4;
    float acc = 0.0f;

    for (uint kb = 0; kb < blocksK; kb++) {
        float w_scale = float(W_sc[col * blocksK + kb]);
        float block_acc = 0.0f;
        uint k0 = kb * INT2_BLOCK_K;
        for (uint k = 0; k < INT2_BLOCK_K; k += 4) {
            uint kAbs = k0 + k;
            uchar byte = W[col * rowBytes + kAbs / 4];
            float w0 = float(sext_i2( byte        & 0x3));
            float w1 = float(sext_i2((byte >> 2)  & 0x3));
            float w2 = float(sext_i2((byte >> 4)  & 0x3));
            float w3 = float(sext_i2((byte >> 6)  & 0x3));
            float a0 = float(A[row * K + kAbs    ]);
            float a1 = float(A[row * K + kAbs + 1]);
            float a2 = float(A[row * K + kAbs + 2]);
            float a3 = float(A[row * K + kAbs + 3]);
            block_acc += a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3;
        }
        acc += block_acc * w_scale;
    }
    C[row * N + col] = acc;
}
