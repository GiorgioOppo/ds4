#include <metal_stdlib>
using namespace metal;

// INT4 W4A16 GEMM: symmetric per-row × per-128-group quantized weights
// packed two-per-byte (low nibble = index 2k, high nibble = index 2k+1).
// Mirrors the wiring of gemm_int8_w8a16_to_f32 in int8_gemm.metal, with
// nibble unpacking + 4-bit sign extension in place of the int8 load.
//
// Layout:
//   A:        [M, K]            f32   or  bfloat — activations
//   W:        [N, K/2]          uchar — packed signed 4-bit (two's complement)
//                                       low nibble = column 2k, high = 2k+1
//                                       value range: [-8, 7]
//   W_sc:     [N, K/128]        half  — per-row × per-128-group scales
//   C:        [M, N]            f32   — output
//
// The K dimension is required to be divisible by 128. K%2==0 is implied
// (128 is even). Enforced at quant time by Int4Quant.shouldQuantizeToInt4
// and by a precondition in Linear.swift at dispatch time.
//
// Sign extension: a Metal `int(byte_nibble) - 16*(nibble>>3)` would also
// work but the bit-shift trick (cast nibble to i32, shift left 28, shift
// right 28) generates a single arithmetic-right pair and is clearer.

constant uint INT4_BLOCK_K = 128;

inline int sext_i4(uint nibble4) {
    // Sign-extend a 4-bit value (in the low 4 bits) to a full int.
    int x = int(nibble4);
    return (x << 28) >> 28;
}

kernel void gemm_int4_w4a16_to_f32(
    device const float* A      [[buffer(0)]],
    device const uchar* W      [[buffer(1)]],   // packed nibbles, [N, K/2]
    device const half*  W_sc   [[buffer(2)]],
    device float*       C      [[buffer(3)]],
    constant uint3&     dims   [[buffer(4)]],   // (M, N, K)
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gid.y, col = gid.x;
    if (row >= M || col >= N) return;

    uint blocksK = K / INT4_BLOCK_K;
    uint rowBytes = K / 2;
    float acc = 0.0f;

    for (uint kb = 0; kb < blocksK; kb++) {
        float w_scale = float(W_sc[col * blocksK + kb]);
        float block_acc = 0.0f;
        uint k0 = kb * INT4_BLOCK_K;
        for (uint k = 0; k < INT4_BLOCK_K; k += 2) {
            uint kAbs = k0 + k;
            uchar byte = W[col * rowBytes + kAbs / 2];
            float wLo = float(sext_i4(byte & 0x0F));
            float wHi = float(sext_i4((byte >> 4) & 0x0F));
            float aLo = A[row * K + kAbs];
            float aHi = A[row * K + kAbs + 1];
            block_acc += aLo * wLo + aHi * wHi;
        }
        acc += block_acc * w_scale;
    }
    C[row * N + col] = acc;
}

// BF16-input variant; identical shape, just casts activations on read.
kernel void gemm_int4_w4a16_bf16_to_f32(
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

    uint blocksK = K / INT4_BLOCK_K;
    uint rowBytes = K / 2;
    float acc = 0.0f;

    for (uint kb = 0; kb < blocksK; kb++) {
        float w_scale = float(W_sc[col * blocksK + kb]);
        float block_acc = 0.0f;
        uint k0 = kb * INT4_BLOCK_K;
        for (uint k = 0; k < INT4_BLOCK_K; k += 2) {
            uint kAbs = k0 + k;
            uchar byte = W[col * rowBytes + kAbs / 2];
            float wLo = float(sext_i4(byte & 0x0F));
            float wHi = float(sext_i4((byte >> 4) & 0x0F));
            float aLo = float(A[row * K + kAbs]);
            float aHi = float(A[row * K + kAbs + 1]);
            block_acc += aLo * wLo + aHi * wHi;
        }
        acc += block_acc * w_scale;
    }
    C[row * N + col] = acc;
}
