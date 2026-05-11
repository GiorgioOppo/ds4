#include <metal_stdlib>
using namespace metal;

// INT8 W8A16 GEMM: symmetric per-row, per-group-128 quantized weights with
// FP32 activations and FP32 output. Mirrors the wiring of `gemm_fp8_to_f32`
// in fp8_gemm.metal but with `char` weights and an F16 scale tensor.
//
// Layout:
//   A:        [M, K]        f32     — activations (unquantized)
//   W:        [N, K]        int8    — weights, signed [-127, 127]
//   W_sc:     [N, K/128]    half    — per-row × per-128-group scales
//   C:        [M, N]        f32     — output
//
// The K dimension is required to be divisible by 128 — this is enforced at
// quant time by `Int8Quant.shouldQuantizeToInt8` in the converter and by a
// precondition in `Linear.swift` at dispatch time.
//
// One thread per output cell. The scale is constant within each 128-element
// K block so it's hoisted out of the inner accumulation loop.

constant uint INT8_BLOCK_K = 128;

kernel void gemm_int8_w8a16_to_f32(
    device const float* A      [[buffer(0)]],
    device const char*  W      [[buffer(1)]],
    device const half*  W_sc   [[buffer(2)]],
    device float*       C      [[buffer(3)]],
    constant uint3&     dims   [[buffer(4)]],   // (M, N, K)
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gid.y, col = gid.x;
    if (row >= M || col >= N) return;

    uint blocksK = K / INT8_BLOCK_K;
    float acc = 0.0f;

    for (uint kb = 0; kb < blocksK; kb++) {
        float w_scale = float(W_sc[col * blocksK + kb]);
        float block_acc = 0.0f;
        uint k0 = kb * INT8_BLOCK_K;
        for (uint k = 0; k < INT8_BLOCK_K; k++) {
            float a = A[row * K + k0 + k];
            float w = float(W[col * K + k0 + k]);
            block_acc += a * w;
        }
        acc += block_acc * w_scale;
    }
    C[row * N + col] = acc;
}

// BF16-input variant: same as above but activations are bfloat instead of
// float. Mirrors the existing `gemm_bf16_to_f32` / `gemm_f32_bf16_to_f32`
// pairing for the dense BF16 path so we can call INT8 GEMM with either
// upstream activation dtype.
kernel void gemm_int8_w8a16_bf16_to_f32(
    device const bfloat* A      [[buffer(0)]],
    device const char*   W      [[buffer(1)]],
    device const half*   W_sc   [[buffer(2)]],
    device float*        C      [[buffer(3)]],
    constant uint3&      dims   [[buffer(4)]],   // (M, N, K)
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gid.y, col = gid.x;
    if (row >= M || col >= N) return;

    uint blocksK = K / INT8_BLOCK_K;
    float acc = 0.0f;

    for (uint kb = 0; kb < blocksK; kb++) {
        float w_scale = float(W_sc[col * blocksK + kb]);
        float block_acc = 0.0f;
        uint k0 = kb * INT8_BLOCK_K;
        for (uint k = 0; k < INT8_BLOCK_K; k++) {
            float a = float(A[row * K + k0 + k]);
            float w = float(W[col * K + k0 + k]);
            block_acc += a * w;
        }
        acc += block_acc * w_scale;
    }
    C[row * N + col] = acc;
}
