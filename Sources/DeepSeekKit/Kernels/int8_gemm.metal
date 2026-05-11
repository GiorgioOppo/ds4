#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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

// ============================================================================
// simdgroup_matrix variants — INT8 → BF16 dequant-on-load + 8×8 MMA.
// ============================================================================
// Apple Silicon doesn't expose int8 MMA, so we dequantize each 128-element
// K-block into a bf16 tile in threadgroup memory once per block, then run
// simdgroup_matrix on the bf16 tiles. The per-128 scale is hoisted out of
// the inner loop; bf16 retains plenty of precision for `int8 * fp16_scale`.
//
// One threadgroup = one simdgroup (32 threads). Each produces a 32×32 C
// block as a 4×4 grid of float 8×8 accumulators.
//
// Requirements (caller's responsibility, gated in Linear.swift):
//   M % 32 == 0, N % 32 == 0, K % 128 == 0.

constant uint SG_INT8_TM = 32;
constant uint SG_INT8_TN = 32;
constant uint SG_INT8_K_BLOCK = 128;

// Shared core: produces Cmat[4][4] given an Atg / Btg already staged for
// one K block. Inline so each variant can stage A from its own dtype.
inline void int8_sg_mma_kblock(
    threadgroup bfloat A_tg[SG_INT8_TM][SG_INT8_K_BLOCK],
    threadgroup bfloat B_tg[SG_INT8_K_BLOCK][SG_INT8_TN],
    thread simdgroup_matrix<float, 8, 8> Cmat[4][4]
) {
    simdgroup_matrix<bfloat, 8, 8> Atile[4];
    simdgroup_matrix<bfloat, 8, 8> Btile[4];
    for (uint kt = 0; kt < SG_INT8_K_BLOCK / 8; kt++) {
        for (uint i = 0; i < 4; i++) {
            simdgroup_load(Atile[i],
                           &A_tg[i * 8][kt * 8], SG_INT8_K_BLOCK,
                           ulong2(0, 0), false);
            simdgroup_load(Btile[i],
                           &B_tg[kt * 8][i * 8], SG_INT8_TN,
                           ulong2(0, 0), false);
        }
        for (uint i = 0; i < 4; i++)
            for (uint j = 0; j < 4; j++)
                simdgroup_multiply_accumulate(Cmat[i][j], Atile[i], Btile[j], Cmat[i][j]);
    }
}

kernel void gemm_int8_w8a16_to_f32_sg(
    device const float* A      [[buffer(0)]],
    device const char*  W      [[buffer(1)]],
    device const half*  W_sc   [[buffer(2)]],
    device float*       C      [[buffer(3)]],
    constant uint3&     dims   [[buffer(4)]],
    uint  tid  [[thread_index_in_simdgroup]],
    uint2 gpos [[threadgroup_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row0 = gpos.y * SG_INT8_TM;
    uint col0 = gpos.x * SG_INT8_TN;
    uint blocksK = K / SG_INT8_K_BLOCK;

    threadgroup bfloat A_tg[SG_INT8_TM][SG_INT8_K_BLOCK];
    threadgroup bfloat B_tg[SG_INT8_K_BLOCK][SG_INT8_TN];

    simdgroup_matrix<float, 8, 8> Cmat[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            Cmat[i][j] = simdgroup_matrix<float, 8, 8>(0);

    for (uint kb = 0; kb < blocksK; kb++) {
        uint k0 = kb * SG_INT8_K_BLOCK;

        // Stage A row `tid` into A_tg[tid][:] (f32 → bf16).
        for (uint k = 0; k < SG_INT8_K_BLOCK; k++) {
            A_tg[tid][k] = bfloat(A[(row0 + tid) * K + k0 + k]);
        }
        // Stage B column `tid` into B_tg[:][tid] with dequant.
        // One scale per col per K-block; hoist out of the inner loop.
        float scale_f = float(W_sc[(col0 + tid) * blocksK + kb]);
        for (uint k = 0; k < SG_INT8_K_BLOCK; k++) {
            float w = float(W[(col0 + tid) * K + k0 + k]);
            B_tg[k][tid] = bfloat(w * scale_f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int8_sg_mma_kblock(A_tg, B_tg, Cmat);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            simdgroup_store(Cmat[i][j],
                            C + (row0 + i * 8) * N + col0 + j * 8, N,
                            ulong2(0, 0), false);
}

// BF16-input variant. Same shape, but no f32→bf16 cast on the A stage.
kernel void gemm_int8_w8a16_bf16_to_f32_sg(
    device const bfloat* A      [[buffer(0)]],
    device const char*   W      [[buffer(1)]],
    device const half*   W_sc   [[buffer(2)]],
    device float*        C      [[buffer(3)]],
    constant uint3&      dims   [[buffer(4)]],
    uint  tid  [[thread_index_in_simdgroup]],
    uint2 gpos [[threadgroup_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row0 = gpos.y * SG_INT8_TM;
    uint col0 = gpos.x * SG_INT8_TN;
    uint blocksK = K / SG_INT8_K_BLOCK;

    threadgroup bfloat A_tg[SG_INT8_TM][SG_INT8_K_BLOCK];
    threadgroup bfloat B_tg[SG_INT8_K_BLOCK][SG_INT8_TN];

    simdgroup_matrix<float, 8, 8> Cmat[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            Cmat[i][j] = simdgroup_matrix<float, 8, 8>(0);

    for (uint kb = 0; kb < blocksK; kb++) {
        uint k0 = kb * SG_INT8_K_BLOCK;

        for (uint k = 0; k < SG_INT8_K_BLOCK; k++) {
            A_tg[tid][k] = A[(row0 + tid) * K + k0 + k];
        }
        float scale_f = float(W_sc[(col0 + tid) * blocksK + kb]);
        for (uint k = 0; k < SG_INT8_K_BLOCK; k++) {
            float w = float(W[(col0 + tid) * K + k0 + k]);
            B_tg[k][tid] = bfloat(w * scale_f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int8_sg_mma_kblock(A_tg, B_tg, Cmat);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            simdgroup_store(Cmat[i][j],
                            C + (row0 + i * 8) * N + col0 + j * 8, N,
                            ulong2(0, 0), false);
}
