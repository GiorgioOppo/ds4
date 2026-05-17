#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// INT8 W8A8 GEMM: int8 activations × int8 weights → f32 output.
// Mirrors the W8A16 wiring (int8_gemm.metal) but reads int8 on BOTH sides
// and applies the activation scale on top of the weight scale.
//
// Layout:
//   A_q:     [M, K]        char (int8)   — quantized activations
//   A_sc:    [M, K/128]    float          — per-row × per-128-group act scales
//                                           (f32 to match `act_quant_int8`'s
//                                           output; NOT pow2-rounded)
//   W:       [N, K]        char (int8)   — weights (same as W8A16)
//   W_sc:    [N, K/128]    half           — per-row × per-128-group weight
//                                           scales (same as W8A16)
//   C:       [M, N]        float          — output
//
// K is required to be divisible by 128 — this matches `Int8Quant.shouldQuantizeToInt8`
// for weights and the `act_quant_int8` block size. Caller must precondition.
//
// Two variants:
//   gemm_int8_w8a8_to_f32      naive, one thread per output cell. Accumulates
//                              int8×int8→int32 per block, dequant at block
//                              boundary. Memory-bound; useful baseline.
//   gemm_int8_w8a8_to_f32_sg   simdgroup_matrix path mirroring W8A16's SG
//                              kernel. Stages both sides as bfloat after
//                              dequant in threadgroup memory (Apple Silicon
//                              has no native int8 matrix MMA).

constant uint INT8_W8A8_BLOCK_K = 128;

kernel void gemm_int8_w8a8_to_f32(
    device const char*  A_q      [[buffer(0)]],
    device const float* A_sc     [[buffer(1)]],
    device const char*  W        [[buffer(2)]],
    device const half*  W_sc     [[buffer(3)]],
    device float*       C        [[buffer(4)]],
    constant uint3&     dims     [[buffer(5)]],   // (M, N, K)
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gid.y, col = gid.x;
    if (row >= M || col >= N) return;

    uint blocksK = K / INT8_W8A8_BLOCK_K;
    float acc = 0.0f;

    for (uint kb = 0; kb < blocksK; kb++) {
        // Hoist both scales out of the inner loop.
        float a_scale = A_sc[row * blocksK + kb];
        float w_scale = float(W_sc[col * blocksK + kb]);
        // Inner reduction in int32 — at 128 elements per block with int8
        // values bounded by ±127, the worst-case partial sum is
        // 128 * 127 * 127 = ~2M, well within int32 range.
        int block_acc = 0;
        uint k0 = kb * INT8_W8A8_BLOCK_K;
        for (uint k = 0; k < INT8_W8A8_BLOCK_K; k++) {
            int a = int(A_q[row * K + k0 + k]);
            int w = int(W[col * K + k0 + k]);
            block_acc += a * w;
        }
        acc += float(block_acc) * a_scale * w_scale;
    }
    C[row * N + col] = acc;
}

// ============================================================================
// simdgroup_matrix variant.
// ============================================================================
//
// Apple Silicon has no int8 MMA. We stage both A and W as bfloat in
// threadgroup memory (after per-block dequant), then run
// simdgroup_matrix<bfloat, 8, 8>. The two scales are applied during the
// dequant-on-stage step, NOT during the accumulation, so the simdgroup MMA
// sees pre-multiplied operands and accumulates a single tile.
//
// One threadgroup = one simdgroup (32 threads). Each produces a 32×32 C
// block as a 4×4 grid of float 8×8 accumulators. Mirrors `gemm_int8_w8a16_to_f32_sg`.
//
// Requirements (caller's responsibility, gated in Linear.swift):
//   M % 32 == 0, N % 32 == 0, K % 128 == 0.

constant uint SG_W8A8_TM = 32;
constant uint SG_W8A8_TN = 32;
constant uint SG_W8A8_K_BLOCK = 128;

kernel void gemm_int8_w8a8_to_f32_sg(
    device const char*  A_q      [[buffer(0)]],
    device const float* A_sc     [[buffer(1)]],
    device const char*  W        [[buffer(2)]],
    device const half*  W_sc     [[buffer(3)]],
    device float*       C        [[buffer(4)]],
    constant uint3&     dims     [[buffer(5)]],
    uint  tid  [[thread_index_in_simdgroup]],
    uint2 gpos [[threadgroup_position_in_grid]]
) {
    // M is implicit in the grid dispatch (gpos.y).
    uint N = dims.y, K = dims.z;
    uint row0 = gpos.y * SG_W8A8_TM;
    uint col0 = gpos.x * SG_W8A8_TN;
    uint blocksK = K / SG_W8A8_K_BLOCK;

    threadgroup bfloat A_tg[SG_W8A8_TM][SG_W8A8_K_BLOCK];
    threadgroup bfloat B_tg[SG_W8A8_K_BLOCK][SG_W8A8_TN];

    simdgroup_matrix<float, 8, 8> Cmat[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            Cmat[i][j] = simdgroup_matrix<float, 8, 8>(0);

    for (uint kb = 0; kb < blocksK; kb++) {
        uint k0 = kb * SG_W8A8_K_BLOCK;

        // Stage A row `tid` into A_tg with dequant: int8 → bfloat * a_scale.
        // One scale per row per K-block; hoist out of the inner loop.
        float a_scale = A_sc[(row0 + tid) * blocksK + kb];
        for (uint k = 0; k < SG_W8A8_K_BLOCK; k++) {
            float a = float(int(A_q[(row0 + tid) * K + k0 + k]));
            A_tg[tid][k] = bfloat(a * a_scale);
        }
        // Stage B column `tid` into B_tg with dequant: int8 → bfloat * w_scale.
        float w_scale = float(W_sc[(col0 + tid) * blocksK + kb]);
        for (uint k = 0; k < SG_W8A8_K_BLOCK; k++) {
            float w = float(W[(col0 + tid) * K + k0 + k]);
            B_tg[k][tid] = bfloat(w * w_scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reuse the same 4×4 MMA pattern of int8_gemm.metal's
        // `int8_sg_mma_kblock`. Local inline to avoid cross-file include
        // (metallib build doesn't do that).
        simdgroup_matrix<bfloat, 8, 8> Atile[4];
        simdgroup_matrix<bfloat, 8, 8> Btile[4];
        for (uint kt = 0; kt < SG_W8A8_K_BLOCK / 8; kt++) {
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(Atile[i],
                               &A_tg[i * 8][kt * 8], SG_W8A8_K_BLOCK,
                               ulong2(0, 0), false);
                simdgroup_load(Btile[i],
                               &B_tg[kt * 8][i * 8], SG_W8A8_TN,
                               ulong2(0, 0), false);
            }
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(Cmat[i][j], Atile[i], Btile[j], Cmat[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            simdgroup_store(Cmat[i][j],
                            C + (row0 + i * 8) * N + col0 + j * 8, N,
                            ulong2(0, 0), false);
}
