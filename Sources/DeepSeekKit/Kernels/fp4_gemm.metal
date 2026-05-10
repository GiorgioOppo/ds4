#include <metal_stdlib>
using namespace metal;

// fp4_gemm — FP8 activation × FP4 weight GEMM with mixed scaling.
//
// Port of `fp4_gemm_kernel` from
// Reference/inference/kernel.py.
//
// Used by MoE expert weights only (`expert_dtype == "fp4"`).
//
// Computes:  C[M, N] = A_fp8[M, K] @ B_fp4[N, K]^T   (out: BF16 or FP32)
// with:
//   A: FP8-E4M3 [M, K], scales [M, K/128] in E8M0  (act per-128 along K)
//   B: FP4-E2M1 [N, K/2] packed (2 nibbles per byte, packed along K),
//      logically [N, K]; scales [N, K/32] in E8M0  (weight per-32 along K)
//
// Reference tile sizes: block_M=32, block_N=128, block_K=32 (= weight block).
// 128/32 = 4 sub-blocks per activation scale group.
//
// Inner loop:
//   for k in pipelined(K/32):
//     load A_tile[32, 32] FP8
//     load B_tile[128, 32] FP4 (packed: actually 128×16 bytes)
//     dequant FP4 → FP16 in threadgroup memory (unpack nibbles + table lookup)
//     dequant FP8 → FP16 on the fly during GEMM accumulation
//     simdgroup_matrix BF16 GEMM into C_local
//     scale_c[i, j] = scales_a[i, k/4] * scales_b[j, k]
//     C_local_accum += C_local * scale_c
//
// NOT IMPLEMENTED. Same machinery as fp8_gemm.metal plus:
//   - FP4-E2M1 nibble unpack: 16-entry lookup table (8 magnitudes × sign)
//   - tighter K-block (32) to match the weight scale granularity, so each
//     accumulator tile naturally folds in the per-32 weight scale
//   - A's scale changes every 4 inner iterations, B's every iteration
//
// Memory note: FP4 weights are 4× smaller than BF16, so for 1.6T-param V4-Pro
// at FP4 the expert weights alone are ~800GB → still won't fit a Mac.
// Targeting V4-Flash (284B / 13B activated) is the realistic on-device option.

kernel void fp4_gemm_unimplemented(
    device float* dst [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) dst[0] = NAN;
}
