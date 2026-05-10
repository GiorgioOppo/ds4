#include <metal_stdlib>
using namespace metal;

// fp8_gemm — FP8 × FP8 matrix multiply with per-128 block scaling.
//
// Port of `fp8_gemm_kernel` from
// Original/DeepSeek-V4-Pro/inference/kernel.py.
//
// Computes:  C[M, N] = A_fp8[M, K] @ B_fp8[N, K]^T   (out: BF16 or FP32)
// with:
//   scales_a: [M, K/128] in E8M0 — one scale per 1×128 row block
//   scales_b: [N/128, K/128] in E8M0 — one scale per 128×128 weight block
//
// Reference tile sizes: block_M=32, block_N=128, block_K=128, num_stages=4.
// Inner loop:
//   for k in pipelined(K/128):
//     load A_tile[32,128] FP8, B_tile[128,128] FP8
//     dequant via FP8→FP16 cast in shader (Metal has no native FP8)
//     simdgroup_matrix bf16 GEMM into C_local
//     scale_c[i] = scales_a[i, k] * scales_b[bx, k]   (FP32)
//     C_local_accum[i, j] += C_local[i, j] * scale_c[i]
//
// NOT IMPLEMENTED. Required machinery:
//   - FP8-E4M3 → FP16 dequant function (see Quantization.swift for the
//     scalar reference; on GPU use a 256-entry lookup table in constant memory
//     or a bit-twiddling implementation)
//   - simdgroup_matrix-based BF16 GEMM (Apple Silicon M3/M4 simdgroup
//     instructions: simdgroup_load, simdgroup_multiply_accumulate)
//   - swizzle-aware threadgroup memory layout to avoid bank conflicts
//   - 2-stage accumulator: per-tile FP32 accum, scaled, then folded into the
//     outer FP32 accumulator (matches reference's C_local / C_local_accum)
//
// One subtle point: the reference does NOT pre-dequant the FP8 to BF16 in
// shared memory; it relies on tilelang's FP8 GEMM. On Metal we have no FP8
// matrix instructions, so we MUST cast on load.

kernel void fp8_gemm_unimplemented(
    device float* dst [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) dst[0] = NAN;
}
