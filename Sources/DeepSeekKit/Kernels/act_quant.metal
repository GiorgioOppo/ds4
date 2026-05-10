#include <metal_stdlib>
using namespace metal;

// act_quant — block-wise activation quantization.
//
// Port of `act_quant_kernel` and `fp4_quant_kernel` from
// Original/DeepSeek-V4-Pro/inference/kernel.py.
//
// FP8 path: input BF16 [M, N] -> output FP8-E4M3 [M, N] + scales E8M0 [M, N/128].
//   For each block of `block_size` (default 128) along the last dim:
//     amax  = max(|x|, 1e-4)
//     scale = round-pow2(amax / fp8_max)         (when scale_fmt == "ue8m0")
//     y     = clamp(x / scale, -fp8_max, fp8_max)
//     S     = scale (stored as E8M0)
//
// FP4 path: input BF16 [M, N] -> output FP4-E2M1 (packed) [M, N/2] + scales E8M0 [M, N/32].
//   Same recipe with fp4_max = 6.0 and block_size = 32.
//
// `inplace=true` variant in the reference does fused quant + dequant back into
// the original buffer — useful for QAT-style activation noise injection.
//
// NOT IMPLEMENTED. Implementing this requires:
//   - simdgroup absmax reduction
//   - float→fp8/fp4 round-to-nearest-even with the E4M3 / E2M1 saturation rules
//   - power-of-2 scale rounding via fast_log2_ceil + fast_pow2 bit hacks
//   - threadgroup-shared tile of `blk_m × group_size` to amortize scale computation
//
// Stub kernels intentionally write NaN to surface accidental invocations early.

kernel void act_quant_fp8_unimplemented(
    device float* dst [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) dst[0] = NAN;
}

kernel void act_quant_fp4_unimplemented(
    device float* dst [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) dst[0] = NAN;
}
