// SPDX-License-Identifier: MIT
#pragma once

#include <stdint.h>
#include <limits.h>

// The generic NVIDIA N=1 MMVQ owns one row, or four for small K. Admit
// complete four-row cohorts so every output is overwritten and the legacy
// small-K loader never reads a padded weight row. Keep its signed block
// indices representable as well. Device/backend and opt-out checks are host
// policy; the persistent K1024 experiment keeps its original epilogue.
static inline bool ds4_q4_mmvq_epilogue_shape_ok(int M, int N, int K) {
    return M > 0 && M % 4 == 0 && N == 1 && K > 0 && K % 256 == 0 &&
           (uint64_t)M * (uint64_t)(K / 256) <= INT_MAX;
}

#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
static inline uint32_t ds4_q4_mmvq_sanitize_bits(uint32_t bits) {
    // Preserve every finite bit pattern, including -0 and subnormals; map
    // all NaNs and both infinities to +0, like ds4_mmq_sanitize_f32_kernel.
    return (bits & 0x7f800000u) == 0x7f800000u ? 0u : bits;
}
