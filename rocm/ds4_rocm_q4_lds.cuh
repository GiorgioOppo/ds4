// SPDX-License-Identifier: MIT
// Word-exact Q8_K staging used only by the Q4 tiled-prefill kernels.
#ifndef DS4_ROCM_Q4_LDS_CUH
#define DS4_ROCM_Q4_LDS_CUH

#include <stdint.h>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define DS4_Q4_LDS_INLINE __host__ __device__ __forceinline__
#else
#define DS4_Q4_LDS_INLINE inline
#endif

namespace ds4_rocm_q4_lds {
enum { block_words = 73u }; // sizeof(Q8_K) / sizeof(uint32_t)

// src already points at this tile's first token/group/K block. A token may
// have other groups or K blocks between the ranges being staged. Destination
// rows always keep their full BLOCKS pitch, including an uninitialized K tail.
template<uint32_t BLOCKS>
DS4_Q4_LDS_INLINE void copy_thread(
        uint32_t *dst, const uint32_t *src, uint32_t tid, uint32_t threads,
        uint32_t nt, uint32_t nb, uint64_t src_stride_words) {
    const uint32_t pitch = BLOCKS * block_words;
    const uint32_t words = nt * pitch;
    if (nb == BLOCKS && src_stride_words == pitch) {
        // In particular, K1024 TILE4 is one contiguous copy. No per-word
        // block division, remainder, token index or 64-bit row multiply.
        for (uint32_t i = tid; i < words; i += threads) dst[i] = src[i];
    } else {
        for (uint32_t i = tid; i < words; i += threads) {
            const uint32_t p = i / pitch;
            const uint32_t word = i - p * pitch;
            if (word < nb * block_words)
                dst[i] = src[(uint64_t)p * src_stride_words + word];
        }
    }
}

// All workgroup threads use the same K loop. The final tile is read only:
// register reduction/output stores cannot overwrite LDS, so no reuse fence
// is necessary after it. The producer-to-consumer barrier remains mandatory.
DS4_Q4_LDS_INLINE bool needs_reuse_barrier(uint32_t b0, uint32_t blocks) {
    return blocks - b0 > 8u; // caller guarantees b0 < blocks
}
} // namespace ds4_rocm_q4_lds

#undef DS4_Q4_LDS_INLINE
#endif
