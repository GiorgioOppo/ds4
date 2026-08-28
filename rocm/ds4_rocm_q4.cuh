// DS4 ROCm dense Q4_K kernels and launch wrappers.
//
// This module is included after ds4_rocm_moe.cuh so it can reuse the
// canonical Q8_K activation quantizer, Q4_K/Q8_K block dot product, and
// quarter-wave reduction already used by the routed-MoE implementation.
// Launches intentionally stay on ROCm's default stream: cuda_tmp_alloc() is a
// single reusable scratch arena whose lifetime is protected by that ordering.

static_assert(sizeof(cuda_block_q4_K) == 144u,
              "ROCm Q4_K block layout must match GGUF");
static_assert(sizeof(cuda_block_q8_K) == 292u,
              "ROCm Q8_K activation block layout must match the dot kernel");

__global__ static void rocm_matmul_q4_K_dense_kernel(
        float *out,
        const char *w_base,
        const cuda_block_q8_K *xq,
        uint64_t row_bytes,
        uint32_t xq_blocks,
        uint32_t out_dim,
        uint32_t n_tok) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t tok = blockIdx.y;
    const uint32_t row = blockIdx.x * 32u + row_lane;
    if (tok >= n_tok || row >= out_dim) return;

    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    const cuda_block_q4_K *wr = reinterpret_cast<const cuda_block_q4_K *>(
            w_base + (uint64_t)row * row_bytes);
    float acc = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        acc += dev_dot_q4_K_q8_K_block(wr + b, xqb + b);
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = acc;
}

/* Latency-oriented pair variant of the canonical dense kernel.  Concatenating
 * the two row-tile domains keeps exactly the same per-row dot and reduction
 * order while sharing one launch between Q-A and KV. */
__global__ static void rocm_matmul_q4_K_dense_pair_kernel(
        float *out0,
        float *out1,
        const char *w0,
        const char *w1,
        const cuda_block_q8_K *xq,
        uint64_t row_bytes,
        uint32_t xq_blocks,
        uint32_t out0_dim,
        uint32_t out1_dim,
        uint32_t n_tok) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t out0_tiles = (out0_dim - 1u) / 32u + 1u;
    const bool second = blockIdx.x >= out0_tiles;
    const uint32_t row_tile = second ? blockIdx.x - out0_tiles : blockIdx.x;
    const uint32_t row = row_tile * 32u + row_lane;
    const uint32_t tok = blockIdx.y;
    const uint32_t out_dim = second ? out1_dim : out0_dim;
    if (tok >= n_tok || row >= out_dim) return;

    float *const out = second ? out1 : out0;
    const char *const w_base = second ? w1 : w0;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    const cuda_block_q4_K *wr = reinterpret_cast<const cuda_block_q4_K *>(
            w_base + (uint64_t)row * row_bytes);
    float acc = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        acc += dev_dot_q4_K_q8_K_block(wr + b, xqb + b);
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = acc;
}

/* One independent activation row and Q4_K matrix per output group.  This is
 * the canonical dense decode walk with only a group grid dimension added. */
__global__ static void rocm_matmul_q4_K_dense_grouped_decode_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t xq_blocks, uint32_t out_dim,
        uint32_t n_groups) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t row = blockIdx.x * 32u + row_lane;
    const uint32_t group = blockIdx.z;
    if (group >= n_groups || row >= out_dim) return;
    const cuda_block_q8_K *xqb = xq + (uint64_t)group * xq_blocks;
    const cuda_block_q4_K *wr = reinterpret_cast<const cuda_block_q4_K *>(
        w_base + ((uint64_t)group * out_dim + row) * row_bytes);
    float acc = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        acc += dev_dot_q4_K_q8_K_block(wr + b, xqb + b);
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0u) out[(uint64_t)group * out_dim + row] = acc;
}

/* Dense Q4_K prefill tile for RDNA/ROCm.
 *
 * The legacy kernel gives one eight-lane group a single (token,row) dot.  As
 * a result every token walks the complete Q4_K row independently and every
 * one of the 32 row groups in a workgroup also fetches the same Q8_K input
 * blocks.  Prefill is therefore dominated by redundant global reads.
 *
 * This kernel keeps the legacy eight-lane block assignment and reduction
 * order, but computes eight token columns at once.  A Q4_K block is decoded
 * once into eight integer dot products, and an 8x8 token/K-block tile of the
 * canonical Q8_K activations is staged in LDS for reuse by all 32 rows.  The
 * K loop advances in groups of eight so lane L still accumulates blocks
 * L,L+8,... in exactly the order used by rocm_matmul_q4_K_dense_kernel.
 *
 * LDS footprint: 8 tokens * 8 K blocks * 292 bytes = 18,688 bytes.
 * The final, partial token and K tiles are handled without an early return;
 * every thread must reach both barriers.
 */
enum {
    ROCM_Q4_PREFILL_TOKEN_TILE = 8u,
    ROCM_Q4_PREFILL_KBLOCK_TILE = 8u,
    ROCM_Q4_PREFILL_K1024_KBLOCK_TILE = 4u,
    ROCM_Q4_PREFILL_K1024_ROWS = 64u,
    ROCM_Q4_Q8K_WORDS = sizeof(cuda_block_q8_K) / sizeof(uint32_t),
};
static_assert((sizeof(cuda_block_q8_K) % sizeof(uint32_t)) == 0u,
              "ROCm Q8_K LDS copies require a whole number of words");

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
/* Experimental compressed Q4_K MMQ for resident gfx1151 prefill.
 *
 * This deliberately mirrors the live Q8 WMMA kernel: four wave32s compute a
 * 64-token x 64-row output tile, K advances by 32, only the 64x32 activation
 * tile is staged in 4 KiB of LDS, and accumulator fragments are written
 * straight to the output.  Each lane dequantizes one Q4_K row/group directly
 * into two F16 register vectors, so there is no Q8_K activation scratch and no
 * persistent F16 weight sidecar.
 *
 * Arithmetic is not bit-identical to Q4_K x Q8_K: activations and transient
 * weights round to F16 before F32 WMMA accumulation.  Host policy therefore
 * keeps this path opt-in, physically device-resident, gfx1151-only, and out of
 * quality mode until device-side A/B plus a prompt-level oracle promote it.
 */
enum {
    ROCM_Q4_WMMA_M_TILE = 64u,
    ROCM_Q4_WMMA_N_TILE = 64u,
    ROCM_Q4_WMMA_K_TILE = 32u,
    ROCM_Q4_WMMA_WAVES = 4u,
    ROCM_Q4_WMMA_FRAGMENT = 16u,
};

typedef _Float16 __attribute__((ext_vector_type(16))) ds4_q4_half16_t;
typedef float __attribute__((ext_vector_type(8))) ds4_q4_float8_t;
typedef uint8_t __attribute__((ext_vector_type(16))) ds4_q4_uchar16_t;

__launch_bounds__(128, 2)
__global__ static void rocm_matmul_q4_K_prefill_wmma64_strided_kernel(
        float *out,
        const char *w_base,
        const float *x,
        uint32_t n_tok,
        uint32_t n_groups,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes,
        uint64_t x_token_stride,
        uint64_t x_group_stride,
        uint64_t out_token_stride) {
    if (warpSize != 32) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t group = blockIdx.z;
    const uint32_t row0 = blockIdx.x * ROCM_Q4_WMMA_N_TILE;
    const uint32_t tok0 = blockIdx.y * ROCM_Q4_WMMA_M_TILE;
    if (group >= n_groups) return;

    const uint32_t wave_row0 = row0 + wave * ROCM_Q4_WMMA_FRAGMENT;
    const uint32_t my_row = wave_row0 + lane16;
    const uint32_t safe_row = my_row < out_dim ? my_row : out_dim - 1u;
    const cuda_block_q4_K *row_blocks =
        reinterpret_cast<const cuda_block_q4_K *>(
            w_base + ((uint64_t)group * out_dim + safe_row) * row_bytes);
    const uint32_t q4_blocks = in_dim / CUDA_QK_K;

    ds4_q4_float8_t acc0 = {0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f};
    ds4_q4_float8_t acc1 = acc0;
    ds4_q4_float8_t acc2 = acc0;
    ds4_q4_float8_t acc3 = acc0;
    __shared__ _Float16 lds_x[ROCM_Q4_WMMA_M_TILE * ROCM_Q4_WMMA_K_TILE];

    for (uint32_t block_index = 0u; block_index < q4_blocks;
         block_index++) {
        const cuda_block_q4_K *block = row_blocks + block_index;
        const float block_d = dev_f16_to_f32(block->d);
        const float block_dm = dev_f16_to_f32(block->dmin);
        for (uint32_t qpair = 0u; qpair < 4u; qpair++) {
            /* Adjacent 32-value groups are the low/high nibbles of the same
             * 32 payload bytes.  Keep them in registers across both K tiles
             * instead of issuing the same global loads twice. */
            ds4_q4_uchar16_t packed0;
            ds4_q4_uchar16_t packed1;
            __builtin_memcpy(
                &packed0, block->qs + qpair * 32u, sizeof(packed0));
            __builtin_memcpy(
                &packed1,
                block->qs + qpair * 32u + ROCM_Q4_WMMA_FRAGMENT,
                sizeof(packed1));

            #pragma unroll
            for (uint32_t nibble = 0u; nibble < 2u; nibble++) {
                const uint32_t qgroup = qpair * 2u + nibble;
                const uint32_t group32 = block_index * 8u + qgroup;
                for (uint32_t j = tid;
                     j < ROCM_Q4_WMMA_M_TILE * ROCM_Q4_WMMA_K_TILE;
                     j += blockDim.x) {
                    const uint32_t tok_local = j >> 5u;
                    const uint32_t kk = j & 31u;
                    const uint32_t tok = tok0 + tok_local;
                    float value = 0.0f;
                    if (tok < n_tok) {
                        value = x[(uint64_t)tok * x_token_stride +
                                  (uint64_t)group * x_group_stride +
                                  (uint64_t)group32 * ROCM_Q4_WMMA_K_TILE +
                                  kk];
                    }
                    lds_x[j] = (_Float16)value;
                }
                __syncthreads();

                uint8_t scale = 0u;
                uint8_t minimum = 0u;
                dev_q4_K_get_scale_min(
                    qgroup, block->scales, &scale, &minimum);
                const float d = block_d * (float)scale;
                const float dm = block_dm * (float)minimum;
                const uint32_t shift = nibble * 4u;
                ds4_q4_half16_t weights0;
                ds4_q4_half16_t weights1;
                #pragma unroll
                for (uint32_t i = 0u; i < ROCM_Q4_WMMA_FRAGMENT; i++) {
                    const uint8_t q0 = (packed0[i] >> shift) & 0x0fu;
                    const uint8_t q1 = (packed1[i] >> shift) & 0x0fu;
                    weights0[i] = (_Float16)(d * (float)q0 - dm);
                    weights1[i] = (_Float16)(d * (float)q1 - dm);
                }

                #pragma unroll
                for (uint32_t token_tile = 0u; token_tile < 4u;
                     token_tile++) {
                    const uint32_t token_local =
                        token_tile * ROCM_Q4_WMMA_FRAGMENT + lane16;
                    const _Float16 *activation =
                        lds_x + token_local * ROCM_Q4_WMMA_K_TILE;
                    const ds4_q4_half16_t activation0 =
                        *reinterpret_cast<const ds4_q4_half16_t *>(activation);
                    const ds4_q4_half16_t activation1 =
                        *reinterpret_cast<const ds4_q4_half16_t *>(
                            activation + ROCM_Q4_WMMA_FRAGMENT);
                    if (token_tile == 0u) {
                        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights0, activation0, acc0);
                        acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights1, activation1, acc0);
                    } else if (token_tile == 1u) {
                        acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights0, activation0, acc1);
                        acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights1, activation1, acc1);
                    } else if (token_tile == 2u) {
                        acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights0, activation0, acc2);
                        acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights1, activation1, acc2);
                    } else {
                        acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights0, activation0, acc3);
                        acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            weights1, activation1, acc3);
                    }
                }
                __syncthreads();
            }
        }
    }

    #pragma unroll
    for (uint32_t token_tile = 0u; token_tile < 4u; token_tile++) {
        const uint32_t tok =
            tok0 + token_tile * ROCM_Q4_WMMA_FRAGMENT + lane16;
        if (tok >= n_tok) continue;
        const ds4_q4_float8_t acc = token_tile == 0u
            ? acc0
            : (token_tile == 1u ? acc1
                                : (token_tile == 2u ? acc2 : acc3));
        #pragma unroll
        for (uint32_t j = 0u; j < 8u; j++) {
            const uint32_t row = wave_row0 + 2u * j + (lane >> 4u);
            if (row < out_dim) {
                out[(uint64_t)tok * out_token_stride +
                    (uint64_t)group * out_dim + row] = acc[j];
            }
        }
    }
}
#endif

__device__ __forceinline__ static void
rocm_dot_q4_K_q8_K_block8_reuse_weights(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[ROCM_Q4_PREFILL_TOKEN_TILE]) {
    const cuda_block_q8_K *ys[ROCM_Q4_PREFILL_TOKEN_TILE] = {
        y0, y1, y2, y3, y4, y5, y6, y7,
    };
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int32_t isum[ROCM_Q4_PREFILL_TOKEN_TILE] = {0, 0, 0, 0, 0, 0, 0, 0};
    int32_t summs[ROCM_Q4_PREFILL_TOKEN_TILE] = {0, 0, 0, 0, 0, 0, 0, 0};

    /* A 32-byte Q4 payload stores the low and high nibbles for two adjacent
     * 32-value groups.  Load those eight packed words once, then reuse them
     * for both groups and every token in the tile.  This makes weight reuse
     * explicit instead of relying on the compiler or vector cache to hoist
     * repeated loads out of the token loop. */
    #pragma unroll
    for (uint32_t jp = 0u; jp < 4u; jp++) {
        const uint32_t j0 = 2u * jp;
        const uint32_t j1 = j0 + 1u;
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j0, x->scales, &sc0, &m0);
        dev_q4_K_get_scale_min(j1, x->scales, &sc1, &m1);

        int32_t qw[8];
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; i++) {
            qw[i] = *reinterpret_cast<const int32_t *>(
                x->qs + jp * 32u + i * 4u);
        }

        #pragma unroll
        for (uint32_t p = 0u; p < ROCM_Q4_PREFILL_TOKEN_TILE; p++) {
            if (p < n) {
                const cuda_block_q8_K *y = ys[p];
                int32_t dot0 = 0;
                int32_t dot1 = 0;
                #pragma unroll
                for (uint32_t i = 0u; i < 8u; i++) {
                    const int32_t w0 = qw[i] & 0x0f0f0f0f;
                    const int32_t w1 = (qw[i] >> 4) & 0x0f0f0f0f;
                    dot0 = __dp4a(w0, *reinterpret_cast<const int32_t *>(
                                           y->qs + j0 * 32u + i * 4u), dot0);
                    dot1 = __dp4a(w1, *reinterpret_cast<const int32_t *>(
                                           y->qs + j1 * 32u + i * 4u), dot1);
                }
                isum[p] += (int32_t)sc0 * dot0;
                isum[p] += (int32_t)sc1 * dot1;
                summs[p] += (int32_t)m0 *
                    (int32_t)(y->bsums[2u * j0] + y->bsums[2u * j0 + 1u]);
                summs[p] += (int32_t)m1 *
                    (int32_t)(y->bsums[2u * j1] + y->bsums[2u * j1 + 1u]);
            }
        }
    }

    #pragma unroll
    for (uint32_t p = 0u; p < ROCM_Q4_PREFILL_TOKEN_TILE; p++) {
        if (p < n) {
            const float yd = ys[p]->d;
            acc[p] += yd * xd * (float)isum[p] -
                      yd * xmin * (float)summs[p];
        }
    }
}

/* K=1024 has exactly four Q8_K blocks.  The generic TILE8 kernel leaves half
 * of each eight-lane row group idle and still reserves LDS for eight blocks.
 * Four-lane groups preserve the legacy block/reduction order while doubling
 * the rows produced by a 256-thread workgroup and halving the LDS footprint
 * to 8 tokens * 4 K blocks * 292 bytes = 9,344 bytes. */
__device__ __forceinline__ static float
rocm_q4_K_lane4_sum_f32(float v) {
    /* Build the active-lane mask relative to the physical wave.  A 32-bit
     * mask repeats lanes 0..31 for the upper half of an AMD wave64 and
     * violates HIP's __shfl_down_sync contract even though width=4 keeps the
     * data exchange inside the intended subgroup. */
    const uint32_t wave_lane = threadIdx.x & (warpSize - 1u);
    const MASK_T mask = static_cast<MASK_T>(0x0fu) << (wave_lane & ~3u);
    v += __shfl_down_sync(mask, v, 2, 4);
    v += __shfl_down_sync(mask, v, 1, 4);
    return v;
}

__global__ static void rocm_matmul_q4_K_prefill_k1024_tile4_kernel(
        float *out,
        const char *w_base,
        const cuda_block_q8_K *xq,
        uint64_t row_bytes,
        uint32_t out_dim,
        uint32_t n_tok) {
    __shared__ cuda_block_q8_K sxq[ROCM_Q4_PREFILL_TOKEN_TILE]
                                         [ROCM_Q4_PREFILL_K1024_KBLOCK_TILE];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 3u;
    const uint32_t row_lane = tid >> 2u;
    const uint32_t row = blockIdx.x * ROCM_Q4_PREFILL_K1024_ROWS + row_lane;
    const uint32_t tok0 = blockIdx.y * ROCM_Q4_PREFILL_TOKEN_TILE;
    const uint32_t nt = n_tok - tok0 < ROCM_Q4_PREFILL_TOKEN_TILE
                      ? n_tok - tok0 : ROCM_Q4_PREFILL_TOKEN_TILE;
    const bool row_valid = row < out_dim;
    const cuda_block_q4_K *wr = row_valid
        ? reinterpret_cast<const cuda_block_q4_K *>(
              w_base + (uint64_t)row * row_bytes)
        : NULL;
    float acc[ROCM_Q4_PREFILL_TOKEN_TILE] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };

    /* The complete K dimension fits one LDS tile.  Flatten the copy so
     * neighboring threads read consecutive words across token/block rows. */
    const uint32_t tile_words = nt * ROCM_Q4_PREFILL_K1024_KBLOCK_TILE *
                                ROCM_Q4_Q8K_WORDS;
    uint32_t *const sxq_words = reinterpret_cast<uint32_t *>(sxq);
    for (uint32_t i = tid; i < tile_words; i += blockDim.x) {
        const uint32_t block_slot = i / ROCM_Q4_Q8K_WORDS;
        const uint32_t word = i - block_slot * ROCM_Q4_Q8K_WORDS;
        const uint32_t p = block_slot >> 2u;
        const uint32_t bb = block_slot & 3u;
        const uint32_t *const src_words =
            reinterpret_cast<const uint32_t *>(
                xq + (uint64_t)(tok0 + p) *
                          ROCM_Q4_PREFILL_K1024_KBLOCK_TILE + bb);
        sxq_words[i] = src_words[word];
    }
    __syncthreads();

    if (row_valid) {
        rocm_dot_q4_K_q8_K_block8_reuse_weights(
            wr + lane,
            sxq[0] + lane, sxq[1] + lane,
            sxq[2] + lane, sxq[3] + lane,
            sxq[4] + lane, sxq[5] + lane,
            sxq[6] + lane, sxq[7] + lane,
            nt, acc);

        #pragma unroll
        for (uint32_t p = 0u; p < ROCM_Q4_PREFILL_TOKEN_TILE; p++) {
            if (p < nt) {
                const float v = rocm_q4_K_lane4_sum_f32(acc[p]);
                if (lane == 0u) {
                    out[(uint64_t)(tok0 + p) * out_dim + row] = v;
                }
            }
        }
    }
}

__global__ static void rocm_matmul_q4_K_prefill_tile8_strided_kernel(
        float *out,
        const char *w_base,
        const cuda_block_q8_K *xq,
        uint64_t row_bytes,
        uint32_t xq_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t xq_token_stride,
        uint64_t out_token_stride) {
    __shared__ cuda_block_q8_K sxq[ROCM_Q4_PREFILL_TOKEN_TILE]
                                         [ROCM_Q4_PREFILL_KBLOCK_TILE];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 7u;
    const uint32_t row_lane = tid >> 3u;
    const uint32_t row = blockIdx.x * 32u + row_lane;
    const uint32_t tok0 = blockIdx.y * ROCM_Q4_PREFILL_TOKEN_TILE;
    const uint32_t group = blockIdx.z;
    const uint32_t nt = n_tok - tok0 < ROCM_Q4_PREFILL_TOKEN_TILE
                      ? n_tok - tok0 : ROCM_Q4_PREFILL_TOKEN_TILE;
    const bool row_valid = row < out_dim;
    const cuda_block_q4_K *wr = row_valid
        ? reinterpret_cast<const cuda_block_q4_K *>(
              w_base + ((uint64_t)group * out_dim + row) * row_bytes)
        : NULL;
    float acc[ROCM_Q4_PREFILL_TOKEN_TILE] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };

    for (uint32_t b0 = 0u; b0 < xq_blocks;
         b0 += ROCM_Q4_PREFILL_KBLOCK_TILE) {
        const uint32_t nb = xq_blocks - b0 < ROCM_Q4_PREFILL_KBLOCK_TILE
                          ? xq_blocks - b0
                          : ROCM_Q4_PREFILL_KBLOCK_TILE;

        /* Copy consecutive 32-bit words cooperatively.  Assigning one 292-B
         * struct per lane makes adjacent lanes issue 292-B-strided global
         * loads; flattening the packed blocks gives the memory coalescer long
         * contiguous runs while preserving the fixed eight-block LDS layout. */
        const uint32_t tile_words = nt * ROCM_Q4_PREFILL_KBLOCK_TILE *
                                    ROCM_Q4_Q8K_WORDS;
        uint32_t *const sxq_words = reinterpret_cast<uint32_t *>(sxq);
        for (uint32_t i = tid; i < tile_words; i += blockDim.x) {
            const uint32_t block_slot = i / ROCM_Q4_Q8K_WORDS;
            const uint32_t word = i - block_slot * ROCM_Q4_Q8K_WORDS;
            const uint32_t p = block_slot >> 3u;
            const uint32_t bb = block_slot & 7u;
            if (bb < nb) {
                const uint64_t src_block =
                    (uint64_t)(tok0 + p) * xq_token_stride +
                    (uint64_t)group * xq_blocks + b0 + bb;
                const uint32_t *const src_words =
                    reinterpret_cast<const uint32_t *>(xq + src_block);
                sxq_words[i] = src_words[word];
            }
        }
        __syncthreads();

        if (row_valid && lane < nb) {
            rocm_dot_q4_K_q8_K_block8_reuse_weights(
                wr + b0 + lane,
                sxq[0] + lane, sxq[1] + lane,
                sxq[2] + lane, sxq[3] + lane,
                sxq[4] + lane, sxq[5] + lane,
                sxq[6] + lane, sxq[7] + lane,
                nt, acc);
        }
        __syncthreads();
    }

    if (row_valid) {
        #pragma unroll
        for (uint32_t p = 0u; p < ROCM_Q4_PREFILL_TOKEN_TILE; p++) {
            if (p < nt) {
                const float v = quarter_warp_sum_f32(acc[p], lane);
                if (lane == 0u) {
                    out[(uint64_t)(tok0 + p) * out_token_stride +
                        (uint64_t)group * out_dim + row] = v;
                }
            }
        }
    }
}

/* Two independent dense projections over the same activation tile.  The
 * row-tile ranges are concatenated in grid.x, so Q/KV prefill shares both
 * the Q8_K quantization and a single launch without padding the smaller
 * projection up to the larger one's row count.  Each workgroup still handles
 * only one weight matrix: this preserves the standalone TILE8 block walk and
 * its accumulation order while removing the second host launch. */
__global__ static void rocm_matmul_q4_K_prefill_tile8_pair_kernel(
        float *out0,
        float *out1,
        const char *w0,
        const char *w1,
        const cuda_block_q8_K *xq,
        uint64_t row_bytes,
        uint32_t xq_blocks,
        uint32_t out0_dim,
        uint32_t out1_dim,
        uint32_t n_tok) {
    __shared__ cuda_block_q8_K sxq[ROCM_Q4_PREFILL_TOKEN_TILE]
                                         [ROCM_Q4_PREFILL_KBLOCK_TILE];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 7u;
    const uint32_t row_lane = tid >> 3u;
    const uint32_t out0_tiles = (out0_dim - 1u) / 32u + 1u;
    const bool second = blockIdx.x >= out0_tiles;
    const uint32_t row_tile = second ? blockIdx.x - out0_tiles : blockIdx.x;
    const uint32_t row = row_tile * 32u + row_lane;
    const uint32_t tok0 = blockIdx.y * ROCM_Q4_PREFILL_TOKEN_TILE;
    const uint32_t out_dim = second ? out1_dim : out0_dim;
    float *const out = second ? out1 : out0;
    const char *const w_base = second ? w1 : w0;
    const uint32_t nt = n_tok - tok0 < ROCM_Q4_PREFILL_TOKEN_TILE
                      ? n_tok - tok0 : ROCM_Q4_PREFILL_TOKEN_TILE;
    const bool row_valid = row < out_dim;
    const cuda_block_q4_K *wr = row_valid
        ? reinterpret_cast<const cuda_block_q4_K *>(
              w_base + (uint64_t)row * row_bytes)
        : NULL;
    float acc[ROCM_Q4_PREFILL_TOKEN_TILE] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    };

    for (uint32_t b0 = 0u; b0 < xq_blocks;
         b0 += ROCM_Q4_PREFILL_KBLOCK_TILE) {
        const uint32_t nb = xq_blocks - b0 < ROCM_Q4_PREFILL_KBLOCK_TILE
                          ? xq_blocks - b0
                          : ROCM_Q4_PREFILL_KBLOCK_TILE;
        const uint32_t tile_words = nt * ROCM_Q4_PREFILL_KBLOCK_TILE *
                                    ROCM_Q4_Q8K_WORDS;
        uint32_t *const sxq_words = reinterpret_cast<uint32_t *>(sxq);
        for (uint32_t i = tid; i < tile_words; i += blockDim.x) {
            const uint32_t block_slot = i / ROCM_Q4_Q8K_WORDS;
            const uint32_t word = i - block_slot * ROCM_Q4_Q8K_WORDS;
            const uint32_t p = block_slot >> 3u;
            const uint32_t bb = block_slot & 7u;
            if (bb < nb) {
                const uint64_t src_block =
                    (uint64_t)(tok0 + p) * xq_blocks + b0 + bb;
                const uint32_t *const src_words =
                    reinterpret_cast<const uint32_t *>(xq + src_block);
                sxq_words[i] = src_words[word];
            }
        }
        __syncthreads();

        if (row_valid && lane < nb) {
            rocm_dot_q4_K_q8_K_block8_reuse_weights(
                wr + b0 + lane,
                sxq[0] + lane, sxq[1] + lane,
                sxq[2] + lane, sxq[3] + lane,
                sxq[4] + lane, sxq[5] + lane,
                sxq[6] + lane, sxq[7] + lane,
                nt, acc);
        }
        __syncthreads();
    }

    if (row_valid) {
        #pragma unroll
        for (uint32_t p = 0u; p < ROCM_Q4_PREFILL_TOKEN_TILE; p++) {
            if (p < nt) {
                const float v = quarter_warp_sum_f32(acc[p], lane);
                if (lane == 0u) {
                    out[(uint64_t)(tok0 + p) * out_dim + row] = v;
                }
            }
        }
    }
}

static int rocm_q4_K_dense_validate(
        const ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        uint64_t *blocks_out,
        uint64_t *row_bytes_out,
        uint64_t *weight_bytes_out) {
    if (!out || !x || !model_map || !blocks_out || !row_bytes_out ||
        !weight_bytes_out || in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX ||
        (in_dim % CUDA_QK_K) != 0u) {
        return 0;
    }

    const uint64_t blocks = in_dim / CUDA_QK_K;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    if (blocks == 0u ||
        !cuda_u64_mul_checked(blocks, sizeof(cuda_block_q4_K), &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_elems2(x, n_tok, in_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(out, n_tok, out_dim, sizeof(float))) {
        return 0;
    }

    *blocks_out = blocks;
    *row_bytes_out = row_bytes;
    *weight_bytes_out = weight_bytes;
    return 1;
}

static cuda_block_q8_K *rocm_q4_K_prequant_alloc(
        uint64_t n_tok,
        uint64_t blocks,
        const char *what) {
    uint64_t bytes = 0;
    if (!cuda_u64_mul3_checked(n_tok, blocks,
                               sizeof(cuda_block_q8_K), &bytes)) {
        return NULL;
    }
    return reinterpret_cast<cuda_block_q8_K *>(cuda_tmp_alloc(bytes, what));
}

static int rocm_q4_K_byte_ranges_overlap(
        const void *ptr0, uint64_t bytes0,
        const void *ptr1, uint64_t bytes1) {
    const uintptr_t p0 = reinterpret_cast<uintptr_t>(ptr0);
    const uintptr_t p1 = reinterpret_cast<uintptr_t>(ptr1);
    return p0 <= p1 ? (uint64_t)(p1 - p0) < bytes0
                    : (uint64_t)(p0 - p1) < bytes1;
}

static int rocm_q4_K_dense_pair_requested(void) {
    return getenv("DS4_ROCM_ENABLE_Q4_DENSE_PAIR") != NULL &&
           getenv("DS4_ROCM_DISABLE_Q4_DENSE_PAIR") == NULL;
}

enum {
    ROCM_Q4_GROUPED_ATTN_A_DEFAULT_K = 4096u,
    ROCM_Q4_GROUPED_ATTN_A_DEFAULT_M = 1024u,
    ROCM_Q4_GROUPED_ATTN_A_DEFAULT_GROUPS = 8u,
};

static int rocm_q4_K_grouped_attn_a_resident_default_scope(
        uint64_t group_dim,
        uint64_t rank,
        uint32_t group0,
        uint32_t group_cnt,
        int resident_decode) {
    /* A batch fallback can pass one row at a time through this same API, so
     * the caller explicitly identifies true decode.  Default only the
     * production decode shape after the complete model has been made
     * resident; explicit ENABLE keeps the existing experimental surface. */
    return resident_decode &&
           !g_ssd_streaming_mode &&
           group_dim == ROCM_Q4_GROUPED_ATTN_A_DEFAULT_K &&
           rank == ROCM_Q4_GROUPED_ATTN_A_DEFAULT_M &&
           group0 == 0u &&
           group_cnt == ROCM_Q4_GROUPED_ATTN_A_DEFAULT_GROUPS;
}

static int rocm_q4_K_prefill_tile8_scope(uint64_t n_tok) {
    /* Keep decode/speculative micro-batches on the latency-oriented legacy
     * kernel.  4096 is DS4's largest supported prefill chunk and bounds the
     * validated tiled-prefill surface. */
    return n_tok > 8u && n_tok <= 4096u;
}

static int rocm_q4_K_prefill_tile8_requested(void) {
    /* TILE8 is the ROCm Q4 prefill default.  Keep the old ENABLE variable
     * harmlessly compatible and retain one authoritative rollback switch. */
    return getenv("DS4_ROCM_DISABLE_Q4_PREFILL_TILE8") == NULL;
}

static int rocm_q4_K_prefill_tile8_required(void) {
    return getenv("DS4_ROCM_REQUIRE_Q4_PREFILL_TILE8") != NULL;
}

enum {
    ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE = -1,
    ROCM_Q4_PREFILL_WMMA_FALLBACK = 0,
    ROCM_Q4_PREFILL_WMMA_USE = 1,
};

/* Test oracle for strict dispatch: REQUIRE must attest this launch wrapper,
 * not merely produce output through a canonical fallback. */
static uint64_t g_rocm_q4_prefill_wmma_launches;

extern "C" void ds4_rocm_test_q4_prefill_wmma_reset(void) {
    __atomic_store_n(&g_rocm_q4_prefill_wmma_launches, 0u,
                     __ATOMIC_RELAXED);
}

extern "C" uint64_t ds4_rocm_test_q4_prefill_wmma_get_calls(void) {
    return __atomic_load_n(&g_rocm_q4_prefill_wmma_launches,
                           __ATOMIC_RELAXED);
}

static int rocm_q4_K_prefill_wmma_select(
        uint64_t n_tok,
        uint64_t in_dim,
        uint64_t out_dim,
        int weight_device_resident) {
    const int enabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA") == 1;
    const int ssd_enabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA_SSD") == 1;
    const int disabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA") == 1;
    const int required = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA") == 1;
    const int requested = enabled || required ||
                          (g_ssd_streaming_mode && ssd_enabled);
    if (!requested) return ROCM_Q4_PREFILL_WMMA_FALLBACK;

    const int shape_ok =
        n_tok >= 256u && n_tok <= 4096u &&
        in_dim != 0u && (in_dim % CUDA_QK_K) == 0u &&
        out_dim != 0u && in_dim <= UINT32_MAX &&
        out_dim <= UINT32_MAX && n_tok <= UINT32_MAX;
    const int storage_ok = !g_ssd_streaming_mode ||
                           ((ssd_enabled || required) &&
                            weight_device_resident);
    const int eligible = !disabled && shape_ok && storage_ok &&
        !g_quality_mode && rocm_attention_runtime_is_gfx1151_wave32();
    if (eligible) return ROCM_Q4_PREFILL_WMMA_USE;
    if (!required) return ROCM_Q4_PREFILL_WMMA_FALLBACK;

    fprintf(stderr,
            DS4_GPU_LOG_PREFIX
            "required Q4_K prefill WMMA is unavailable "
            "(N=%llu K=%llu M=%llu disabled=%d ssd=%d ssd_opt=%d "
            "resident=%d quality=%d)\n",
            (unsigned long long)n_tok,
            (unsigned long long)in_dim,
            (unsigned long long)out_dim,
            disabled,
            g_ssd_streaming_mode ? 1 : 0,
            ssd_enabled,
            weight_device_resident,
            g_quality_mode ? 1 : 0);
    return ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE;
}

static int rocm_q4_K_prefill_wmma_launch(
        float *out,
        const char *w,
        const float *x,
        uint32_t n_tok,
        uint32_t n_groups,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes,
        uint64_t x_token_stride,
        uint64_t x_group_stride,
        uint64_t out_token_stride,
        const char *label) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (!out || !w || !x || n_groups == 0u) return 0;
    const dim3 grid(
        (unsigned)(((uint64_t)out_dim + ROCM_Q4_WMMA_N_TILE - 1u) /
                   ROCM_Q4_WMMA_N_TILE),
        (unsigned)(((uint64_t)n_tok + ROCM_Q4_WMMA_M_TILE - 1u) /
                   ROCM_Q4_WMMA_M_TILE),
        n_groups);
    rocm_matmul_q4_K_prefill_wmma64_strided_kernel<<<grid, 128u>>>(
        out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
        x_token_stride, x_group_stride, out_token_stride);
    const int ok = cuda_ok(
        cudaGetLastError(), label ? label : "q4_K prefill WMMA64 launch");
    if (ok) {
        __atomic_fetch_add(&g_rocm_q4_prefill_wmma_launches, 1u,
                           __ATOMIC_RELAXED);
    }
    return ok;
#else
    (void)out;
    (void)w;
    (void)x;
    (void)n_tok;
    (void)n_groups;
    (void)in_dim;
    (void)out_dim;
    (void)row_bytes;
    (void)x_token_stride;
    (void)x_group_stride;
    (void)out_token_stride;
    (void)label;
    return 0;
#endif
}

enum {
    ROCM_Q4_PREFILL_K1024_TILE4_REQUIRED_FAILURE = -1,
    ROCM_Q4_PREFILL_K1024_TILE4_FALLBACK = 0,
    ROCM_Q4_PREFILL_K1024_TILE4_USE = 1,
};

/* Keep this decision independent from the device lookup so the complete
 * policy matrix has a hardware-free oracle.  Resident execution preserves
 * the established automatic default.  SSD execution stays opt-in and may
 * only select TILE4 after the exact weight range has been found in actual
 * device storage; mapped/registered host memory is deliberately insufficient.
 * REQUIRE requests the candidate as well as asserting it, while DISABLE is
 * authoritative in both modes. */
static int rocm_q4_K_prefill_k1024_tile4_policy(
        int ssd_streaming,
        int weight_device_resident,
        int ssd_enabled,
        int disabled,
        int required) {
    if (disabled) {
        return required ? ROCM_Q4_PREFILL_K1024_TILE4_REQUIRED_FAILURE
                        : ROCM_Q4_PREFILL_K1024_TILE4_FALLBACK;
    }
    if (!ssd_streaming) return ROCM_Q4_PREFILL_K1024_TILE4_USE;
    if (!ssd_enabled && !required) {
        return ROCM_Q4_PREFILL_K1024_TILE4_FALLBACK;
    }
    if (!weight_device_resident) {
        return required ? ROCM_Q4_PREFILL_K1024_TILE4_REQUIRED_FAILURE
                        : ROCM_Q4_PREFILL_K1024_TILE4_FALLBACK;
    }
    return ROCM_Q4_PREFILL_K1024_TILE4_USE;
}

/* Test-only pure-policy entry point.  It intentionally performs no HIP call,
 * so hosts with a ROCm toolchain but no visible device can still validate the
 * SSD default, residency gate, and DISABLE/REQUIRE precedence. */
extern "C" int ds4_rocm_test_q4_prefill_k1024_tile4_policy(
        int ssd_streaming,
        int weight_device_resident,
        int ssd_enabled,
        int disabled,
        int required) {
    return rocm_q4_K_prefill_k1024_tile4_policy(
        ssd_streaming != 0, weight_device_resident != 0,
        ssd_enabled != 0, disabled != 0, required != 0);
}

static int rocm_q4_K_prefill_k1024_tile4_resolve(
        uint64_t blocks,
        uint64_t out_dim,
        const void *model_map,
        uint64_t weight_offset,
        uint64_t weight_bytes,
        const char *weight_ptr) {
    if (blocks != ROCM_Q4_PREFILL_K1024_KBLOCK_TILE ||
        out_dim != DS4_ROCM_Q4_ATTN_Q_B_OUT_DIM) {
        return ROCM_Q4_PREFILL_K1024_TILE4_FALLBACK;
    }

    const int enabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_ENABLE_Q4_PREFILL_K1024_TILE4_SSD") == 1;
    const int disabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_DISABLE_Q4_PREFILL_K1024_TILE4") == 1;
    const int required = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_REQUIRE_Q4_PREFILL_K1024_TILE4") == 1;
    const char *resident_ptr = g_ssd_streaming_mode
        ? rocm_q4_attn_q_b_device_resident_source(
              model_map, weight_offset, weight_bytes)
        : weight_ptr;
    const int weight_device_resident =
        resident_ptr != NULL && resident_ptr == weight_ptr;
    const int decision = rocm_q4_K_prefill_k1024_tile4_policy(
        g_ssd_streaming_mode, weight_device_resident, enabled, disabled,
        required);
    if (decision == ROCM_Q4_PREFILL_K1024_TILE4_REQUIRED_FAILURE) {
        if (disabled) {
            fprintf(stderr,
                    "ds4: required ROCm Q4_K prefill K1024 tile4 is "
                    "disabled\n");
        } else {
            fprintf(stderr,
                    "ds4: required ROCm Q4_K prefill K1024 tile4 has no "
                    "device-resident SSD weight range "
                    "(offset=%llu bytes=%llu)\n",
                    (unsigned long long)weight_offset,
                    (unsigned long long)weight_bytes);
        }
    }
    return decision;
}

static uint64_t g_rocm_q4_prefill_tile8_dense_calls;
static uint64_t g_rocm_q4_prefill_tile8_pair_calls;
static uint64_t g_rocm_q4_prefill_tile8_attention_batch_calls;
static uint64_t g_rocm_q4_prefill_k1024_tile4_calls;
static uint64_t g_rocm_q4_prefill_k1024_tile4_ssd_calls;
static uint64_t g_rocm_q4_prefill_tile8_tokens;
static int g_rocm_q4_prefill_tile8_report_registered;
static pthread_mutex_t g_rocm_q4_prefill_tile8_stats_mutex =
    PTHREAD_MUTEX_INITIALIZER;

static uint64_t g_rocm_q4_grouped_attn_a_calls;
static uint64_t g_rocm_q4_grouped_attn_a_dispatches;
static uint64_t g_rocm_q4_grouped_attn_a_groups;
static uint64_t g_rocm_q4_grouped_attn_a_fallbacks;
static uint64_t g_rocm_q4_grouped_attn_a_failures;
static int g_rocm_q4_grouped_attn_a_report_registered;
static pthread_mutex_t g_rocm_q4_grouped_attn_a_stats_mutex =
    PTHREAD_MUTEX_INITIALIZER;

static void rocm_q4_K_grouped_attn_a_report(void) {
    pthread_mutex_lock(&g_rocm_q4_grouped_attn_a_stats_mutex);
    const uint64_t calls = g_rocm_q4_grouped_attn_a_calls;
    const uint64_t dispatches = g_rocm_q4_grouped_attn_a_dispatches;
    const uint64_t groups = g_rocm_q4_grouped_attn_a_groups;
    const uint64_t fallbacks = g_rocm_q4_grouped_attn_a_fallbacks;
    const uint64_t failures = g_rocm_q4_grouped_attn_a_failures;
    pthread_mutex_unlock(&g_rocm_q4_grouped_attn_a_stats_mutex);
    fprintf(stderr,
            "ds4: ROCm Q4_K grouped attention-A decode stats: "
            "calls=%llu dispatches=%llu groups=%llu fallbacks=%llu failures=%llu\n",
            (unsigned long long)calls,
            (unsigned long long)dispatches,
            (unsigned long long)groups,
            (unsigned long long)fallbacks,
            (unsigned long long)failures);
}

static int rocm_q4_K_grouped_attn_a_result(int rc, uint32_t n_groups) {
    if (getenv("DS4_ROCM_Q4_GROUPED_ATTN_A_STATS") != NULL) {
        pthread_mutex_lock(&g_rocm_q4_grouped_attn_a_stats_mutex);
        if (!g_rocm_q4_grouped_attn_a_report_registered) {
            g_rocm_q4_grouped_attn_a_report_registered = 1;
            (void)atexit(rocm_q4_K_grouped_attn_a_report);
        }
        g_rocm_q4_grouped_attn_a_calls++;
        if (rc > 0) {
            g_rocm_q4_grouped_attn_a_dispatches++;
            g_rocm_q4_grouped_attn_a_groups += n_groups;
        } else if (rc < 0) {
            g_rocm_q4_grouped_attn_a_failures++;
        } else {
            g_rocm_q4_grouped_attn_a_fallbacks++;
        }
        pthread_mutex_unlock(&g_rocm_q4_grouped_attn_a_stats_mutex);
    }
    return rc;
}

static void rocm_q4_K_prefill_tile8_report(void) {
    pthread_mutex_lock(&g_rocm_q4_prefill_tile8_stats_mutex);
    const uint64_t dense_calls = g_rocm_q4_prefill_tile8_dense_calls;
    const uint64_t pair_calls = g_rocm_q4_prefill_tile8_pair_calls;
    const uint64_t attention_batch_calls =
        g_rocm_q4_prefill_tile8_attention_batch_calls;
    const uint64_t k1024_tile4_calls =
        g_rocm_q4_prefill_k1024_tile4_calls;
    const uint64_t k1024_tile4_ssd_calls =
        g_rocm_q4_prefill_k1024_tile4_ssd_calls;
    const uint64_t tokens = g_rocm_q4_prefill_tile8_tokens;
    pthread_mutex_unlock(&g_rocm_q4_prefill_tile8_stats_mutex);
    fprintf(stderr,
            "ds4: ROCm Q4_K tiled-prefill stats: "
            "dense_calls=%llu pair_calls=%llu attention_batch_calls=%llu "
            "k1024_tile4_calls=%llu k1024_tile4_ssd_calls=%llu "
            "tokens=%llu\n",
            (unsigned long long)dense_calls,
            (unsigned long long)pair_calls,
            (unsigned long long)attention_batch_calls,
            (unsigned long long)k1024_tile4_calls,
            (unsigned long long)k1024_tile4_ssd_calls,
            (unsigned long long)tokens);
}

static void rocm_q4_K_prefill_tile8_note(
        uint32_t dense_calls,
        uint32_t pair_calls,
        uint32_t attention_batch_calls,
        uint32_t k1024_tile4_calls,
        uint64_t tokens) {
    if (getenv("DS4_ROCM_Q4_PREFILL_TILE8_STATS") == NULL) return;
    pthread_mutex_lock(&g_rocm_q4_prefill_tile8_stats_mutex);
    if (!g_rocm_q4_prefill_tile8_report_registered) {
        g_rocm_q4_prefill_tile8_report_registered = 1;
        (void)atexit(rocm_q4_K_prefill_tile8_report);
    }
    g_rocm_q4_prefill_tile8_dense_calls += dense_calls;
    g_rocm_q4_prefill_tile8_pair_calls += pair_calls;
    g_rocm_q4_prefill_tile8_attention_batch_calls += attention_batch_calls;
    g_rocm_q4_prefill_k1024_tile4_calls += k1024_tile4_calls;
    if (k1024_tile4_calls && g_ssd_streaming_mode) {
        g_rocm_q4_prefill_k1024_tile4_ssd_calls += k1024_tile4_calls;
    }
    g_rocm_q4_prefill_tile8_tokens += tokens;
    pthread_mutex_unlock(&g_rocm_q4_prefill_tile8_stats_mutex);
}

extern "C" int ds4_rocm_matmul_q4_K_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    uint64_t blocks = 0;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    if (!rocm_q4_K_dense_validate(out, model_map, model_size, weight_offset,
                                  in_dim, out_dim, x, n_tok, &blocks,
                                  &row_bytes, &weight_bytes)) {
        return 0;
    }

    const int prefill_scope = rocm_q4_K_prefill_tile8_scope(n_tok);
    const int prefill_tile8 = rocm_q4_K_prefill_tile8_requested();
    const int prefill_tile8_required = rocm_q4_K_prefill_tile8_required();
    const int k1024_tile4_shape =
        blocks == ROCM_Q4_PREFILL_K1024_KBLOCK_TILE &&
        out_dim == DS4_ROCM_Q4_ATTN_Q_B_OUT_DIM;
    const int k1024_tile4_required = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_REQUIRE_Q4_PREFILL_K1024_TILE4") == 1;
    if (prefill_scope && prefill_tile8_required && !prefill_tile8) {
        fprintf(stderr,
                "ds4: required ROCm Q4_K prefill tile8 is disabled "
                "(n_tok=%llu)\n",
                (unsigned long long)n_tok);
        return 0;
    }
    if (prefill_scope && k1024_tile4_shape && k1024_tile4_required &&
        !prefill_tile8) {
        fprintf(stderr,
                "ds4: required ROCm Q4_K prefill K1024 tile4 cannot run "
                "because tiled prefill is disabled (n_tok=%llu)\n",
                (unsigned long long)n_tok);
        return 0;
    }

    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, "q4_K dense");
    if (!wptr) return 0;
    const char *resident_wptr = g_ssd_streaming_mode
        ? rocm_q4_attn_q_b_device_resident_source(
              model_map, weight_offset, weight_bytes)
        : wptr;
    const int weight_device_resident =
        resident_wptr != NULL && resident_wptr == wptr;
    int prefill_wmma = rocm_q4_K_prefill_wmma_select(
        n_tok, in_dim, out_dim, weight_device_resident);
    if (prefill_wmma == ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE) return 0;
    if (prefill_wmma == ROCM_Q4_PREFILL_WMMA_USE &&
        prefill_scope && prefill_tile8_required) {
        if (rocm_q4_attn_q_b_env_bool(
                "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA") == 1) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4_K prefill cannot require both WMMA and TILE8\n");
            return 0;
        }
        /* REQUIRE_TILE8 owns the dispatch when WMMA is only an optional
         * experiment. */
        prefill_wmma = ROCM_Q4_PREFILL_WMMA_FALLBACK;
    }
    if (prefill_wmma == ROCM_Q4_PREFILL_WMMA_USE &&
        k1024_tile4_shape && k1024_tile4_required) {
        if (rocm_q4_attn_q_b_env_bool(
                "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA") == 1) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4_K prefill cannot require both WMMA and K1024 TILE4\n");
            return 0;
        }
        prefill_wmma = ROCM_Q4_PREFILL_WMMA_FALLBACK;
    }
    if (prefill_wmma == ROCM_Q4_PREFILL_WMMA_USE) {
        return rocm_q4_K_prefill_wmma_launch(
            reinterpret_cast<float *>(out->ptr), wptr,
            reinterpret_cast<const float *>(x->ptr),
            (uint32_t)n_tok, 1u, (uint32_t)in_dim, (uint32_t)out_dim,
            row_bytes, in_dim, 0u, out_dim,
            "q4_K dense prefill WMMA64 launch");
    }
    int k1024_tile4 = ROCM_Q4_PREFILL_K1024_TILE4_FALLBACK;
    if (prefill_scope && prefill_tile8) {
        k1024_tile4 = rocm_q4_K_prefill_k1024_tile4_resolve(
            blocks, out_dim, model_map, weight_offset, weight_bytes, wptr);
        if (k1024_tile4 ==
            ROCM_Q4_PREFILL_K1024_TILE4_REQUIRED_FAILURE) {
            return 0;
        }
    }
    cuda_block_q8_K *xq = rocm_q4_K_prequant_alloc(
            n_tok, blocks, "q4_K dense prequant");
    if (!xq) return 0;

    const dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1u);
    q8_K_quantize_kernel<<<qgrid, 256>>>(
            xq, reinterpret_cast<const float *>(x->ptr),
            (uint32_t)in_dim, (uint32_t)n_tok);
    if (!cuda_ok(cudaGetLastError(), "q4_K dense quantize launch")) return 0;

    if (prefill_scope && prefill_tile8) {
        if (k1024_tile4 == ROCM_Q4_PREFILL_K1024_TILE4_USE) {
            const dim3 tiled_grid(
                (unsigned)((out_dim - 1u) /
                           ROCM_Q4_PREFILL_K1024_ROWS + 1u),
                (unsigned)((n_tok - 1u) /
                           ROCM_Q4_PREFILL_TOKEN_TILE + 1u),
                1u);
            rocm_matmul_q4_K_prefill_k1024_tile4_kernel<<<tiled_grid, 256>>>(
                    reinterpret_cast<float *>(out->ptr), wptr, xq,
                    row_bytes, (uint32_t)out_dim, (uint32_t)n_tok);
            const int ok = cuda_ok(
                    cudaGetLastError(),
                    "q4_K dense prefill K1024 tile4 launch");
            if (ok) {
                rocm_q4_K_prefill_tile8_note(1u, 0u, 0u, 1u, n_tok);
            }
            return ok;
        }
        const dim3 tiled_grid((unsigned)((out_dim - 1u) / 32u + 1u),
                              (unsigned)((n_tok - 1u) /
                                         ROCM_Q4_PREFILL_TOKEN_TILE + 1u),
                              1u);
        rocm_matmul_q4_K_prefill_tile8_strided_kernel<<<tiled_grid, 256>>>(
                reinterpret_cast<float *>(out->ptr), wptr, xq, row_bytes,
                (uint32_t)blocks, (uint32_t)out_dim, (uint32_t)n_tok,
                blocks, out_dim);
        const int ok = cuda_ok(cudaGetLastError(),
                               "q4_K dense prefill tile8 launch");
        if (ok) rocm_q4_K_prefill_tile8_note(1u, 0u, 0u, 0u, n_tok);
        return ok;
    }

    const dim3 grid((unsigned)((out_dim - 1u) / 32u + 1u),
                    (unsigned)n_tok, 1u);
    rocm_matmul_q4_K_dense_kernel<<<grid, 256>>>(
            reinterpret_cast<float *>(out->ptr), wptr, xq, row_bytes,
            (uint32_t)blocks, (uint32_t)out_dim, (uint32_t)n_tok);
    return cuda_ok(cudaGetLastError(), "q4_K dense matmul launch");
}

extern "C" int ds4_gpu_matmul_q4_K_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    const int prefill_scope = rocm_q4_K_prefill_tile8_scope(n_tok);
    const int prefill_required = prefill_scope &&
                                 rocm_q4_K_prefill_tile8_required();
    const int wmma_required = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA") == 1;

    /* The fused pair consumes a shared Q8_K activation tile.  When the direct
     * F16 WMMA experiment is selected, return before validation/enqueue so the
     * graph's established fallback issues two dense calls and both can take
     * the strict WMMA path.  This also prevents REQUIRE from falsely passing
     * after silently measuring the TILE8 pair.  SSD preflight is deliberately
     * optimistic about residency: it only decides whether to yield; each dense
     * fallback then proves its exact physical device range before enqueue. */
    if (!prefill_required) {
        const int wmma0 = rocm_q4_K_prefill_wmma_select(
            n_tok, in_dim, out0_dim, 1);
        const int wmma1 = rocm_q4_K_prefill_wmma_select(
            n_tok, in_dim, out1_dim, 1);
        if (wmma0 == ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE ||
            wmma1 == ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE ||
            wmma0 == ROCM_Q4_PREFILL_WMMA_USE ||
            wmma1 == ROCM_Q4_PREFILL_WMMA_USE) {
            return 0;
        }
    } else if (wmma_required) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "Q4_K prefill pair cannot require both WMMA and TILE8\n");
        return 0;
    }

    const int prefill_pair = prefill_scope &&
                             rocm_q4_K_prefill_tile8_requested();
    if (prefill_required && !prefill_pair) {
        fprintf(stderr,
                "ds4: required ROCm Q4_K prefill tile8 pair is disabled "
                "(n_tok=%llu)\n",
                (unsigned long long)n_tok);
        return 0;
    }

    /* Decode keeps its original, separately gated pair path.  Prefill uses
     * the common tile8 gate and shares one canonical Q8_K quantization and
     * one tiled launch across the two projections. */
    const int decode_pair = n_tok <= 8u &&
                            rocm_q4_K_dense_pair_requested();
    if ((!prefill_pair && !decode_pair) ||
        !out0 || !out1 || out0 == out1 ||
        (out0 && out1 && out0->ptr == out1->ptr)) {
        return 0;
    }

    uint64_t blocks0 = 0, blocks1 = 0;
    uint64_t row_bytes0 = 0, row_bytes1 = 0;
    uint64_t weight0_bytes = 0, weight1_bytes = 0;
    if (!rocm_q4_K_dense_validate(out0, model_map, model_size, weight0_offset,
                                  in_dim, out0_dim, x, n_tok, &blocks0,
                                  &row_bytes0, &weight0_bytes) ||
        !rocm_q4_K_dense_validate(out1, model_map, model_size, weight1_offset,
                                  in_dim, out1_dim, x, n_tok, &blocks1,
                                  &row_bytes1, &weight1_bytes) ||
        blocks0 != blocks1 || row_bytes0 != row_bytes1) {
        return 0;
    }
    uint64_t out0_bytes = 0;
    uint64_t out1_bytes = 0;
    if (!cuda_u64_mul3_checked(n_tok, out0_dim, sizeof(float), &out0_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out1_dim, sizeof(float), &out1_bytes) ||
        rocm_q4_K_byte_ranges_overlap(out0->ptr, out0_bytes,
                                      out1->ptr, out1_bytes)) {
        return 0;
    }

    const char *w0 = cuda_model_range_ptr(model_map, weight0_offset,
                                          weight0_bytes, "q4_K dense pair0");
    const char *w1 = cuda_model_range_ptr(model_map, weight1_offset,
                                          weight1_bytes, "q4_K dense pair1");
    if (!w0 || !w1) return 0;
    cuda_block_q8_K *xq = rocm_q4_K_prequant_alloc(
            n_tok, blocks0, "q4_K dense pair prequant");
    if (!xq) return 0;

    const dim3 qgrid((unsigned)blocks0, (unsigned)n_tok, 1u);
    q8_K_quantize_kernel<<<qgrid, 256>>>(
            xq, reinterpret_cast<const float *>(x->ptr),
            (uint32_t)in_dim, (uint32_t)n_tok);
    if (!cuda_ok(cudaGetLastError(), "q4_K dense pair quantize launch")) {
        return 0;
    }

    if (prefill_pair) {
        const uint64_t out0_tiles = (out0_dim - 1u) / 32u + 1u;
        const uint64_t out1_tiles = (out1_dim - 1u) / 32u + 1u;
        const dim3 grid((unsigned)(out0_tiles + out1_tiles),
                        (unsigned)((n_tok - 1u) /
                                   ROCM_Q4_PREFILL_TOKEN_TILE + 1u),
                        1u);
        rocm_matmul_q4_K_prefill_tile8_pair_kernel<<<grid, 256>>>(
                reinterpret_cast<float *>(out0->ptr),
                reinterpret_cast<float *>(out1->ptr), w0, w1, xq,
                row_bytes0, (uint32_t)blocks0, (uint32_t)out0_dim,
                (uint32_t)out1_dim, (uint32_t)n_tok);
        const int ok = cuda_ok(cudaGetLastError(),
                               "q4_K dense prefill pair tile8 launch");
        if (ok) rocm_q4_K_prefill_tile8_note(0u, 1u, 0u, 0u, n_tok);
        return ok;
    }

    const uint64_t out0_tiles = (out0_dim - 1u) / 32u + 1u;
    const uint64_t out1_tiles = (out1_dim - 1u) / 32u + 1u;
    const dim3 grid((unsigned)(out0_tiles + out1_tiles),
                    (unsigned)n_tok, 1u);
    rocm_matmul_q4_K_dense_pair_kernel<<<grid, 256>>>(
            reinterpret_cast<float *>(out0->ptr),
            reinterpret_cast<float *>(out1->ptr), w0, w1, xq,
            row_bytes0, (uint32_t)blocks0, (uint32_t)out0_dim,
            (uint32_t)out1_dim, (uint32_t)n_tok);
    return cuda_ok(cudaGetLastError(), "q4_K dense pair matmul launch");
}

extern "C" int ds4_gpu_attention_output_low_q4_K_slice_tensor(
        ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t group0, uint32_t group_cnt,
        const ds4_gpu_tensor *heads, int resident_decode) {
    const int disabled =
        getenv("DS4_ROCM_DISABLE_Q4_GROUPED_ATTN_A") != NULL;
    const int required =
        getenv("DS4_ROCM_REQUIRE_Q4_GROUPED_ATTN_A") != NULL;
    const int enabled =
        getenv("DS4_ROCM_ENABLE_Q4_GROUPED_ATTN_A") != NULL;
    const int resident_default =
        rocm_q4_K_grouped_attn_a_resident_default_scope(
            group_dim, rank, group0, group_cnt, resident_decode);
    /* DISABLE is authoritative; REQUIRE reports that rollback as a failure
     * instead of allowing the graph to false-green through its fallback. */
    if (disabled) {
        if (required) {
            fprintf(stderr,
                    "ds4: required ROCm Q4_K grouped attention-A decode "
                    "is disabled\n");
        }
        return rocm_q4_K_grouped_attn_a_result(required ? -1 : 0, 0u);
    }
    if (!resident_default && !enabled && !required) {
        return rocm_q4_K_grouped_attn_a_result(0, 0u);
    }
    const int pre_enqueue_failure = required ? -1 : 0;
    if (!low || !heads || !model_map || group_dim == 0u || rank == 0u ||
        group_cnt == 0u || group_dim > UINT32_MAX || rank > UINT32_MAX ||
        group_cnt > UINT16_MAX || (group_dim % CUDA_QK_K) != 0u ||
        group0 > UINT32_MAX - group_cnt) {
        return rocm_q4_K_grouped_attn_a_result(pre_enqueue_failure, 0u);
    }

    const uint64_t blocks = group_dim / CUDA_QK_K;
    uint64_t row_bytes = 0, group_weight_bytes = 0, group_skip = 0;
    uint64_t selected_weight_bytes = 0, selected_offset = 0;
    if (blocks == 0u ||
        !cuda_u64_mul_checked(blocks, sizeof(cuda_block_q4_K), &row_bytes) ||
        !cuda_u64_mul_checked(rank, row_bytes, &group_weight_bytes) ||
        !cuda_u64_mul_checked(group0, group_weight_bytes, &group_skip) ||
        !cuda_u64_mul_checked(group_cnt, group_weight_bytes,
                              &selected_weight_bytes) ||
        !cuda_u64_add_checked(out_a_offset, group_skip, &selected_offset) ||
        !cuda_model_range_fits(model_size, selected_offset,
                               selected_weight_bytes) ||
        !cuda_tensor_has_elems2(heads, group_cnt, group_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(low, group_cnt, rank, sizeof(float))) {
        return rocm_q4_K_grouped_attn_a_result(pre_enqueue_failure, 0u);
    }

    const char *w = cuda_model_range_ptr(
        model_map, selected_offset, selected_weight_bytes,
        "q4_K grouped attention output A decode");
    cuda_block_q8_K *xq = rocm_q4_K_prequant_alloc(
        group_cnt, blocks, "q4_K grouped attention output A decode prequant");
    if (!w || !xq) {
        return rocm_q4_K_grouped_attn_a_result(pre_enqueue_failure, 0u);
    }

    const dim3 qgrid((unsigned)blocks, group_cnt, 1u);
    q8_K_quantize_kernel<<<qgrid, 256>>>(
        xq, reinterpret_cast<const float *>(heads->ptr),
        (uint32_t)group_dim, group_cnt);
    if (!cuda_ok(cudaGetLastError(),
                 "q4_K grouped attention output A decode quantize launch")) {
        return rocm_q4_K_grouped_attn_a_result(-1, 0u);
    }
    const dim3 grid((unsigned)((rank - 1u) / 32u + 1u), 1u, group_cnt);
    rocm_matmul_q4_K_dense_grouped_decode_kernel<<<grid, 256>>>(
        reinterpret_cast<float *>(low->ptr), w, xq, row_bytes,
        (uint32_t)blocks, (uint32_t)rank, group_cnt);
    if (!cuda_ok(cudaGetLastError(),
                 "q4_K grouped attention output A decode matmul launch")) {
        return rocm_q4_K_grouped_attn_a_result(-1, 0u);
    }
    return rocm_q4_K_grouped_attn_a_result(1, group_cnt);
}

/* Quantize token-major [token][group][K] rows once, then apply group-major
 * [group][out_row][K] Q4_K weights directly into token-major output.  A
 * return of -1 means the quantize launch was accepted and callers must not
 * replay a row fallback over potentially submitted work. */
static int rocm_q4_K_prefill_tile8_quant_launch(
        float *out,
        const char *w,
        const float *x,
        uint32_t n_tok,
        uint32_t n_groups,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes,
        int prefill_wmma,
        const char *label) {
    uint64_t n_rows = 0;
    uint64_t xq_token_stride = 0;
    uint64_t x_token_stride = 0;
    uint64_t out_token_stride = 0;
    const uint64_t blocks = in_dim / CUDA_QK_K;
    if (!out || !w || !x || n_tok == 0u || n_groups == 0u ||
        in_dim == 0u || out_dim == 0u || blocks == 0u ||
        (in_dim % CUDA_QK_K) != 0u ||
        !cuda_u64_mul_checked(n_tok, n_groups, &n_rows) ||
        /* HIP keeps the portable grid-y limit at 65535.  Real AProjQ4 uses
         * eight groups, so even the 4096-token ceiling remains in range. */
        n_rows > UINT16_MAX ||
        !cuda_u64_mul_checked(n_groups, blocks, &xq_token_stride) ||
        !cuda_u64_mul_checked(n_groups, in_dim, &x_token_stride) ||
        !cuda_u64_mul_checked(n_groups, out_dim, &out_token_stride)) {
        return 0;
    }

    if (prefill_wmma == ROCM_Q4_PREFILL_WMMA_USE) {
        return rocm_q4_K_prefill_wmma_launch(
                   out, w, x, n_tok, n_groups, in_dim, out_dim, row_bytes,
                   x_token_stride, in_dim, out_token_stride,
                   label ? label : "q4_K attention-output WMMA64 launch")
            ? 1 : -1;
    }

    cuda_block_q8_K *xq = rocm_q4_K_prequant_alloc(
            n_rows, blocks, label ? label : "q4_K prefill tile8 prequant");
    if (!xq) return 0;

    const dim3 qgrid((unsigned)blocks, (unsigned)n_rows, 1u);
    q8_K_quantize_kernel<<<qgrid, 256>>>(
            xq, x, in_dim, (uint32_t)n_rows);
    if (!cuda_ok(cudaGetLastError(),
                 "q4_K prefill tile8 quantize launch")) {
        return 0;
    }

    const dim3 grid((unsigned)((out_dim - 1u) / 32u + 1u),
                    (unsigned)((n_tok - 1u) /
                               ROCM_Q4_PREFILL_TOKEN_TILE + 1u),
                    n_groups);
    rocm_matmul_q4_K_prefill_tile8_strided_kernel<<<grid, 256>>>(
            out, w, xq, row_bytes, (uint32_t)blocks, out_dim, n_tok,
            xq_token_stride, out_token_stride);
    if (!cuda_ok(cudaGetLastError(),
                 "q4_K prefill tile8 matmul launch")) {
        return -1;
    }
    return 1;
}

extern "C" int ds4_gpu_attention_output_q4_K_batch_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *low,
        ds4_gpu_tensor *group_tmp,
        ds4_gpu_tensor *low_tmp,
        const void *model_map,
        uint64_t model_size,
        uint64_t out_a_offset,
        uint64_t out_b_offset,
        uint32_t out_b_type,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint64_t out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t n_tokens) {
    (void)group_tmp;
    (void)low_tmp;

    const int tile8_scope = rocm_q4_K_prefill_tile8_scope(n_tokens);
    const int tile8_requested = rocm_q4_K_prefill_tile8_requested();
    const int tile8_required = tile8_scope &&
                               rocm_q4_K_prefill_tile8_required();
    const int wmma_enabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA") == 1;
    const int wmma_ssd_enabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA_SSD") == 1;
    const int wmma_disabled = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA") == 1;
    const int wmma_required = rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA") == 1;
    const int wmma_requested = wmma_required ||
        (!wmma_disabled && ((!g_ssd_streaming_mode && wmma_enabled) ||
                           (g_ssd_streaming_mode && wmma_ssd_enabled)));
    if (!tile8_scope) return 0;
    if (!tile8_requested && !wmma_requested) {
        if (tile8_required) {
            fprintf(stderr,
                    "ds4: required ROCm Q4_K attention-output prefill "
                    "tile8 is disabled (n_tok=%u)\n",
                    n_tokens);
            return -1;
        }
        return 0;
    }
    const int pre_enqueue_failure =
        (tile8_required || wmma_required) ? -1 : 0;

    if (!out || !low || !heads || !model_map || group_dim == 0u ||
        rank == 0u || n_groups == 0u || out_dim == 0u ||
        n_groups > UINT16_MAX || group_dim > UINT32_MAX ||
        rank > UINT32_MAX || out_dim > UINT32_MAX ||
        (group_dim % CUDA_QK_K) != 0u ||
        (out_b_type != 12u && out_b_type != 8u)) {
        return pre_enqueue_failure;
    }

    uint64_t low_dim = 0;
    uint64_t heads_rows = 0;
    uint64_t heads_bytes = 0;
    uint64_t low_bytes = 0;
    uint64_t out_bytes = 0;
    if (!cuda_u64_mul_checked(n_groups, rank, &low_dim) ||
        low_dim == 0u || low_dim > UINT32_MAX ||
        !cuda_u64_mul_checked(n_tokens, n_groups, &heads_rows) ||
        !cuda_u64_mul3_checked(heads_rows, group_dim,
                               sizeof(float), &heads_bytes) ||
        !cuda_u64_mul3_checked(n_tokens, low_dim,
                               sizeof(float), &low_bytes) ||
        !cuda_u64_mul3_checked(n_tokens, out_dim,
                               sizeof(float), &out_bytes) ||
        heads->bytes < heads_bytes || low->bytes < low_bytes ||
        out->bytes < out_bytes) {
        return pre_enqueue_failure;
    }

    const uint64_t a_blocks = group_dim / CUDA_QK_K;
    uint64_t row_a_bytes = 0;
    uint64_t out_a_bytes = 0;
    if (!cuda_u64_mul_checked(a_blocks, sizeof(cuda_block_q4_K),
                              &row_a_bytes) ||
        !cuda_u64_mul_checked(low_dim, row_a_bytes, &out_a_bytes) ||
        !cuda_model_range_fits(model_size, out_a_offset, out_a_bytes)) {
        return pre_enqueue_failure;
    }

    uint64_t row_b_bytes = 0;
    uint64_t out_b_bytes = 0;
    if (out_b_type == 12u) {
        if ((low_dim % CUDA_QK_K) != 0u ||
            !cuda_u64_mul_checked(low_dim / CUDA_QK_K,
                                  sizeof(cuda_block_q4_K), &row_b_bytes)) {
            return pre_enqueue_failure;
        }
    } else {
        const uint64_t b_blocks = (low_dim + 31u) / 32u;
        if (!cuda_u64_mul_checked(b_blocks, 34u, &row_b_bytes)) {
            return pre_enqueue_failure;
        }
    }
    if (!cuda_u64_mul_checked(out_dim, row_b_bytes, &out_b_bytes) ||
        !cuda_model_range_fits(model_size, out_b_offset, out_b_bytes)) {
        return pre_enqueue_failure;
    }

    const char *out_a = cuda_model_range_ptr(
            model_map, out_a_offset, out_a_bytes, "q4_K attention output A");
    const char *out_b = cuda_model_range_ptr(
            model_map, out_b_offset, out_b_bytes, "q4_K attention output B");
    if (!out_a || !out_b) return pre_enqueue_failure;

    const char *resident_out_a = g_ssd_streaming_mode
        ? rocm_q4_attn_q_b_device_resident_source(
              model_map, out_a_offset, out_a_bytes)
        : out_a;
    const char *resident_out_b = g_ssd_streaming_mode && out_b_type == 12u
        ? rocm_q4_attn_q_b_device_resident_source(
              model_map, out_b_offset, out_b_bytes)
        : out_b;
    const int out_a_device_resident =
        resident_out_a != NULL && resident_out_a == out_a;
    const int out_b_device_resident =
        resident_out_b != NULL && resident_out_b == out_b;

    /* Resolve every strict WMMA decision before A can enqueue work.  That
     * keeps REQUIRE fail-closed: an ineligible B projection can never make
     * the graph replay a fallback over an already submitted A projection. */
    int a_wmma = rocm_q4_K_prefill_wmma_select(
        n_tokens, group_dim, rank, out_a_device_resident);
    int b_wmma = out_b_type == 12u
        ? rocm_q4_K_prefill_wmma_select(
              n_tokens, low_dim, out_dim, out_b_device_resident)
        : ROCM_Q4_PREFILL_WMMA_FALLBACK;
    if (a_wmma == ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE ||
        b_wmma == ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE) {
        return -1;
    }
    if (tile8_required &&
        (a_wmma == ROCM_Q4_PREFILL_WMMA_USE ||
         b_wmma == ROCM_Q4_PREFILL_WMMA_USE)) {
        if (wmma_required) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4_K attention-output prefill cannot require both "
                    "WMMA and TILE8\n");
            return -1;
        }
        a_wmma = ROCM_Q4_PREFILL_WMMA_FALLBACK;
        b_wmma = ROCM_Q4_PREFILL_WMMA_FALLBACK;
    }
    if (!tile8_requested &&
        (a_wmma != ROCM_Q4_PREFILL_WMMA_USE ||
         (out_b_type == 12u &&
          b_wmma != ROCM_Q4_PREFILL_WMMA_USE))) {
        return pre_enqueue_failure;
    }

    /* A: the WMMA candidate consumes F32 heads directly; TILE8 retains one
     * quantization over [token,group].  Neither path needs group pack/unpack
     * buffers or an n_tokens*n_groups dispatch loop. */
    const int a_rc = rocm_q4_K_prefill_tile8_quant_launch(
            reinterpret_cast<float *>(low->ptr), out_a,
            reinterpret_cast<const float *>(heads->ptr), n_tokens, n_groups,
            (uint32_t)group_dim, (uint32_t)rank, row_a_bytes,
            a_wmma,
            "q4_K attention output A WMMA64/tile8");
    if (a_rc <= 0) {
        return a_rc < 0 ? -1 : pre_enqueue_failure;
    }

    int b_rc = 0;
    if (out_b_type == 12u) {
        b_rc = rocm_q4_K_prefill_tile8_quant_launch(
                reinterpret_cast<float *>(out->ptr), out_b,
                reinterpret_cast<const float *>(low->ptr), n_tokens, 1u,
                (uint32_t)low_dim, (uint32_t)out_dim, row_b_bytes,
                b_wmma,
                "q4_K attention output B WMMA64/tile8");
    } else {
        b_rc = ds4_gpu_matmul_q8_0_tensor(
                out, model_map, model_size, out_b_offset, low_dim, out_dim,
                low, n_tokens);
    }
    if (b_rc <= 0) return -1;

    if (a_wmma != ROCM_Q4_PREFILL_WMMA_USE &&
        b_wmma != ROCM_Q4_PREFILL_WMMA_USE) {
        rocm_q4_K_prefill_tile8_note(0u, 0u, 1u, 0u, n_tokens);
    }
    return 1;
}
