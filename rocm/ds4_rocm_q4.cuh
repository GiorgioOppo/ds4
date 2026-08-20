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
    ROCM_Q4_Q8K_WORDS = sizeof(cuda_block_q8_K) / sizeof(uint32_t),
};
static_assert((sizeof(cuda_block_q8_K) % sizeof(uint32_t)) == 0u,
              "ROCm Q8_K LDS copies require a whole number of words");

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

static int rocm_q4_K_dense_pair_requested(void) {
    return getenv("DS4_ROCM_ENABLE_Q4_DENSE_PAIR") != NULL &&
           getenv("DS4_ROCM_DISABLE_Q4_DENSE_PAIR") == NULL;
}

static int rocm_q4_K_prefill_tile8_scope(uint64_t n_tok) {
    /* Keep decode/speculative micro-batches on the latency-oriented legacy
     * kernel.  4096 is DS4's largest supported prefill chunk and bounds the
     * A/B surface while this path remains opt-in. */
    return n_tok > 8u && n_tok <= 4096u;
}

static int rocm_q4_K_prefill_tile8_requested(void) {
    return getenv("DS4_ROCM_ENABLE_Q4_PREFILL_TILE8") != NULL &&
           getenv("DS4_ROCM_DISABLE_Q4_PREFILL_TILE8") == NULL;
}

static int rocm_q4_K_prefill_tile8_required(void) {
    return getenv("DS4_ROCM_REQUIRE_Q4_PREFILL_TILE8") != NULL;
}

static uint64_t g_rocm_q4_prefill_tile8_dense_calls;
static uint64_t g_rocm_q4_prefill_tile8_pair_calls;
static uint64_t g_rocm_q4_prefill_tile8_attention_batch_calls;
static uint64_t g_rocm_q4_prefill_tile8_tokens;
static int g_rocm_q4_prefill_tile8_report_registered;

static void rocm_q4_K_prefill_tile8_report(void) {
    fprintf(stderr,
            "ds4: ROCm Q4_K prefill tile8 stats: "
            "dense_calls=%llu pair_calls=%llu attention_batch_calls=%llu "
            "tokens=%llu\n",
            (unsigned long long)g_rocm_q4_prefill_tile8_dense_calls,
            (unsigned long long)g_rocm_q4_prefill_tile8_pair_calls,
            (unsigned long long)g_rocm_q4_prefill_tile8_attention_batch_calls,
            (unsigned long long)g_rocm_q4_prefill_tile8_tokens);
}

static void rocm_q4_K_prefill_tile8_note(
        uint32_t dense_calls,
        uint32_t pair_calls,
        uint32_t attention_batch_calls,
        uint64_t tokens) {
    if (getenv("DS4_ROCM_Q4_PREFILL_TILE8_STATS") == NULL) return;
    if (!g_rocm_q4_prefill_tile8_report_registered) {
        g_rocm_q4_prefill_tile8_report_registered = 1;
        (void)atexit(rocm_q4_K_prefill_tile8_report);
    }
    g_rocm_q4_prefill_tile8_dense_calls += dense_calls;
    g_rocm_q4_prefill_tile8_pair_calls += pair_calls;
    g_rocm_q4_prefill_tile8_attention_batch_calls += attention_batch_calls;
    g_rocm_q4_prefill_tile8_tokens += tokens;
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
    if (prefill_scope && prefill_tile8_required && !prefill_tile8) {
        fprintf(stderr,
                "ds4: required ROCm Q4_K prefill tile8 is disabled "
                "(n_tok=%llu)\n",
                (unsigned long long)n_tok);
        return 0;
    }

    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, "q4_K dense");
    if (!wptr) return 0;
    cuda_block_q8_K *xq = rocm_q4_K_prequant_alloc(
            n_tok, blocks, "q4_K dense prequant");
    if (!xq) return 0;

    const dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1u);
    q8_K_quantize_kernel<<<qgrid, 256>>>(
            xq, reinterpret_cast<const float *>(x->ptr),
            (uint32_t)in_dim, (uint32_t)n_tok);
    if (!cuda_ok(cudaGetLastError(), "q4_K dense quantize launch")) return 0;

    if (prefill_scope && prefill_tile8) {
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
        if (ok) rocm_q4_K_prefill_tile8_note(1u, 0u, 0u, n_tok);
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
    const int prefill_pair = prefill_scope &&
                             rocm_q4_K_prefill_tile8_requested();
    const int prefill_required = prefill_scope &&
                                 rocm_q4_K_prefill_tile8_required();
    if (prefill_required && !prefill_pair) {
        fprintf(stderr,
                "ds4: required ROCm Q4_K prefill tile8 pair is disabled "
                "(n_tok=%llu)\n",
                (unsigned long long)n_tok);
        return 0;
    }

    /* Decode keeps its original, separately gated pair path.  Prefill uses
     * the common tile8 gate and shares one canonical Q8_K quantization across
     * the two projections, then issues one tiled launch per weight matrix. */
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
        const dim3 grid0((unsigned)((out0_dim - 1u) / 32u + 1u),
                         (unsigned)((n_tok - 1u) /
                                    ROCM_Q4_PREFILL_TOKEN_TILE + 1u),
                         1u);
        rocm_matmul_q4_K_prefill_tile8_strided_kernel<<<grid0, 256>>>(
                reinterpret_cast<float *>(out0->ptr), w0, xq, row_bytes0,
                (uint32_t)blocks0, (uint32_t)out0_dim, (uint32_t)n_tok,
                blocks0, out0_dim);
        if (!cuda_ok(cudaGetLastError(),
                     "q4_K dense prefill pair0 tile8 launch")) {
            return 0;
        }

        const dim3 grid1((unsigned)((out1_dim - 1u) / 32u + 1u),
                         (unsigned)((n_tok - 1u) /
                                    ROCM_Q4_PREFILL_TOKEN_TILE + 1u),
                         1u);
        rocm_matmul_q4_K_prefill_tile8_strided_kernel<<<grid1, 256>>>(
                reinterpret_cast<float *>(out1->ptr), w1, xq, row_bytes1,
                (uint32_t)blocks1, (uint32_t)out1_dim, (uint32_t)n_tok,
                blocks1, out1_dim);
        const int ok = cuda_ok(cudaGetLastError(),
                               "q4_K dense prefill pair1 tile8 launch");
        if (ok) rocm_q4_K_prefill_tile8_note(0u, 1u, 0u, n_tok);
        return ok;
    }

    // Use the exact standalone kernel twice. This preserves its block walk and
    // reduction path while still eliminating the second Q8_K quantization.
    const dim3 grid0((unsigned)((out0_dim - 1u) / 32u + 1u),
                     (unsigned)n_tok, 1u);
    rocm_matmul_q4_K_dense_kernel<<<grid0, 256>>>(
            reinterpret_cast<float *>(out0->ptr), w0, xq, row_bytes0,
            (uint32_t)blocks0, (uint32_t)out0_dim, (uint32_t)n_tok);
    if (!cuda_ok(cudaGetLastError(), "q4_K dense pair0 matmul launch")) {
        return 0;
    }

    const dim3 grid1((unsigned)((out1_dim - 1u) / 32u + 1u),
                     (unsigned)n_tok, 1u);
    rocm_matmul_q4_K_dense_kernel<<<grid1, 256>>>(
            reinterpret_cast<float *>(out1->ptr), w1, xq, row_bytes1,
            (uint32_t)blocks1, (uint32_t)out1_dim, (uint32_t)n_tok);
    return cuda_ok(cudaGetLastError(), "q4_K dense pair1 matmul launch");
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
        const char *label) {
    uint64_t n_rows = 0;
    uint64_t xq_token_stride = 0;
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
        !cuda_u64_mul_checked(n_groups, out_dim, &out_token_stride)) {
        return 0;
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
    if (!tile8_scope) return 0;
    if (!tile8_requested) {
        if (tile8_required) {
            fprintf(stderr,
                    "ds4: required ROCm Q4_K attention-output prefill "
                    "tile8 is disabled (n_tok=%u)\n",
                    n_tokens);
            return -1;
        }
        return 0;
    }
    const int pre_enqueue_failure = tile8_required ? -1 : 0;

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

    /* A: one quantization over [token,group] plus one z-grouped tile8 launch;
     * no group pack/unpack buffers or n_tokens*n_groups dispatch loop. */
    const int a_rc = rocm_q4_K_prefill_tile8_quant_launch(
            reinterpret_cast<float *>(low->ptr), out_a,
            reinterpret_cast<const float *>(heads->ptr), n_tokens, n_groups,
            (uint32_t)group_dim, (uint32_t)rank, row_a_bytes,
            "q4_K attention output A prequant");
    if (a_rc <= 0) {
        return a_rc < 0 ? -1 : pre_enqueue_failure;
    }

    int b_rc = 0;
    if (out_b_type == 12u) {
        b_rc = rocm_q4_K_prefill_tile8_quant_launch(
                reinterpret_cast<float *>(out->ptr), out_b,
                reinterpret_cast<const float *>(low->ptr), n_tokens, 1u,
                (uint32_t)low_dim, (uint32_t)out_dim, row_b_bytes,
                "q4_K attention output B prequant");
    } else {
        b_rc = ds4_gpu_matmul_q8_0_tensor(
                out, model_map, model_size, out_b_offset, low_dim, out_dim,
                low, n_tokens);
    }
    if (b_rc <= 0) return -1;

    rocm_q4_K_prefill_tile8_note(0u, 0u, 1u, n_tokens);
    return 1;
}
