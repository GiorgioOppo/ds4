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
    // Keep this optional hook scoped to decode/speculative micro-batches. Larger
    // prefill batches fall back to the two required dense primitives.
    if (!rocm_q4_K_dense_pair_requested() ||
        n_tok > 8u || !out0 || !out1 || out0 == out1 ||
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
