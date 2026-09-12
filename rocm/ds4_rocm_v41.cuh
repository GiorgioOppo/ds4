/* DeepSeek V4.1 baseline primitives. Float-addressable storage preserves the released BF16/FP8/FP4 graph boundaries. */
#ifdef __HIP_PLATFORM_AMD__

/* AMDGPU ignores float_control(precise), and HIP's math wrappers select native trig while preprocessing -ffast-math. Use explicit OCML operations and division instructions so the existing translation unit keeps its tuned arithmetic. */
__device__ static float v41_add(float x, float y) { return __ocml_add_rte_f32(x, y); }
__device__ static float v41_sub(float x, float y) { return __ocml_sub_rte_f32(x, y); }
__device__ static float v41_mul(float x, float y) { return __ocml_mul_rte_f32(x, y); }
__device__ static float v41_div(float x, float y) {
    /* ROCm 10 declares but does not define __ocml_div_rte_f32. A local reciprocal(off) pragma still leaves afn, which AMDGPU lowers to an approximate reciprocal. Encode the backend's full F32 division refinement, including temporary denorm preservation, as one indivisible block. */
    float q, d, n, r, e;
    uint32_t mode;
#define DS4_V41_DIV_ASM(VCC) asm volatile( \
        "v_div_scale_f32 %1, " VCC ", %7, %7, %6\n\t" \
        "v_div_scale_f32 %2, " VCC ", %6, %7, %6\n\t" \
        "v_rcp_f32 %3, %1\n\t" \
        "s_getreg_b32 %5, hwreg(HW_REG_MODE, 4, 2)\n\t" \
        "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3\n\t" \
        "v_fma_f32 %4, -%1, %3, 1.0\n\t" \
        "v_fma_f32 %3, %4, %3, %3\n\t" \
        "v_mul_f32 %0, %2, %3\n\t" \
        "v_fma_f32 %4, -%1, %0, %2\n\t" \
        "v_fma_f32 %0, %4, %3, %0\n\t" \
        "v_fma_f32 %4, -%1, %0, %2\n\t" \
        "s_setreg_b32 hwreg(HW_REG_MODE, 4, 2), %5\n\t" \
        "v_div_fmas_f32 %0, %4, %3, %0\n\t" \
        "v_div_fixup_f32 %0, %0, %7, %6\n\t" \
        : "=&v"(q), "=&v"(d), "=&v"(n), "=&v"(r), "=&v"(e), "=&s"(mode) \
        : "v"(x), "v"(y) : "vcc", "memory")
    /* The HIP compiler need not define a wave-size macro. Its target builtin folds before assembly, selecting the valid VCC operand for the actual wave mode. */
#if defined(__AMDGCN__)
    if (__builtin_amdgcn_wavefrontsize() == 64) DS4_V41_DIV_ASM("vcc");
    else
#endif
        DS4_V41_DIV_ASM("vcc_lo");
#undef DS4_V41_DIV_ASM
    return q;
}

__device__ static float v41_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

__device__ static float v41_pow2_ceil(float x) {
    const uint32_t bits = __float_as_uint(x);
    return __uint_as_float((bits & 0x7f800000u) + ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

__device__ static float v41_sum32(float x) {
    for (int delta = 16; delta; delta >>= 1)
        x = v41_add(x, __shfl_down(x, delta, 32));
    return __shfl(x, 0, 32);
}

__global__ static void v41_bf16_kernel(float *x, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = v41_bf16(x[i]);
}

__global__ static void v41_quantize_kernel(float *x, uint32_t format) {
    const uint32_t block = format == DS4_V41_FP4_E4M3 ? 16u : 32u;
    const uint32_t lane = threadIdx.x;
    const uint64_t i = (uint64_t)blockIdx.x * block + lane;
    const float value = lane < block ? v41_bf16(x[i]) : 0.0f;
    float amax = fabsf(value);
    for (int delta = 16; delta; delta >>= 1)
        amax = fmaxf(amax, __shfl_down(amax, delta, 32));
    amax = __shfl(amax, 0, 32);
    float result;
    if (format == DS4_V41_FP8_E8M0) {
        const float scale = v41_pow2_ceil(v41_mul(fmaxf(amax, 1.0e-4f), 1.0f / 448.0f));
        result = v41_mul(dsv4_e4m3fn_dequant_dev(v41_div(fabsf(value), scale)), scale);
    } else {
        const float scale = format == DS4_V41_FP4_E4M3 ?
            dsv4_e4m3fn_dequant_dev(v41_div(fmaxf(amax, 0.01171875f), 6.0f)) :
            v41_pow2_ceil(v41_mul(fmaxf(amax, 7.052966104933725e-38f), 1.0f / 6.0f));
        result = v41_mul(dsv4_e2m1fn_dequant_dev(v41_div(fabsf(value), scale)), scale);
    }
    if (lane < block) {
        /* Preserve signed zero even though the surrounding translation unit permits -fno-signed-zeros. */
        const uint32_t bits = (__float_as_uint(v41_bf16(result)) & 0x7fffffffu) | (__float_as_uint(value) & 0x80000000u);
        x[i] = __uint_as_float(bits);
    }
}

extern "C" int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *x, uint32_t width, uint32_t rows,
                                      ds4_v41_activation_format format) {
    const uint32_t block = format == DS4_V41_FP4_E4M3 ? 16u : 32u;
    if (!width || !rows || format < DS4_V41_BF16 || format > DS4_V41_FP4_E4M3 ||
        (format != DS4_V41_BF16 && width % block) ||
        !cuda_tensor_has_elems2(x, width, rows, sizeof(float))) return 0;
    if (format == DS4_V41_BF16) {
        const uint64_t count = (uint64_t)width * rows;
        if ((count + 255u) / 256u > UINT32_MAX) return 0;
        v41_bf16_kernel<<<(unsigned)((count + 255u) / 256u), 256>>>((float *)x->ptr, count);
    } else {
        const uint64_t blocks = (uint64_t)(width / block) * rows;
        if (blocks > UINT32_MAX) return 0;
        v41_quantize_kernel<<<(unsigned)blocks, 32>>>((float *)x->ptr, format);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 activation quantization");
}

struct v41_rope_args {
    uint32_t width, heads, start, stride, inverse;
    float frequencies[32];
};

__global__ static void v41_rope_kernel(float *x, v41_rope_args args) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x / args.heads;
    const float theta = v41_mul((float)(args.start + row * args.stride), args.frequencies[lane]);
    const float c = __ocml_cos_f32(theta), s = args.inverse ? -__ocml_sin_f32(theta) : __ocml_sin_f32(theta);
    const uint64_t i = (uint64_t)blockIdx.x * args.width + args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = v41_bf16(v41_sub(v41_mul(re, c), v41_mul(im, s)));
    x[i + 1u] = v41_bf16(v41_add(v41_mul(re, s), v41_mul(im, c)));
}

static float v41_rope_frequencies[2][32];
static pthread_once_t v41_rope_once = PTHREAD_ONCE_INIT;

/* Keep the reference's host pow/reciprocal and YaRN operation order; a frequency ULP grows into a phase error at long positions. */
#ifndef __HIP_DEVICE_COMPILE__
#pragma float_control(precise, on, push)
#pragma clang fp contract(off)
#endif
static void v41_init_rope_frequencies(void) {
    for (int kind = 0; kind < 2; kind++) {
        const float base = kind ? 160000.0f : 10000.0f;
        const float low = (float)floor(64.0 * log(65536.0 / (32.0 * 2.0 * M_PI)) / (2.0 * log(base)));
        const float high = (float)ceil(64.0 * log(65536.0 / (2.0 * M_PI)) / (2.0 * log(base)));
        for (int i = 0; i < 32; i++) {
            const float denominator = powf(base, (float)i / 32.0f);
            float f = 1.0f / denominator;
            if (kind) {
                const float ramp = fminf(1.0f, fmaxf(0.0f, (i - low) / (high - low)));
                const float smooth = 1.0f - ramp;
                const float interpolate = (f / 16.0f) * (1.0f - smooth);
                const float extrapolate = f * smooth;
                f = interpolate + extrapolate;
            }
            v41_rope_frequencies[kind][i] = f;
        }
    }
}
#ifndef __HIP_DEVICE_COMPILE__
#pragma float_control(pop)
#endif

extern "C" int ds4_gpu_dsv41_rope_stride(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                                         uint32_t rows, uint32_t start, uint32_t stride,
                                         bool compressed, bool inverse) {
    uint64_t elems = 0;
    if (width < 64u || !heads || !rows || rows > 1048576u || !stride ||
        (uint64_t)start + (uint64_t)(rows - 1u) * stride >= 1048576u ||
        (uint64_t)heads * rows > UINT32_MAX ||
        !cuda_u64_mul3_checked(width, heads, rows, &elems) || !cuda_tensor_has_f32(x, elems)) return 0;
    if (pthread_once(&v41_rope_once, v41_init_rope_frequencies)) return 0;
    v41_rope_args args = {width, heads, start, stride, inverse, {0}};
    memcpy(args.frequencies, v41_rope_frequencies[compressed ? 1 : 0], sizeof(args.frequencies));
    v41_rope_kernel<<<heads * rows, 32>>>((float *)x->ptr, args);
    return cuda_ok(cudaGetLastError(), "V4.1 unit-magnitude RoPE");
}

extern "C" int ds4_gpu_dsv41_rope(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                                  uint32_t rows, uint32_t start, bool compressed, bool inverse) {
    return ds4_gpu_dsv41_rope_stride(x, width, heads, rows, start, 1, compressed, inverse);
}

__global__ static void v41_engram_kernel(float *residual, const float *kv, const float *qw,
                                        const float *kw, const uint8_t *mask, uint32_t width, float eps) {
    const uint32_t token = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    if (mask && !mask[token]) return;
    const uint64_t offset = ((uint64_t)token * 4u + head) * width;
    const uint64_t key = ((uint64_t)token * 5u + head) * width;
    const uint64_t value = ((uint64_t)token * 5u + 4u) * width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint32_t i = lane; i < width; i += 32u) {
        const float h = residual[offset + i], k = v41_bf16(kv[key + i]);
        const uint64_t wi = (uint64_t)head * width + i;
        h2 = v41_add(h2, v41_mul(h, h));
        k2 = v41_add(k2, v41_mul(k, k));
        dot = v41_add(dot, v41_mul(v41_mul(h, v41_mul(qw[wi], kw[wi])), k));
    }
    h2 = v41_sum32(h2);
    k2 = v41_sum32(k2);
    dot = v41_mul(v41_sum32(dot), __ocml_rsqrt_f32(v41_add(v41_div(h2, (float)width), eps)));
    dot = v41_mul(dot, __ocml_rsqrt_f32(v41_add(v41_div(k2, (float)width), eps)));
    dot = v41_mul(dot, __ocml_rsqrt_f32((float)width));
    const float gate = v41_div(1.0f, v41_add(1.0f,
        __ocml_exp_f32(-copysignf(__ocml_sqrt_f32(fmaxf(fabsf(dot), 1.0e-6f)), dot))));
    for (uint32_t i = lane; i < width; i += 32u)
        residual[offset + i] = v41_bf16(v41_add(residual[offset + i], v41_mul(gate, v41_bf16(kv[value + i]))));
}

extern "C" int ds4_gpu_dsv41_engram_add(ds4_gpu_tensor *residual, const ds4_gpu_tensor *kv,
                                        const ds4_gpu_tensor *q_weight, const ds4_gpu_tensor *k_weight,
                                        const ds4_gpu_tensor *mask, uint32_t width, uint32_t rows, float eps) {
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || !isfinite(eps) || eps <= 0 || count > UINT64_MAX / 5u ||
        !cuda_tensor_has_f32(residual, count * 4u) || !cuda_tensor_has_f32(kv, count * 5u) ||
        !cuda_tensor_has_f32(q_weight, (uint64_t)width * 4u) ||
        !cuda_tensor_has_f32(k_weight, (uint64_t)width * 4u) ||
        (mask && !cuda_tensor_has_bytes(mask, rows))) return 0;
    v41_engram_kernel<<<dim3(rows, 4), 32>>>((float *)residual->ptr, (const float *)kv->ptr,
        (const float *)q_weight->ptr, (const float *)k_weight->ptr,
        mask ? (const uint8_t *)mask->ptr : NULL, width, eps);
    return cuda_ok(cudaGetLastError(), "V4.1 Engram gate");
}

__global__ static void v41_pool_kernel(float *out, const float *kv, const float *scores,
                                      const float *previous_kv, const float *previous_scores,
                                      uint32_t width, uint32_t tail) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;
    const int64_t a = (int64_t)blockIdx.y * 2 - tail;
    const uint64_t b = (uint64_t)(a + 1) * width + col;
    const float ka = a < 0 ? previous_kv[col] : kv[(uint64_t)a * width + col];
    const float sa = a < 0 ? previous_scores[col] : scores[(uint64_t)a * width + col];
    const float sb = scores[b], peak = fmaxf(sa, sb);
    const float ea = __ocml_exp_f32(v41_sub(sa, peak)), eb = __ocml_exp_f32(v41_sub(sb, peak));
    out[(uint64_t)blockIdx.y * width + col] = v41_bf16(v41_div(
        v41_add(v41_mul(ka, ea), v41_mul(kv[b], eb)), v41_add(ea, eb)));
}

extern "C" int ds4_gpu_dsv41_pool2(ds4_gpu_tensor *out, const ds4_gpu_tensor *kv,
                                   const ds4_gpu_tensor *scores, ds4_gpu_tensor *previous_kv,
                                   ds4_gpu_tensor *previous_scores, uint32_t width, uint32_t rows, uint32_t start) {
    const uint64_t count = (uint64_t)width * rows;
    const uint32_t pairs = (uint32_t)(((uint64_t)rows + (start & 1u)) / 2u);
    if (!width || !rows || rows > UINT32_MAX - start ||
        !cuda_tensor_has_f32(kv, count) || !cuda_tensor_has_f32(scores, count) ||
        !cuda_tensor_has_f32(previous_kv, width) || !cuda_tensor_has_f32(previous_scores, width) ||
        (pairs && !cuda_tensor_has_f32(out, (uint64_t)width * pairs))) return 0;
    if (pairs) {
        v41_pool_kernel<<<dim3((unsigned)(((uint64_t)width + 255u) / 256u), pairs), 256>>>(
            (float *)out->ptr, (const float *)kv->ptr, (const float *)scores->ptr,
            (const float *)previous_kv->ptr, (const float *)previous_scores->ptr, width, start & 1u);
        if (!cuda_ok(cudaGetLastError(), "V4.1 KV pair pooling")) return 0;
    }
    /* Retain the last even input even at even frontiers, so snapshots are independent of chunk partitioning. */
    const uint32_t last_even = (start + rows - 1u) & ~1u;
    if (last_even >= start) {
        const uint64_t bytes = (uint64_t)width * sizeof(float), offset = (last_even - start) * bytes;
        if (!ds4_gpu_tensor_copy(previous_kv, 0, kv, offset, bytes) ||
            !ds4_gpu_tensor_copy(previous_scores, 0, scores, offset, bytes)) return 0;
    }
    return 1;
}

template <bool FILTER>
__global__ static void v41_candidates_kernel(float *out, const float *scores, const float *mask,
                                            uint32_t width, uint32_t start, uint32_t ratio) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    const uint32_t blocks = (width + 7u) / 8u;
    const uint32_t visible = min(width, (start + row + 1u) / ratio);
    if (FILTER) {
        if (col >= width) return;
        const uint64_t i = (uint64_t)row * width + col;
        out[i] = col < visible && mask[(uint64_t)row * blocks + col / 8u] == 0.0f ? scores[i] : -INFINITY;
    } else {
        if (col >= blocks) return;
        float best = -INFINITY;
        for (uint32_t i = col * 8u; i < min(visible, (col + 1u) * 8u); i++)
            best = fmaxf(best, scores[(uint64_t)row * width + i]);
        if (visible && col == (visible - 1u) / 8u) best = INFINITY;
        out[(uint64_t)row * blocks + col] = best;
    }
}

static int v41_candidates(ds4_gpu_tensor *out, const ds4_gpu_tensor *scores,
                           const ds4_gpu_tensor *mask, uint32_t width, uint32_t rows,
                           uint32_t start, uint32_t ratio) {
    if (!width || width > UINT32_MAX - 7u || !rows || !ratio || rows > UINT32_MAX - start) return 0;
    const uint32_t blocks = (width + 7u) / 8u, out_width = mask ? width : blocks;
    if (!cuda_tensor_has_elems2(scores, width, rows, 4u) ||
        !cuda_tensor_has_elems2(out, out_width, rows, 4u) ||
        (mask && !cuda_tensor_has_elems2(mask, blocks, rows, 4u))) return 0;
    const dim3 grid((unsigned)(((uint64_t)out_width + 255u) / 256u), rows);
    if (mask) v41_candidates_kernel<true><<<grid, 256>>>((float *)out->ptr, (const float *)scores->ptr,
        (const float *)mask->ptr, width, start, ratio);
    else v41_candidates_kernel<false><<<grid, 256>>>((float *)out->ptr, (const float *)scores->ptr,
        NULL, width, start, ratio);
    return cuda_ok(cudaGetLastError(), "V4.1 candidate selection");
}

extern "C" int ds4_gpu_dsv41_candidate_blocks(ds4_gpu_tensor *blocks, const ds4_gpu_tensor *scores,
                                              uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    return v41_candidates(blocks, scores, NULL, width, rows, start, ratio);
}

extern "C" int ds4_gpu_dsv41_candidate_filter(ds4_gpu_tensor *scores, const ds4_gpu_tensor *block_mask,
                                              uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    return block_mask && v41_candidates(scores, scores, block_mask, width, rows, start, ratio);
}

__global__ static void v41_carry_bf16_kernel(uint16_t *packed, float *plain, uint32_t width,
                                           uint32_t words, bool pack) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;
    const uint64_t p = (uint64_t)blockIdx.y * words * 2u + col;
    const uint64_t f = (uint64_t)blockIdx.y * width + col;
    if (pack) packed[p] = (uint16_t)(__float_as_uint(plain[f]) >> 16u);
    else plain[f] = __uint_as_float((uint32_t)packed[p] << 16u);
}

__global__ static void v41_carry_mask_kernel(uint32_t *packed, float *plain, uint32_t width,
                                           uint32_t words, bool pack) {
    const uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
    if (word >= words) return;
    const uint64_t p = (uint64_t)blockIdx.y * words + word;
    uint32_t bits = pack ? 0u : packed[p];
    for (uint32_t bit = 0; bit < 32u && (uint64_t)word * 32u + bit < width; bit++) {
        const uint64_t f = (uint64_t)blockIdx.y * width + (uint64_t)word * 32u + bit;
        if (pack) bits |= plain[f] == 0.0f ? 1u << bit : 0u;
        else plain[f] = bits & (1u << bit) ? 0.0f : -INFINITY;
    }
    if (pack) packed[p] = bits;
}

extern "C" int ds4_gpu_dsv41_carry_copy(ds4_gpu_tensor *packed, uint32_t row_offset,
                                        ds4_gpu_tensor *plain, uint32_t width, uint32_t rows,
                                        uint32_t format, bool pack) {
    if (!width || !rows || rows > UINT32_MAX - row_offset ||
        format > DS4_V41_CARRY_MASK || packed == plain) return 0;
    const uint32_t words = format == DS4_V41_CARRY_BF16 ?
        (uint32_t)(((uint64_t)width + 1u) / 2u) : (uint32_t)(((uint64_t)width + 31u) / 32u);
    if (!cuda_tensor_has_elems2(packed, (uint64_t)row_offset + rows, words, 4u) ||
        !cuda_tensor_has_elems2(plain, rows, width, 4u)) return 0;
    uint32_t *p = (uint32_t *)packed->ptr + (uint64_t)row_offset * words;
    if (format == DS4_V41_CARRY_BF16)
        v41_carry_bf16_kernel<<<dim3((unsigned)(((uint64_t)width + 255u) / 256u), rows), 256>>>(
            (uint16_t *)p, (float *)plain->ptr, width, words, pack);
    else v41_carry_mask_kernel<<<dim3((unsigned)(((uint64_t)words + 255u) / 256u), rows), 256>>>(
        p, (float *)plain->ptr, width, words, pack);
    return cuda_ok(cudaGetLastError(), "V4.1 compact prefill carry");
}

__global__ static void v41_gather_kernel(float *out, const float *source, const int32_t *ids,
                                        uint32_t source_rows) {
    const uint32_t row = blockIdx.x, col = threadIdx.x;
    const int32_t id = ids[row];
    /* Graph IDs come from top-k. Keep malformed IDs from turning a validation failure into an out-of-bounds load. */
    if ((uint32_t)id >= source_rows) {
        out[(uint64_t)row * 512u + col] = NAN;
        out[(uint64_t)row * 512u + col + 256u] = NAN;
        return;
    }
    out[(uint64_t)row * 512u + col] = source[(uint64_t)id * 512u + col];
    out[(uint64_t)row * 512u + col + 256u] = source[(uint64_t)id * 512u + col + 256u];
}

extern "C" int ds4_gpu_dsv41_gather_kv(ds4_gpu_tensor *out, const ds4_gpu_tensor *source,
                                       const ds4_gpu_tensor *ids, uint32_t source_rows, uint32_t selected_rows) {
    if (!source_rows || !selected_rows || selected_rows > 512u || selected_rows > source_rows ||
        !cuda_tensor_has_elems2(source, source_rows, 512u, 4u) ||
        !cuda_tensor_has_elems2(out, selected_rows, 512u, 4u) || !cuda_tensor_has_f32(ids, selected_rows)) return 0;
    v41_gather_kernel<<<selected_rows, 256>>>((float *)out->ptr, (const float *)source->ptr,
        (const int32_t *)ids->ptr, source_rows);
    return cuda_ok(cudaGetLastError(), "V4.1 sparse KV gather");
}

__global__ static void v41_indexer_kernel(float *scores, const float *q, const float *weights,
                                         const float *keys, uint32_t width, uint32_t start, uint32_t ratio) {
    const uint32_t key = blockIdx.x, token = blockIdx.y, lane = threadIdx.x & 31u, wave = threadIdx.x >> 5u;
    if (key >= (start + token + 1u) / ratio) {
        if (!threadIdx.x) scores[(uint64_t)token * width + key] = -INFINITY;
        return;
    }
    __shared__ float head_values[4];
    float total = 0.0f;
    for (uint32_t head0 = 0; head0 < 32u; head0 += 4u) {
        const uint32_t head = head0 + wave;
        const float *query = q + ((uint64_t)token * 32u + head) * 128u;
        const float *kv = keys + (uint64_t)key * 128u;
        float dot = 0.0f;
        for (uint32_t col = lane; col < 128u; col += 32u) dot = v41_add(dot, v41_mul(query[col], kv[col]));
        dot = v41_sum32(dot);
        if (!lane) head_values[wave] = v41_mul(fmaxf(v41_mul(dot, 1.0f / 64.0f), 0.0f), weights[(uint64_t)token * 32u + head]);
        __syncthreads();
        if (!threadIdx.x) for (uint32_t h = 0; h < 4u; h++) total = v41_add(total, head_values[h]);
        __syncthreads();
    }
    if (!threadIdx.x) scores[(uint64_t)token * width + key] = total;
}

extern "C" int ds4_gpu_dsv41_indexer_scores_batch(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
                                                  const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys,
                                                  uint32_t source_rows, uint32_t rows, uint32_t start, uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !source_rows || !rows || rows > UINT32_MAX - start ||
        (start + rows) / ratio > source_rows || source_rows > INT32_MAX || rows > INT32_MAX ||
        !cuda_tensor_has_elems2(scores, source_rows, rows, 4u) ||
        !cuda_tensor_has_elems2(q, rows, 32u * 128u, 4u) ||
        !cuda_tensor_has_elems2(keys, source_rows, 128u, 4u) ||
        !cuda_tensor_has_elems2(weights, rows, 32u, 4u)) return 0;
    v41_indexer_kernel<<<dim3(source_rows, rows), 128>>>((float *)scores->ptr, (const float *)q->ptr,
        (const float *)weights->ptr, (const float *)keys->ptr, source_rows, start, ratio);
    return cuda_ok(cudaGetLastError(), "V4.1 causal FP4 index scores");
}

extern "C" int ds4_gpu_dsv41_indexer_topk_batch(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores,
                                               uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !rows || rows > UINT32_MAX - start ||
        width > INT32_MAX || rows > INT32_MAX || (start + rows) / ratio > width ||
        !cuda_tensor_has_elems2(scores, width, rows, 4u) ||
        !cuda_tensor_has_elems2(selected, 512u, rows, 4u)) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        const uint32_t visible = (start + row + 1u) / ratio;
        if (!visible) continue;
        const uint32_t top = visible < 512u ? visible : 512u;
        ds4_gpu_tensor in = {(float *)scores->ptr + (uint64_t)row * width, (uint64_t)visible * 4u, 0};
        ds4_gpu_tensor out = {(uint32_t *)selected->ptr + (uint64_t)row * 512u, 512u * 4u, 0};
        if (visible > 1u && visible < 512u && ds4_rocm_is_gfx1151()) {
            /* Preserve the scalar order and untouched tail slots, while
             * avoiding its serial insertion sort for short prefill rows. */
            indexer_topk_1024_kernel<<<1u, 1024u>>>((uint32_t *)out.ptr,
                (const float *)in.ptr, visible, 1u, top);
            if (!cuda_ok(cudaGetLastError(), "V4.1 short causal top-k")) return 0;
        } else if (!ds4_gpu_indexer_topk_tensor(&out, &in, visible, 1u, top)) return 0;
    }
    return 1;
}

/* Metal tensor packing is an optional acceleration. The graph selects the complete F32 baseline above when this capability is absent. */
extern "C" int ds4_gpu_dsv41_tensor_ops_available(void) { return 0; }

extern "C" uint64_t ds4_gpu_dsv41_indexer_packed_bytes(uint32_t source_rows, uint32_t rows) {
    const uint64_t tiles = ((uint64_t)source_rows + 63u) / 64u;
    const uint64_t flags = (((uint64_t)rows + tiles) * 4u + 255u) & ~UINT64_C(255);
    return flags + (uint64_t)rows * 32u * 128u * 2u + tiles * 64u * 128u * 2u;
}

extern "C" int ds4_gpu_dsv41_indexer_pack(ds4_gpu_tensor *packed, const ds4_gpu_tensor *q,
                                         const ds4_gpu_tensor *keys, uint32_t source_rows, uint32_t rows) {
    (void)packed; (void)q; (void)keys; (void)source_rows; (void)rows;
    return 0;
}

extern "C" int ds4_gpu_dsv41_indexer_scores_packed(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
                                                   const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys,
                                                   const ds4_gpu_tensor *packed, uint32_t source_rows,
                                                   uint32_t rows, uint32_t start, uint32_t ratio,
                                                   uint32_t packed_rows, uint32_t offset) {
    (void)scores; (void)q; (void)weights; (void)keys; (void)packed;
    (void)source_rows; (void)rows; (void)start; (void)ratio; (void)packed_rows; (void)offset;
    return 0;
}

/* Each wave still owns one output row. Lane l sums the same contiguous
 * ceil(K/32) inputs in ascending order. Tokens have separate accumulators;
 * only the exactly decoded F16 weight is reused. Lane zero still adds the
 * 32 partials in order. No activation conversion or split-K reduction. */
template<unsigned TT, unsigned WAVES>
__global__ static void f16_ordered_token_reuse(
        float *out, const __half *w, const float *x,
        uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint64_t row = (uint64_t)blockIdx.x * WAVES + wave;
    const uint64_t token = (uint64_t)blockIdx.y * TT;
    __shared__ float partial[WAVES][TT][32];
    float sum[TT] = {};
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)lane * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    if (row < out_dim) {
        const __half *wr = w + row * in_dim;
        for (uint64_t i = k0; i < k1; i++) {
            const float weight = __half2float(wr[i]);
#pragma unroll
            for (unsigned t = 0; t < TT; t++) {
                if (token + t < n_tok)
                    sum[t] += weight * x[(token + t) * in_dim + i];
            }
        }
    }
#pragma unroll
    for (unsigned t = 0; t < TT; t++) partial[wave][t][lane] = sum[t];
    __syncthreads();
    if (row < out_dim && lane == 0u) {
#pragma unroll
        for (unsigned t = 0; t < TT; t++) {
            if (token + t < n_tok) {
                float total = 0.0f;
                for (uint32_t i = 0; i < 32u; i++) total += partial[wave][t][i];
                out[(token + t) * out_dim + row] = total;
            }
        }
    }
}

__global__ static void v41_hc_half_to_float(float *out, const __half *weight, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __half2float(weight[i]);
}
static hipError_t v41_hc_widen(float *out, const uint16_t *weight, uint64_t count) {
    v41_hc_half_to_float<<<(count + 255u) / 256u, 256u>>>(out, (const __half *)weight, count);
    return hipGetLastError();
}
static bool v41_hc_disjoint(const void *a, uint64_t na, const void *b, uint64_t nb) {
    const uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    return na <= UINTPTR_MAX - pa && nb <= UINTPTR_MAX - pb &&
        (pa + na <= pb || pb + nb <= pa);
}
#include "ds4_rocm_hc_sgemm.cuh"

extern "C" void ds4_gpu_dsv41_hc_plan_free(ds4_gpu_dsv41_hc_plan *plan) {
    v41_hc_plan_destroy(plan);
}
extern "C" int ds4_gpu_dsv41_hc_project(ds4_gpu_dsv41_hc_plan **plan,
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t rows, const ds4_gpu_tensor *input,
        ds4_gpu_tensor *full_heads_scratch) {
    if (rows != 2048u || !ds4_rocm_is_gfx1151() ||
        g_quality_mode || cuda_runtime_config()->graph_dump || !g_rocblas_ready ||
        g_rocblas_f16_solution_set != DS4_ROCBLAS_F16_SOLUTIONS_5_6_8D1AE90E) return 0;
    const uint64_t weight_bytes = UINT64_C(20480) * 24u * 2u;
    const uint64_t in_bytes = UINT64_C(20480) * 2048u * 4u;
    const uint64_t out_bytes = UINT64_C(24) * 2048u * 4u;
    if (!plan || !model_map || !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_bytes(input, in_bytes) || !cuda_tensor_has_bytes(out, out_bytes)) return -1;
    if (!cuda_tensor_has_bytes(full_heads_scratch, v41_hc_scratch_bytes)) return 0;
    if ((uintptr_t)full_heads_scratch->ptr % 256u ||
        !v41_hc_disjoint(out->ptr, out_bytes, input->ptr, in_bytes) ||
        !v41_hc_disjoint(out->ptr, out_bytes, full_heads_scratch->ptr, v41_hc_scratch_bytes) ||
        !v41_hc_disjoint(input->ptr, in_bytes, full_heads_scratch->ptr, v41_hc_scratch_bytes)) return -1;
    const uint16_t *weight = (const uint16_t *)cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "V4.1 HC F16");
    if (!weight ||
        !v41_hc_disjoint(weight, weight_bytes, out->ptr, out_bytes) ||
        !v41_hc_disjoint(weight, weight_bytes, input->ptr, in_bytes) ||
        !v41_hc_disjoint(weight, weight_bytes, full_heads_scratch->ptr, v41_hc_scratch_bytes)) return -1;
    return v41_hc_plan_run(plan, (float *)out->ptr, weight, (const float *)input->ptr, full_heads_scratch->ptr);
}

/* Default Engram lane l consumes l+32*i, i=0..191, then the existing
 * shuffle16/8/4/2/1. Each token retains an independent accumulation chain.
 * Only F32 input reuse across the eight output waves changes. */
template<unsigned TT>
__global__ static void engram_lds_token_reuse(float *out,const __half *w,const float *x) {
    constexpr unsigned K=6144,N=25600,WAVES=8,SEG=256;
    const unsigned tid=threadIdx.x,lane=tid&31u,wave=tid>>5u;
    const unsigned row=blockIdx.x*WAVES+wave,token=blockIdx.y*TT;
    __shared__ float tile[TT][SEG];
    float acc[TT]={};
    for(unsigned base=0;base<K;base+=SEG) {
        for(unsigned j=tid;j<TT*SEG;j+=256u) {
            const unsigned t=j/SEG,k=j%SEG;
            tile[t][k]=x[(uint64_t)(token+t)*K+base+k];
        }
        __syncthreads();
#pragma unroll
        for(unsigned step=0;step<8;step++) {
            const unsigned k=step*32u+lane;
            const float weight=__half2float(w[(uint64_t)row*K+base+k]);
#pragma unroll
            for(unsigned t=0;t<TT;t++)acc[t]+=weight*tile[t][k];
        }
        // Every reader finishes before any thread overwrites the next segment.
        __syncthreads();
    }
#pragma unroll
    for(unsigned t=0;t<TT;t++) {
        const float total=warp_sum_f32(acc[t]);
        if(lane==0)out[(uint64_t)(token+t)*N+row]=total;
    }
}

extern "C" int ds4_gpu_dsv41_projection_rows(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                             uint64_t weight_offset, uint32_t width, uint32_t outputs,
                                             uint32_t rows, const ds4_gpu_tensor *in) {
    uint64_t weight_bytes = 0;
    if (!width || !outputs || !rows || rows > 8192u || !model_map ||
        !cuda_u64_mul3_checked(width, outputs, sizeof(uint16_t), &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_elems2(in, width, rows, 4u) || !cuda_tensor_has_elems2(out, outputs, rows, 4u)) return 0;
    if (width == 6144u && outputs == 25600u && rows == 2048u &&
        ds4_rocm_is_gfx1151() && !g_quality_mode && !cuda_runtime_config()->graph_dump) {
        const __half *w = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "V4.1 exact Engram F16");
        if (!w) return 0;
        engram_lds_token_reuse<16><<<dim3(3200u, 128u), 256u>>>(
            (float *)out->ptr, w, (const float *)in->ptr);
        return cuda_ok(cudaGetLastError(), "V4.1 exact Engram F32-input projection");
    }
    if (width == 20480u && outputs == 24u && rows >= 8u && rows <= 2048u &&
        ds4_rocm_is_gfx1151()) {
        const __half *w = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "f16");
        if (!w) return 0;
        /* Keep four-token tiles for smaller batches; the measured 384-row
         * workload and full 2048-row tiles favor eight-token reuse. */
        if (rows < 384u) {
            f16_ordered_token_reuse<4,8><<<dim3(3u, (rows + 3u) / 4u), 256u>>>(
                (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
        } else {
            f16_ordered_token_reuse<8,8><<<dim3(3u, (rows + 7u) / 8u), 256u>>>(
                (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
        }
        return cuda_ok(cudaGetLastError(), "V4.1 F32-input HC projection");
    }
    /* The general batched F16 API casts inputs to F16. Row views preserve decode arithmetic and retain F32 activations. */
    for (uint32_t row = 0; row < rows; row++) {
        ds4_gpu_tensor x = {(float *)in->ptr + (uint64_t)row * width, (uint64_t)width * 4u, 0};
        ds4_gpu_tensor y = {(float *)out->ptr + (uint64_t)row * outputs, (uint64_t)outputs * 4u, 0};
        if (!ds4_gpu_matmul_f16_tensor(&y, model_map, model_size, weight_offset, width, outputs, &x, 1u)) return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_hc_rms_scale_project_f16_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *scale_scratch,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x, uint32_t n_rows, float eps) {
    if (!in_dim || !out_dim || !n_rows || !isfinite(eps) || eps <= 0.0f) return 0;
    return ds4_gpu_rms_norm_plain_rows_tensor(scale_scratch, x, in_dim, n_rows, eps) &&
        ds4_gpu_dsv41_projection_rows(out, model_map, model_size, weight_offset, in_dim, out_dim, n_rows, scale_scratch);
}

extern "C" int ds4_gpu_dsv41_q8_projection_rows(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                                uint64_t weight_offset, uint32_t width, uint32_t outputs,
                                                uint32_t rows, const ds4_gpu_tensor *in) {
    uint64_t weight_bytes = 0;
    if (!width || width % 32u || !outputs || !rows || rows > 8192u || !model_map ||
        !cuda_u64_mul3_checked(width / 32u, outputs, 34u, &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_elems2(in, width, rows, 4u) || !cuda_tensor_has_elems2(out, outputs, rows, 4u)) return 0;
    const unsigned char *weights = (const unsigned char *)cuda_model_range_ptr(
        model_map, weight_offset, weight_bytes, "V4.1 Q8 projection");
    if (!weights) return 0;
    if (rows == 1u && width == 1280u && outputs == 32768u && ds4_rocm_is_gfx1151()) {
        /* Share the query-B activation row across 32 output waves; preserve
         * the existing per-lane Q8 accumulation and wave reduction. */
        matmul_q8_0_f32_sharedx_warp_rows_w32_kernel<<<1024u, 1024u, 1280u * sizeof(float)>>>(
            (float *)out->ptr, weights, (const float *)in->ptr,
            40u, 32768u, UINT64_C(40) * 34u);
    } else if (!g_quality_mode && width == 1280u && outputs == 32768u && rows >= 32u && rows <= 2048u && ds4_rocm_is_gfx1151()) {
        /* Use the existing generic bulk matrix kernel on bulk prefill rows.
         * This numerical path rounds activations and decoded Q8 weights to
         * F16 before F32 accumulation; quality mode retains the F32 path. */
        matmul_q8_0_f32_batch_wmma_rowtile_kernel<256u, 16u><<<dim3(128u, (rows + 63u) / 64u), 512u>>>(
            (float *)out->ptr, weights, (const float *)in->ptr,
            rows, width, outputs, UINT64_C(40) * 34u);
    } else if (!g_quality_mode && rows == 2048u && ds4_rocm_is_gfx1151() &&
               ((width == 5120u && (outputs == 512u || outputs == 1280u || outputs == 2304u)) ||
                (width == 2304u && outputs == 5120u))) {
        /* Query-A, KV and shared-expert projections on a complete prefill tile.
         * Reuse the generic Q8-to-F16 WMMA path and its F32 accumulation. */
        matmul_q8_0_f32_batch_wmma_rowtile_kernel<128u, 8u><<<dim3(outputs / 128u, 32u), 256u>>>(
            (float *)out->ptr, weights, (const float *)in->ptr,
            rows, width, outputs, (uint64_t)(width / 32u) * 34u);
    } else if (width == 1280u && outputs == 32768u && rows >= 32u && rows <= 2048u && ds4_rocm_is_gfx1151()) {
        /* Reuse sixteen query-B activation rows with the same F32 lane
         * accumulation and wave reduction; keep the existing block tile. */
        cuda_launch_q8_batch_sharedx((float *)out->ptr, weights, (const float *)in->ptr,
            width / 32u, outputs, rows, (uint64_t)(width / 32u) * 34u, 8u, 16u, 8u);
    } else if (rows >= 32u && ds4_rocm_is_gfx1151()) {
        /* Reuse eight F32 activation rows without changing each lane's block
         * accumulation or wave reduction. No F16 cast or expanded weights. */
        cuda_launch_q8_batch_sharedx((float *)out->ptr, weights, (const float *)in->ptr,
            width / 32u, outputs, rows, (uint64_t)(width / 32u) * 34u, 8u, 8u, 8u);
    } else {
        matmul_q8_0_f32_batch_warp8_kernel<<<dim3((unsigned)(((uint64_t)outputs + 7u) / 8u), rows), 256>>>(
            (float *)out->ptr, weights, (const float *)in->ptr, width, outputs, rows, width / 32u);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 F32-input Q8 projection");
}

/* V4.1 grouped output-A: retain physical token strides while using the
 * existing F16-operand/F32-accumulator WMMA body on bulk prefill rows. */
template <uint32_t M_TILE, uint32_t WARPS>
__launch_bounds__(WARPS * 32u, 1)
__global__ static void v41_grouped_q8_f32_wmma_rowtile_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_tokens,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes) {
    const uint32_t group = (uint32_t)blockIdx.z;
    w += (uint64_t)group * out_dim * row_bytes;
    x += (uint64_t)group * in_dim;
    out += (uint64_t)group * out_dim;
    constexpr uint32_t N_TILE = 64u;
    constexpr uint32_t K_TILE = 32u;
    constexpr uint32_t M_PER_WARP = M_TILE / WARPS;
    constexpr uint32_t N_TILES_PER_WARP = N_TILE / 16u;

    const uint32_t block_m = (uint32_t)blockIdx.x * M_TILE;
    const uint32_t block_n = (uint32_t)blockIdx.y * N_TILE;
    if (block_m >= out_dim || block_n >= n_tokens) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t warp_m = block_m + warp_id * M_PER_WARP;
    const uint32_t my_row = warp_m + lane16;
    const uint32_t safe_row = my_row < out_dim ? my_row : (out_dim - 1u);
    const unsigned char *row_base = w + (uint64_t)safe_row * row_bytes;
    const uint32_t n_blocks = in_dim >> 5u;

    ds4_q8_float8_t acc0 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ds4_q8_float8_t acc1 = acc0;
    ds4_q8_float8_t acc2 = acc0;
    ds4_q8_float8_t acc3 = acc0;

    __shared__ _Float16 lds_x[N_TILE * K_TILE];

    for (uint32_t bi = 0; bi < n_blocks; bi++) {
        for (uint32_t j = tid * 2u; j < N_TILE * K_TILE; j += blockDim.x * 2u) {
            const uint32_t nt = j >> 5u;
            const uint32_t kk = j & 31u;
            const uint32_t tok = block_n + nt;
            half2 xv = __floats2half2_rn(0.0f, 0.0f);
            if (tok < n_tokens) {
                const float2 f = *(const float2 *)(x + (uint64_t)tok * 32768u + bi * 32u + kk);
                xv = __floats2half2_rn(f.x, f.y);
            }
            *(half2 *)(lds_x + j) = xv;
        }
        __syncthreads();

        const unsigned char *bp = row_base + (uint64_t)bi * 34u;
        _Float16 sc;
        {
            uint16_t s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        const int8_t *w0 = (const int8_t *)(bp + 2u);
        const int8_t *w1 = (const int8_t *)(bp + 18u);
        ds4_q8_half16_t a0;
        ds4_q8_half16_t a1;
#pragma unroll
        for (uint32_t i = 0; i < 16u; i++) {
            a0[i] = sc * (_Float16)(float)(int)w0[i];
            a1[i] = sc * (_Float16)(float)(int)w1[i];
        }

#pragma unroll
        for (uint32_t ntile = 0; ntile < N_TILES_PER_WARP; ntile++) {
            const uint32_t nt = ntile * 16u + lane16;
            const _Float16 *xb = lds_x + nt * K_TILE;
            const ds4_q8_half16_t b0 = *(const ds4_q8_half16_t *)(xb);
            const ds4_q8_half16_t b1 = *(const ds4_q8_half16_t *)(xb + 16u);
            if (ntile == 0u) {
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc0);
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc0);
            } else if (ntile == 1u) {
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc1);
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc1);
            } else if (ntile == 2u) {
                acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc2);
                acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc2);
            } else {
                acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc3);
                acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc3);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t ntile = 0; ntile < N_TILES_PER_WARP; ntile++) {
        const uint32_t tok = block_n + ntile * 16u + lane16;
        if (tok >= n_tokens) continue;
        ds4_q8_float8_t acc = ntile == 0u ? acc0 : (ntile == 1u ? acc1 : (ntile == 2u ? acc2 : acc3));
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t row = warp_m + 2u * j + (lane >> 4u);
            if (row < out_dim) out[(uint64_t)tok * 8192u + row] = acc[j];
        }
    }
}

extern "C" int ds4_gpu_dsv41_attention_output_batch(ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    const uint64_t a_bytes = UINT64_C(8192) * 128u * 34u, b_bytes = UINT64_C(5120) * 256u * 34u;
    if (!model_map || !n_tokens || !cuda_model_range_fits(model_size, out_a_offset, a_bytes) ||
        !cuda_model_range_fits(model_size, out_b_offset, b_bytes) ||
        !cuda_tensor_has_elems2(heads, n_tokens, 32768u, 4u) ||
        !cuda_tensor_has_elems2(low, n_tokens, 8192u, 4u) || !cuda_tensor_has_elems2(out, n_tokens, 5120u, 4u)) return 0;
    const unsigned char *a = (const unsigned char *)cuda_model_range_ptr(model_map, out_a_offset, a_bytes, "V4.1 attn_out_a");
    const unsigned char *b = (const unsigned char *)cuda_model_range_ptr(model_map, out_b_offset, b_bytes, "V4.1 attn_out_b");
    if (!a || !b) return 0;
    if (!g_quality_mode && n_tokens >= 32u && n_tokens <= 2048u && ds4_rocm_is_gfx1151()) {
        /* Canonical eight groups of 4096 -> 1024, with physical F32 token
         * strides32768/8192. Keep the explicit BF16 low boundary below. */
        v41_grouped_q8_f32_wmma_rowtile_kernel<128u, 8u><<<dim3(8u, (n_tokens + 63u) / 64u, 8u), 256u>>>(
            (float *)low->ptr, a, (const float *)heads->ptr,
            n_tokens, 4096u, 1024u, UINT64_C(128) * 34u);
    } else if (n_tokens >= 32u && ds4_rocm_is_gfx1151()) {
        cuda_launch_grouped_q8_a_sharedx((float *)low->ptr, a, (const float *)heads->ptr,
            n_tokens, 8u, 128u, 1024u, 128u * 34u, 8u, 8u, 8u);
    } else {
        grouped_q8_0_a_f32_batch_warp8_kernel<<<dim3(1024u, n_tokens), 256>>>((float *)low->ptr, a,
            (const float *)heads->ptr, 4096u, 1024u, 8u, n_tokens, 128u);
    }
    if (!cuda_ok(cudaGetLastError(), "V4.1 attention low projection") ||
        !ds4_gpu_dsv41_quantize(low, 8192u, n_tokens, DS4_V41_BF16)) return 0;
    if (n_tokens == 1u && ds4_rocm_is_gfx1151()) {
        /* The shared input is the same BF16-rounded output-A row above. */
        matmul_q8_0_f32_sharedx_warp_rows_w32_kernel<<<160u, 1024u, 8192u * sizeof(float)>>>(
            (float *)out->ptr, b, (const float *)low->ptr,
            256u, 5120u, UINT64_C(256) * 34u);
    } else if (!g_quality_mode && n_tokens >= 32u && n_tokens <= 2048u && ds4_rocm_is_gfx1151()) {
        /* Keep the BF16 boundary above and use the existing generic bulk
         * matrix path: F16-rounded operands with F32 accumulation. */
        matmul_q8_0_f32_batch_wmma_rowtile_kernel<128u, 8u><<<dim3(40u, (n_tokens + 63u) / 64u), 256u>>>(
            (float *)out->ptr, b, (const float *)low->ptr,
            n_tokens, 8192u, 5120u, UINT64_C(256) * 34u);
    } else if (n_tokens >= 32u && n_tokens <= 2048u && ds4_rocm_is_gfx1151()) {
        /* Keep the existing BF16-rounded low rows and reuse sixteen tokens. */
        cuda_launch_q8_batch_sharedx((float *)out->ptr, b, (const float *)low->ptr,
            256u, 5120u, n_tokens, 256u * 34u, 8u, 16u, 8u);
    } else if (n_tokens >= 32u && ds4_rocm_is_gfx1151()) {
        cuda_launch_q8_batch_sharedx((float *)out->ptr, b, (const float *)low->ptr,
            256u, 5120u, n_tokens, 256u * 34u, 8u, 8u, 8u);
    } else {
        matmul_q8_0_f32_batch_warp8_kernel<<<dim3(640u, n_tokens), 256>>>((float *)out->ptr, b,
            (const float *)low->ptr, 8192u, 5120u, n_tokens, 256u);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 attention output projection");
}

extern "C" int ds4_gpu_dsv41_attention_output_tp_batch(ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens, uint32_t tp_rank) {
    (void)out; (void)low; (void)model_map; (void)model_size; (void)out_a_offset; (void)out_b_offset;
    (void)heads; (void)n_tokens; (void)tp_rank;
    return 0;
}

#endif
