// SPDX-License-Identifier: MIT
// test_mmq_parity.cu - parity tests for ds4_mmq_*_dense vs CPU references.
//
// Tests three quant types:
//   - Q8_0:    full F32 -> Q8_0 -> mmq round-trip vs CPU dequant+GEMM
//   - Q2_K:    random Q2_K bytes -> CPU dequant -> reference GEMM
//                                -> mmq GEMM -> compare
//   - IQ2_XXS: random IQ2_XXS bytes -> CPU dequant -> reference GEMM
//                                   -> mmq GEMM -> compare
//
// For Q2_K and IQ2_XXS we don't need a CPU quantizer (those are complex and
// iterative).  Generating random block bytes and dequantizing them produces
// a F32 weight tensor that mmq sees identically - the test exercises the
// FULL kernel path including dequant + tensor-core matmul.
//
// Build:
//   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_120 \
//        -I/path/to/cuda/mmq \
//        test_mmq_parity.cu libds4mmq.a -lcudart -lcublas -lcuda \
//        -o test_mmq_parity

#include "ds4_mmq.h"
#include "iq2_host_tables.h"

// Pull in the block_* struct definitions.  We use the CUDA decl/impl mode
// so the field paths match what the vendored mmq code uses (anonymous
// outer union + named "data" inner struct).  cuda_fp16.h is available
// because nvcc compiles this TU.  Half-precision conversions go via
// __half_raw <-> uint16_t bit patterns, which makes the CPU-side
// fp16<->float helpers below independent of any host-side fp16 ABI.
//
// We DON'T use the host IQ2 lookup tables from this mode (they'd be
// __device__).  iq2_host_tables.h instead provides plain host const
// arrays generated directly from ggml-common.h's bit-for-bit contents.
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "../ggml-common.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int QK_K_LOCAL = 256;

// --------------------------------------------------------------------------
// Half-precision conversion (standalone, no CUDA host fp16 needed).
// --------------------------------------------------------------------------

float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = (h >>  0) & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            while ((mant & 0x400) == 0) { mant <<= 1; exp -= 1; }
            exp += 1; mant &= 0x3ff;
            f = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xff << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

uint16_t float_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 31) & 0x1u;
    int32_t  exp  = ((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    uint16_t h;
    if (exp >= 31) {
        h = (sign << 15) | (0x1f << 10) | (mant ? 0x200 : 0);
    } else if (exp <= 0) {
        if (exp < -10) {
            h = sign << 15;
        } else {
            mant |= 0x800000;
            uint32_t shift = 14 - exp;
            uint32_t r = mant >> shift;
            if (mant & (1u << (shift - 1))) r += 1;
            h = (sign << 15) | r;
        }
    } else {
        if (mant & 0x1000) {
            mant += 0x2000;
            if (mant & 0x800000) { mant = 0; exp += 1; }
        }
        h = (sign << 15) | (exp << 10) | (mant >> 13);
    }
    return h;
}

// --------------------------------------------------------------------------
// Q8_0 quantize + dequant (mirrors ggml's reference).
// --------------------------------------------------------------------------

struct cpu_block_q8_0 {
    uint16_t d;
    int8_t   qs[QK8_0];
};
static_assert(sizeof(cpu_block_q8_0) == 34, "block_q8_0 must be 34 bytes");

void quantize_row_q8_0_cpu(const float * src, cpu_block_q8_0 * dst, int K) {
    const int nb = K / QK8_0;
    for (int b = 0; b < nb; b++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            const float v = std::fabs(src[b * QK8_0 + j]);
            if (v > amax) amax = v;
        }
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        dst[b].d = float_to_fp16(d);
        for (int j = 0; j < QK8_0; j++) {
            const float x = src[b * QK8_0 + j] * id;
            dst[b].qs[j] = (int8_t) std::lround(std::max(-128.f, std::min(127.f, x)));
        }
    }
}

// --------------------------------------------------------------------------
// Q2_K random generator + CPU dequant (ported from ggml-quants.c).
//
// Layout (ggml-common.h:288):
//   uint8_t scales[16]; // packed sc:4 | m:4 per 16-element group
//   uint8_t qs[64];     // 2-bit quants, 4 elements packed per byte
//   half d;             // super-block scale
//   half dmin;          // super-block min
// Total: 84 bytes per 256-value super-block.
// --------------------------------------------------------------------------

// Set the half-precision d / dmin via __half_raw bit-pattern injection.
inline void set_half_from_u16(__half & dst, uint16_t bits) {
    __half_raw r;
    r.x = bits;
    dst = r;
}

inline uint16_t u16_from_half(const __half & h) {
    __half_raw r = h;
    return r.x;
}

void generate_random_block_q2_K(block_q2_K * blk, std::mt19937 & rng) {
    std::uniform_int_distribution<int> u8(0, 255);
    std::uniform_int_distribution<int> u4(0, 15);
    for (int i = 0; i < QK_K_LOCAL/16; i++) {
        blk->scales[i] = (uint8_t)((u4(rng) << 4) | u4(rng));
    }
    for (int i = 0; i < QK_K_LOCAL/4; i++) {
        blk->qs[i] = (uint8_t)u8(rng);
    }
    // d, dmin chosen so the resulting F32 stays in roughly unit variance:
    // q in [0, 3], sc in [0, 15], dl = d*sc -> ~mid ~ 1 => d ~ 0.04.
    std::uniform_real_distribution<float> ud(0.02f, 0.10f);
    set_half_from_u16(blk->data.d,    float_to_fp16(ud(rng)));
    set_half_from_u16(blk->data.dmin, float_to_fp16(ud(rng)));
}

// Port of dequantize_row_q2_K from ggml/src/ggml-quants.c:899.
void dequantize_row_q2_K_cpu(const block_q2_K * x, float * y, int K) {
    const int nb = K / QK_K_LOCAL;
    for (int i = 0; i < nb; i++) {
        const float d   = fp16_to_float(u16_from_half(x[i].data.d));
        const float min = fp16_to_float(u16_from_half(x[i].data.dmin));
        const uint8_t * q = x[i].qs;
        int is = 0;
        for (int n = 0; n < QK_K_LOCAL; n += 128) {
            (void)n;
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                float dl = d * (sc & 0xF);
                float ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                sc = x[i].scales[is++];
                dl = d * (sc & 0xF);
                ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

// --------------------------------------------------------------------------
// IQ2_XXS random generator + CPU dequant (ported from ggml-quants.c).
//
// Layout (ggml-common.h:371):
//   half d;             // super-block scale
//   uint16_t qs[32];    // 32 uint16_t = 256 / 8 = 32 lookup-encoded groups
// Total: 66 bytes per 256-value super-block.
//
// Each sub-block of 32 values consumes 4 uint16_t (= 8 bytes = 2 uint32_t):
// 8 grid indices in the low 32 bits + (signs * 4) | (scale * 1) in the
// high 32 bits.
// --------------------------------------------------------------------------

void generate_random_block_iq2_xxs(block_iq2_xxs * blk, std::mt19937 & rng) {
    std::uniform_int_distribution<int> u16(0, 65535);
    for (int i = 0; i < QK_K_LOCAL/8; i++) {
        blk->qs[i] = (uint16_t)u16(rng);
    }
    std::uniform_real_distribution<float> ud(0.05f, 0.20f);
    set_half_from_u16(blk->d, float_to_fp16(ud(rng)));
}

// --------------------------------------------------------------------------
// Q4_K random generator + CPU dequant.
//
// Layout (ggml-common.h:317):
//   half d, dmin;                 // super-block scales
//   uint8_t scales[K_SCALE_SIZE]; // 12 packed 6-bit scale/min nibbles
//   uint8_t qs[QK_K/2];           // 128 bytes of 4-bit quants (256 quants)
// Total: 144 bytes per 256-value super-block.
// --------------------------------------------------------------------------

void generate_random_block_q4_K(block_q4_K * blk, std::mt19937 & rng) {
    std::uniform_int_distribution<int> u8(0, 255);
    for (int i = 0; i < K_SCALE_SIZE; i++) {
        blk->scales[i] = (uint8_t)u8(rng);
    }
    for (int i = 0; i < QK_K_LOCAL/2; i++) {
        blk->qs[i] = (uint8_t)u8(rng);
    }
    // d/dmin chosen so dequanted values stay near unit variance: per-element
    // value is roughly (d * sc) * q - (dmin * m) where sc <= 63 and q <= 15.
    // Pick d, dmin ~ uniform(0.005, 0.02) so the dominant term peaks at ~12.
    std::uniform_real_distribution<float> ud(0.005f, 0.020f);
    set_half_from_u16(blk->data.d,    float_to_fp16(ud(rng)));
    set_half_from_u16(blk->data.dmin, float_to_fp16(ud(rng)));
}

// 6-bit scale/min extraction. Mirrors get_scale_min_k4 in ggml-quants.c.
static inline void get_scale_min_k4_cpu(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j]     & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Port of dequantize_row_q4_K from ggml/src/ggml-quants.c.
void dequantize_row_q4_K_cpu(const block_q4_K * x, float * y, int K) {
    const int nb = K / QK_K_LOCAL;
    for (int i = 0; i < nb; i++) {
        const float d   = fp16_to_float(u16_from_half(x[i].data.d));
        const float min = fp16_to_float(u16_from_half(x[i].data.dmin));
        const uint8_t * q = x[i].qs;
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K_LOCAL; j += 64) {
            (void)j;
            get_scale_min_k4_cpu(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4_cpu(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

// Port of dequantize_row_iq2_xxs from ggml/src/ggml-quants.c:2412.  The
// CPU-side lookup tables live in iq2_host_tables.h - generated from the
// canonical bit-patterns in cuda/mmq/ggml-common.h.

void dequantize_row_iq2_xxs_cpu(const block_iq2_xxs * x, float * y, int K) {
    const int nb = K / QK_K_LOCAL;
    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_float(u16_from_half(x[i].d));
        for (int ib32 = 0; ib32 < QK_K_LOCAL/32; ++ib32) {
            std::memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2_host::iq2xxs_grid + aux8[l]);
                const uint8_t  signs = iq2_host::ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & iq2_host::kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

// --------------------------------------------------------------------------
// CPU reference matmul: works directly on dequanted F32 weights.
//   W: row-major [M rows, K cols] in F32
//   X: row-major [N rows, K cols] in F32 (K innermost - ggml convention)
//   Y: column-major [M rows, N cols] - Y[col*M + row]  (matches mmq)
// --------------------------------------------------------------------------

void ref_matmul_f32(
        const float * W, const float * X, float * Y,
        int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float acc = 0.0f;
            const float * w_row = W + row * K;
            const float * x_col = X + col * K;
            for (int k = 0; k < K; k++) acc += w_row[k] * x_col[k];
            Y[col * M + row] = acc;
        }
    }
}

// --------------------------------------------------------------------------
// Comparison helper.
// --------------------------------------------------------------------------

bool check_close(const std::vector<float> & got, const std::vector<float> & ref,
                 float abs_tol, float rel_tol, int max_print = 8) {
    int n_bad = 0;
    float worst_abs = 0.0f, worst_rel = 0.0f;
    int worst_i = -1;
    for (size_t i = 0; i < got.size(); i++) {
        const float ag = got[i];
        const float ar = ref[i];
        const float ae = std::fabs(ag - ar);
        const float re = ar != 0.0f ? ae / std::fabs(ar) : (ae > 0 ? INFINITY : 0.0f);
        if (ae > abs_tol && re > rel_tol) {
            if (n_bad < max_print) {
                fprintf(stderr, "  [%zu] got=%.6g ref=%.6g abs=%.3g rel=%.3g\n",
                        i, ag, ar, ae, re);
            }
            n_bad++;
        }
        if (ae > worst_abs) { worst_abs = ae; worst_i = (int)i; }
        if (re > worst_rel) { worst_rel = re; }
    }
    fprintf(stderr, "  worst abs=%.3g  worst rel=%.3g  bad=%d / %zu  (at i=%d)\n",
            worst_abs, worst_rel, n_bad, got.size(), worst_i);
    return n_bad == 0;
}

// --------------------------------------------------------------------------
// Per-type test runners.
// --------------------------------------------------------------------------

bool run_q8_0(int M, int N, int K, uint32_t seed, float abs_scale = 0.05f) {
    fprintf(stderr, "=== Q8_0   M=%d N=%d K=%d  seed=%u ===\n", M, N, K, seed);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> W_f32(M * K);
    for (auto & v : W_f32) v = nd(rng);
    const int nb_per_row = K / QK8_0;
    std::vector<cpu_block_q8_0> W_q8(M * nb_per_row);
    for (int row = 0; row < M; row++) {
        quantize_row_q8_0_cpu(&W_f32[row * K], &W_q8[row * nb_per_row], K);
    }
    // CPU reference uses dequanted weight to match what mmq sees.
    std::vector<float> W_deq(M * K);
    for (int row = 0; row < M; row++) {
        const cpu_block_q8_0 * blk = &W_q8[row * nb_per_row];
        for (int b = 0; b < nb_per_row; b++) {
            const float d = fp16_to_float(blk[b].d);
            for (int j = 0; j < QK8_0; j++) {
                W_deq[row * K + b * QK8_0 + j] = d * blk[b].qs[j];
            }
        }
    }

    std::vector<float> X_f32(K * N);
    for (auto & v : X_f32) v = nd(rng);

    std::vector<float> ref_out(M * N, 0.0f);
    ref_matmul_f32(W_deq.data(), X_f32.data(), ref_out.data(), M, N, K);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void * dW = nullptr; float * dX = nullptr; float * dY = nullptr;
    cudaMalloc(&dW, W_q8.size() * sizeof(cpu_block_q8_0));
    cudaMalloc(&dX, X_f32.size() * sizeof(float));
    cudaMalloc(&dY, M * N * sizeof(float));
    cudaMemcpyAsync(dW, W_q8.data(), W_q8.size() * sizeof(cpu_block_q8_0), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dX, X_f32.data(), X_f32.size() * sizeof(float),       cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(dY, 0, M * N * sizeof(float), stream);
    int rc = ds4_mmq_q8_0_dense(dW, dX, dY, M, N, K, stream);
    if (rc != 0) { fprintf(stderr, "ds4_mmq_q8_0_dense returned %d\n", rc); return false; }
    std::vector<float> got_out(M * N, 0.0f);
    cudaMemcpyAsync(got_out.data(), dY, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(dW); cudaFree(dX); cudaFree(dY); cudaStreamDestroy(stream);

    const float abs_tol = abs_scale * std::sqrt((float)K);
    const bool ok = check_close(got_out, ref_out, abs_tol, 0.05f);
    fprintf(stderr, "%s\n\n", ok ? "PASS" : "FAIL");
    return ok;
}

bool run_q2_K(int M, int N, int K, uint32_t seed, float abs_scale = 0.05f) {
    fprintf(stderr, "=== Q2_K   M=%d N=%d K=%d  seed=%u ===\n", M, N, K, seed);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const int nb_per_row = K / QK_K_LOCAL;
    std::vector<block_q2_K> W_q2(M * nb_per_row);
    for (auto & blk : W_q2) generate_random_block_q2_K(&blk, rng);

    std::vector<float> W_deq(M * K);
    for (int row = 0; row < M; row++) {
        dequantize_row_q2_K_cpu(&W_q2[row * nb_per_row], &W_deq[row * K], K);
    }

    std::vector<float> X_f32(K * N);
    for (auto & v : X_f32) v = nd(rng);

    std::vector<float> ref_out(M * N, 0.0f);
    ref_matmul_f32(W_deq.data(), X_f32.data(), ref_out.data(), M, N, K);

    cudaStream_t stream; cudaStreamCreate(&stream);
    void * dW = nullptr; float * dX = nullptr; float * dY = nullptr;
    cudaMalloc(&dW, W_q2.size() * sizeof(block_q2_K));
    cudaMalloc(&dX, X_f32.size() * sizeof(float));
    cudaMalloc(&dY, M * N * sizeof(float));
    cudaMemcpyAsync(dW, W_q2.data(), W_q2.size() * sizeof(block_q2_K), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dX, X_f32.data(), X_f32.size() * sizeof(float),    cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(dY, 0, M * N * sizeof(float), stream);
    int rc = ds4_mmq_q2_K_dense(dW, dX, dY, M, N, K, stream);
    if (rc != 0) { fprintf(stderr, "ds4_mmq_q2_K_dense returned %d\n", rc); return false; }
    std::vector<float> got_out(M * N, 0.0f);
    cudaMemcpyAsync(got_out.data(), dY, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(dW); cudaFree(dX); cudaFree(dY); cudaStreamDestroy(stream);

    const float abs_tol = abs_scale * std::sqrt((float)K);
    const bool ok = check_close(got_out, ref_out, abs_tol, 0.05f);
    fprintf(stderr, "%s\n\n", ok ? "PASS" : "FAIL");
    return ok;
}

bool run_q4_K(int M, int N, int K, uint32_t seed, float abs_scale = 0.20f) {
    fprintf(stderr, "=== Q4_K   M=%d N=%d K=%d  seed=%u ===\n", M, N, K, seed);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const int nb_per_row = K / QK_K_LOCAL;
    std::vector<block_q4_K> W_q4(M * nb_per_row);
    for (auto & blk : W_q4) generate_random_block_q4_K(&blk, rng);

    std::vector<float> W_deq(M * K);
    for (int row = 0; row < M; row++) {
        dequantize_row_q4_K_cpu(&W_q4[row * nb_per_row], &W_deq[row * K], K);
    }

    std::vector<float> X_f32(K * N);
    for (auto & v : X_f32) v = nd(rng);

    std::vector<float> ref_out(M * N, 0.0f);
    ref_matmul_f32(W_deq.data(), X_f32.data(), ref_out.data(), M, N, K);

    cudaStream_t stream; cudaStreamCreate(&stream);
    void * dW = nullptr; float * dX = nullptr; float * dY = nullptr;
    cudaMalloc(&dW, W_q4.size() * sizeof(block_q4_K));
    cudaMalloc(&dX, X_f32.size() * sizeof(float));
    cudaMalloc(&dY, M * N * sizeof(float));
    cudaMemcpyAsync(dW, W_q4.data(), W_q4.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dX, X_f32.data(), X_f32.size() * sizeof(float),    cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(dY, 0, M * N * sizeof(float), stream);
    int rc = ds4_mmq_q4_K_dense(dW, dX, dY, M, N, K, stream);
    if (rc != 0) { fprintf(stderr, "ds4_mmq_q4_K_dense returned %d\n", rc); return false; }
    std::vector<float> got_out(M * N, 0.0f);
    cudaMemcpyAsync(got_out.data(), dY, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(dW); cudaFree(dX); cudaFree(dY); cudaStreamDestroy(stream);

    const float abs_tol = abs_scale * std::sqrt((float)K);
    const bool ok = check_close(got_out, ref_out, abs_tol, 0.05f);
    fprintf(stderr, "%s\n\n", ok ? "PASS" : "FAIL");
    return ok;
}

// IQ2_XXS internally accumulates in int8 via SIMD intrinsics
// (__vsub4 / __vcmpne4 in vec_dot_iq2_xxs_q8_1) and applies the scale
// post-accumulation, while the CPU reference does per-element float
// multiplies.  The two paths agree to within a few units of grid scale.
// Loosen abs_scale to 0.20*sqrt(K) which covers observed worst-case
// disagreement of ~10.5 at K=4096 (db_max ~ 4 * d_max with d_max ~ 0.2).
bool run_iq2_xxs(int M, int N, int K, uint32_t seed, float abs_scale = 0.20f) {
    fprintf(stderr, "=== IQ2_XXS M=%d N=%d K=%d  seed=%u ===\n", M, N, K, seed);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const int nb_per_row = K / QK_K_LOCAL;
    std::vector<block_iq2_xxs> W_iq2(M * nb_per_row);
    for (auto & blk : W_iq2) generate_random_block_iq2_xxs(&blk, rng);

    std::vector<float> W_deq(M * K);
    for (int row = 0; row < M; row++) {
        dequantize_row_iq2_xxs_cpu(&W_iq2[row * nb_per_row], &W_deq[row * K], K);
    }

    std::vector<float> X_f32(K * N);
    for (auto & v : X_f32) v = nd(rng);

    std::vector<float> ref_out(M * N, 0.0f);
    ref_matmul_f32(W_deq.data(), X_f32.data(), ref_out.data(), M, N, K);

    cudaStream_t stream; cudaStreamCreate(&stream);
    void * dW = nullptr; float * dX = nullptr; float * dY = nullptr;
    cudaMalloc(&dW, W_iq2.size() * sizeof(block_iq2_xxs));
    cudaMalloc(&dX, X_f32.size() * sizeof(float));
    cudaMalloc(&dY, M * N * sizeof(float));
    cudaMemcpyAsync(dW, W_iq2.data(), W_iq2.size() * sizeof(block_iq2_xxs), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dX, X_f32.data(), X_f32.size() * sizeof(float),         cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(dY, 0, M * N * sizeof(float), stream);
    int rc = ds4_mmq_iq2_xxs_dense(dW, dX, dY, M, N, K, stream);
    if (rc != 0) { fprintf(stderr, "ds4_mmq_iq2_xxs_dense returned %d\n", rc); return false; }
    std::vector<float> got_out(M * N, 0.0f);
    cudaMemcpyAsync(got_out.data(), dY, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(dW); cudaFree(dX); cudaFree(dY); cudaStreamDestroy(stream);

    const float abs_tol = abs_scale * std::sqrt((float)K);
    const bool ok = check_close(got_out, ref_out, abs_tol, 0.05f);
    fprintf(stderr, "%s\n\n", ok ? "PASS" : "FAIL");
    return ok;
}

// --------------------------------------------------------------------------
// MoE parity runners.
//
// Generates per-expert weight slabs in the type-specific block format
// stacked contiguously, a routing table [n_tokens, n_expert_used], and an
// F32 activation [n_tokens, K].  Computes the reference by dequantizing
// every expert's weights and running a naive per-(token, slot) GEMM.
// Compares against ds4_mmq_*_moe.
// --------------------------------------------------------------------------

template <typename BlockT, typename DequantFn>
bool run_moe_generic(
        const char * tag, int blck_size,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        uint32_t seed, float abs_scale,
        DequantFn gen_and_dequant,
        int (*entry)(const void *, const float *, const int32_t *, float *,
                     int, int, int, int, int, cudaStream_t)) {
    fprintf(stderr, "=== %s   M=%d K=%d ntok=%d nexp=%d nused=%d  seed=%u ===\n",
            tag, M, K, n_tokens, n_experts, n_expert_used, seed);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const int blocks_per_row    = K / blck_size;
    const int blocks_per_expert = M * blocks_per_row;

    std::vector<BlockT> W_blk((size_t)n_experts * blocks_per_expert);
    std::vector<float>  W_deq((size_t)n_experts * M * K);
    gen_and_dequant(W_blk.data(), W_deq.data(), n_experts, M, K, blocks_per_expert, rng);

    // Routing: distinct experts per token.
    std::vector<int32_t> ids((size_t)n_tokens * n_expert_used);
    std::uniform_int_distribution<int> uexp(0, n_experts - 1);
    for (int t = 0; t < n_tokens; t++) {
        std::vector<int> picked;
        while ((int)picked.size() < n_expert_used) {
            int e = uexp(rng);
            if (std::find(picked.begin(), picked.end(), e) == picked.end()) picked.push_back(e);
        }
        for (int s = 0; s < n_expert_used; s++) ids[t * n_expert_used + s] = picked[s];
    }

    std::vector<float> X((size_t)n_tokens * K);
    for (auto & v : X) v = nd(rng);

    const int64_t ne_get_rows = (int64_t)n_tokens * n_expert_used;
    std::vector<float> ref_out((size_t)M * ne_get_rows, 0.0f);
    for (int t = 0; t < n_tokens; t++) {
        for (int s = 0; s < n_expert_used; s++) {
            const int e   = ids[t * n_expert_used + s];
            const int col = t * n_expert_used + s;
            const float * w_e = &W_deq[(size_t)e * M * K];
            const float * x_t = &X[(size_t)t * K];
            for (int row = 0; row < M; row++) {
                float acc = 0.0f;
                const float * w_row = w_e + (size_t)row * K;
                for (int k = 0; k < K; k++) acc += w_row[k] * x_t[k];
                ref_out[(size_t)col * M + row] = acc;
            }
        }
    }

    cudaStream_t stream; cudaStreamCreate(&stream);
    void    * dW   = nullptr;
    float   * dX   = nullptr;
    int32_t * dIds = nullptr;
    float   * dY   = nullptr;
    cudaMalloc(&dW,   W_blk.size() * sizeof(BlockT));
    cudaMalloc(&dX,   X.size() * sizeof(float));
    cudaMalloc(&dIds, ids.size() * sizeof(int32_t));
    cudaMalloc(&dY,   (size_t)M * ne_get_rows * sizeof(float));
    cudaMemcpyAsync(dW,   W_blk.data(), W_blk.size() * sizeof(BlockT), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dX,   X.data(),     X.size() * sizeof(float),       cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dIds, ids.data(),   ids.size() * sizeof(int32_t),   cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(dY, 0, (size_t)M * ne_get_rows * sizeof(float), stream);

    int rc = entry(dW, dX, dIds, dY, M, K, n_tokens, n_experts, n_expert_used, stream);
    if (rc != 0) { fprintf(stderr, "%s entry returned %d\n", tag, rc);
                   cudaFree(dW); cudaFree(dX); cudaFree(dIds); cudaFree(dY); cudaStreamDestroy(stream);
                   return false; }

    std::vector<float> got_out((size_t)M * ne_get_rows, 0.0f);
    cudaMemcpyAsync(got_out.data(), dY, got_out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(dW); cudaFree(dX); cudaFree(dIds); cudaFree(dY); cudaStreamDestroy(stream);

    const float abs_tol = abs_scale * std::sqrt((float)K);
    const bool ok = check_close(got_out, ref_out, abs_tol, 0.05f);
    fprintf(stderr, "%s\n\n", ok ? "PASS" : "FAIL");
    return ok;
}

bool run_q8_0_moe(int M, int K, int nt, int ne, int nu, uint32_t seed) {
    auto fn = [](cpu_block_q8_0 * blk, float * out,
                 int n_experts, int M, int K, int blocks_per_expert,
                 std::mt19937 & rng) {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        const int blocks_per_row = K / QK8_0;
        for (int e = 0; e < n_experts; e++) {
            cpu_block_q8_0 * eblk = blk + (size_t)e * blocks_per_expert;
            std::vector<float> src((size_t)M * K);
            for (auto & v : src) v = nd(rng);
            for (int row = 0; row < M; row++) {
                quantize_row_q8_0_cpu(&src[(size_t)row * K], &eblk[row * blocks_per_row], K);
                const cpu_block_q8_0 * rowblk = &eblk[row * blocks_per_row];
                float * orow = out + ((size_t)e * M + row) * K;
                for (int b = 0; b < blocks_per_row; b++) {
                    const float d = fp16_to_float(rowblk[b].d);
                    for (int j = 0; j < QK8_0; j++) {
                        orow[b * QK8_0 + j] = d * rowblk[b].qs[j];
                    }
                }
            }
        }
    };
    return run_moe_generic<cpu_block_q8_0>(
        "Q8_0/MOE", QK8_0, M, K, nt, ne, nu, seed, 0.05f, fn, ds4_mmq_q8_0_moe);
}

bool run_q2_K_moe(int M, int K, int nt, int ne, int nu, uint32_t seed) {
    auto fn = [](block_q2_K * blk, float * out,
                 int n_experts, int M, int K, int blocks_per_expert,
                 std::mt19937 & rng) {
        const int blocks_per_row = K / QK_K_LOCAL;
        for (int e = 0; e < n_experts; e++) {
            block_q2_K * eblk = blk + (size_t)e * blocks_per_expert;
            for (int row = 0; row < M; row++) {
                for (int b = 0; b < blocks_per_row; b++) {
                    generate_random_block_q2_K(&eblk[row * blocks_per_row + b], rng);
                }
                dequantize_row_q2_K_cpu(&eblk[row * blocks_per_row],
                                        out + ((size_t)e * M + row) * K, K);
            }
        }
    };
    return run_moe_generic<block_q2_K>(
        "Q2_K/MOE", QK_K_LOCAL, M, K, nt, ne, nu, seed, 0.05f, fn, ds4_mmq_q2_K_moe);
}

bool run_iq2_xxs_moe(int M, int K, int nt, int ne, int nu, uint32_t seed) {
    auto fn = [](block_iq2_xxs * blk, float * out,
                 int n_experts, int M, int K, int blocks_per_expert,
                 std::mt19937 & rng) {
        const int blocks_per_row = K / QK_K_LOCAL;
        for (int e = 0; e < n_experts; e++) {
            block_iq2_xxs * eblk = blk + (size_t)e * blocks_per_expert;
            for (int row = 0; row < M; row++) {
                for (int b = 0; b < blocks_per_row; b++) {
                    generate_random_block_iq2_xxs(&eblk[row * blocks_per_row + b], rng);
                }
                dequantize_row_iq2_xxs_cpu(&eblk[row * blocks_per_row],
                                           out + ((size_t)e * M + row) * K, K);
            }
        }
    };
    return run_moe_generic<block_iq2_xxs>(
        "IQ2_XXS/MOE", QK_K_LOCAL, M, K, nt, ne, nu, seed, 0.20f, fn, ds4_mmq_iq2_xxs_moe);
}

// Pair-API verifier.  Compares ds4_mmq_<type>_moe_pair(W_a, W_b, X, ids)
// against two back-to-back single-W ds4_mmq_<type>_moe(W_a, X, ids) and
// ds4_mmq_<type>_moe(W_b, X, ids) calls.  Both paths share quantize +
// helper internally; the pair API just fuses the two so the shared work
// runs once.  Output should be bit-identical (same kernel, same Q8_1
// buffer; only the second mul_mat_q_case launch is added).
template <typename BlockT, typename DequantFn>
bool run_moe_pair_generic(
        const char * tag, int blck_size,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        uint32_t seed,
        DequantFn gen_and_dequant,
        int (*pair_entry)(const void *, const void *, const float *,
                          const int32_t *, float *, float *,
                          int, int, int, int, int, cudaStream_t),
        int (*single_entry)(const void *, const float *, const int32_t *,
                            float *, int, int, int, int, int, cudaStream_t)) {
    fprintf(stderr, "=== %s/PAIR  M=%d K=%d ntok=%d nexp=%d nused=%d  seed=%u ===\n",
            tag, M, K, n_tokens, n_experts, n_expert_used, seed);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const int blocks_per_row    = K / blck_size;
    const int blocks_per_expert = M * blocks_per_row;

    // Generate two independent weight tensors of the same type+shape.
    std::vector<BlockT> W_a((size_t)n_experts * blocks_per_expert);
    std::vector<BlockT> W_b((size_t)n_experts * blocks_per_expert);
    std::vector<float>  unused((size_t)n_experts * M * K);
    gen_and_dequant(W_a.data(), unused.data(), n_experts, M, K, blocks_per_expert, rng);
    gen_and_dequant(W_b.data(), unused.data(), n_experts, M, K, blocks_per_expert, rng);

    std::vector<int32_t> ids((size_t)n_tokens * n_expert_used);
    std::uniform_int_distribution<int> uexp(0, n_experts - 1);
    for (int t = 0; t < n_tokens; t++) {
        std::vector<int> picked;
        while ((int)picked.size() < n_expert_used) {
            int e = uexp(rng);
            if (std::find(picked.begin(), picked.end(), e) == picked.end()) picked.push_back(e);
        }
        for (int s = 0; s < n_expert_used; s++) ids[t * n_expert_used + s] = picked[s];
    }

    std::vector<float> X((size_t)n_tokens * K);
    for (auto & v : X) v = nd(rng);

    const int64_t ne_get_rows = (int64_t)n_tokens * n_expert_used;
    const size_t  out_count   = (size_t)M * ne_get_rows;

    cudaStream_t stream; cudaStreamCreate(&stream);
    void * dWa = nullptr; void * dWb = nullptr;
    float * dX = nullptr; int32_t * dIds = nullptr;
    float * dYa_single = nullptr; float * dYb_single = nullptr;
    float * dYa_pair = nullptr;   float * dYb_pair = nullptr;
    cudaMalloc(&dWa, W_a.size() * sizeof(BlockT));
    cudaMalloc(&dWb, W_b.size() * sizeof(BlockT));
    cudaMalloc(&dX,  X.size() * sizeof(float));
    cudaMalloc(&dIds, ids.size() * sizeof(int32_t));
    cudaMalloc(&dYa_single, out_count * sizeof(float));
    cudaMalloc(&dYb_single, out_count * sizeof(float));
    cudaMalloc(&dYa_pair,   out_count * sizeof(float));
    cudaMalloc(&dYb_pair,   out_count * sizeof(float));
    cudaMemcpyAsync(dWa, W_a.data(), W_a.size() * sizeof(BlockT), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dWb, W_b.data(), W_b.size() * sizeof(BlockT), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dX,  X.data(),   X.size()  * sizeof(float),   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dIds, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(dYa_single, 0, out_count * sizeof(float), stream);
    cudaMemsetAsync(dYb_single, 0, out_count * sizeof(float), stream);
    cudaMemsetAsync(dYa_pair,   0, out_count * sizeof(float), stream);
    cudaMemsetAsync(dYb_pair,   0, out_count * sizeof(float), stream);

    int rc_sa = single_entry(dWa, dX, dIds, dYa_single, M, K, n_tokens, n_experts, n_expert_used, stream);
    int rc_sb = single_entry(dWb, dX, dIds, dYb_single, M, K, n_tokens, n_experts, n_expert_used, stream);
    int rc_p  = pair_entry  (dWa, dWb, dX, dIds, dYa_pair, dYb_pair,
                             M, K, n_tokens, n_experts, n_expert_used, stream);
    if (rc_sa != 0 || rc_sb != 0 || rc_p != 0) {
        fprintf(stderr, "%s pair entry: rc_sa=%d rc_sb=%d rc_p=%d\n", tag, rc_sa, rc_sb, rc_p);
        cudaFree(dWa); cudaFree(dWb); cudaFree(dX); cudaFree(dIds);
        cudaFree(dYa_single); cudaFree(dYb_single); cudaFree(dYa_pair); cudaFree(dYb_pair);
        cudaStreamDestroy(stream);
        return false;
    }

    std::vector<float> ya_single(out_count, 0.0f), yb_single(out_count, 0.0f);
    std::vector<float> ya_pair  (out_count, 0.0f), yb_pair  (out_count, 0.0f);
    cudaMemcpyAsync(ya_single.data(), dYa_single, out_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(yb_single.data(), dYb_single, out_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(ya_pair.data(),   dYa_pair,   out_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(yb_pair.data(),   dYb_pair,   out_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(dWa); cudaFree(dWb); cudaFree(dX); cudaFree(dIds);
    cudaFree(dYa_single); cudaFree(dYb_single); cudaFree(dYa_pair); cudaFree(dYb_pair);
    cudaStreamDestroy(stream);

    // Both pair outputs should be bit-identical to their single counterparts -
    // same kernel, same Q8_1 buffer.  Allow exactly zero abs/rel tolerance.
    const bool ok_a = check_close(ya_pair, ya_single, 0.0f, 0.0f);
    const bool ok_b = check_close(yb_pair, yb_single, 0.0f, 0.0f);
    const bool ok   = ok_a && ok_b;
    fprintf(stderr, "%s\n\n", ok ? "PASS" : "FAIL");
    return ok;
}

// Reference glue for the raw fused IQ2/Q2 pipeline.  Keep the expression and
// non-finite handling identical to ds4_swiglu_weighted_f32 in ds4_mmq.cu so
// parity below isolates routing-map reuse rather than activation math.
__global__ void test_swiglu_weighted_f32(
        const float * gate, const float * up, const float * router_weights,
        float * mid, uint64_t n, int K, float clamp) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t pair = i / (uint64_t)K;
    float g = isfinite(gate[i]) ? gate[i] : 0.0f;
    float u = isfinite(up[i]) ? up[i] : 0.0f;
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    mid[i] = (g / (1.0f + expf(-g))) * u * router_weights[pair];
}

// Exercise the SSD-facing raw fused ABI with a non-identity compact expert
// table.  The reference is the established materialized chain:
//   raw pair -> weighted SwiGLU -> raw down.
// Every token has six distinct experts.  Flattening ids to [assignments, 1]
// for the reference down leg produces the same stable expert ordering as the
// single routing map reused by the fused candidate.
bool run_iq2_xxs_q2_K_fused_raw_parity(
        int n_tokens, uint32_t seed, bool persistent_q81 = false) {
    constexpr int global_experts = 13;
    constexpr int compact_experts = 8;
    constexpr int n_expert_used = 6;
    // Deliberately asymmetric and multi-block in both legs.  A 256x256
    // fixture degenerates every raw row to one GGUF block and cannot catch a
    // bad row/expert stride in either the IQ2 gate/up or Q2 down tensor.
    constexpr int expert_mid_dim = 512;
    constexpr int expert_in_dim = 768;
    constexpr int out_dim = 768;
    constexpr float clamp = 6.0f;
    const int compact_to_global[compact_experts] = {11, 2, 9, 0, 7, 12, 4, 6};
    int persistent_device = 0;

    if (persistent_q81) {
        cudaDeviceProp prop = {};
        if (cudaGetDevice(&persistent_device) != cudaSuccess ||
            cudaGetDeviceProperties(&prop, persistent_device) != cudaSuccess ||
            !prop.integrated || prop.major != 12 || prop.minor != 1) {
            (void)cudaGetLastError();
            fprintf(stderr,
                    "=== IQ2_XXS+Q2_K/FUSED_RAW persistent Q8_1: "
                    "SKIP (requires integrated sm_121) ===\n\n");
            return true;
        }
    }

    fprintf(stderr,
            "=== IQ2_XXS+Q2_K/FUSED_RAW%s compact-remap ntok=%d "
            "nexp=%d nused=%d seed=%u ===\n",
            persistent_q81 ? "/PERSISTENT_Q81" : "",
            n_tokens, compact_experts, n_expert_used, seed);

    int initial_arena_cleanup = 0;
    int q81_lazy_init_rc = 0;
    uint64_t q81_init_allocations0 = 0, q81_init_resizes0 = 0;
    uint64_t q81_init_allocations1 = 0, q81_init_resizes1 = 0;
    size_t q81_init_arena0 = 0, q81_init_arena1 = 0;
    if (persistent_q81) {
        initial_arena_cleanup = ds4_mmq_q81_persistent_cleanup();
        ds4_mmq_set_gb10_optimizations(1);
        ds4_mmq_q81_persistent_counters(
            nullptr, nullptr, nullptr, nullptr,
            &q81_init_allocations0, &q81_init_resizes0,
            &q81_init_arena0, nullptr);
        setenv("DS4_CUDA_MMQ_Q81_PERSISTENT", "1", 1);
        q81_lazy_init_rc = ds4_mmq_init(persistent_device);
        ds4_mmq_q81_persistent_counters(
            nullptr, nullptr, nullptr, nullptr,
            &q81_init_allocations1, &q81_init_resizes1,
            &q81_init_arena1, nullptr);
        // A real `=0` dispatch below is the value-aware opt-out oracle.
        setenv("DS4_CUDA_MMQ_Q81_PERSISTENT", "0", 1);
    }

    std::mt19937 rng(seed);
    std::normal_distribution<float> activation(0.0f, 0.05f);
    const size_t iq2_blocks_per_expert =
        (size_t)expert_mid_dim * (expert_in_dim / QK_K_LOCAL);
    const size_t q2_blocks_per_expert =
        (size_t)out_dim * (expert_mid_dim / QK_K_LOCAL);

    std::vector<block_iq2_xxs> gate_global(
        (size_t)global_experts * iq2_blocks_per_expert);
    std::vector<block_iq2_xxs> up_global(
        (size_t)global_experts * iq2_blocks_per_expert);
    std::vector<block_q2_K> down_global(
        (size_t)global_experts * q2_blocks_per_expert);
    for (auto & block : gate_global) generate_random_block_iq2_xxs(&block, rng);
    for (auto & block : up_global) generate_random_block_iq2_xxs(&block, rng);
    for (auto & block : down_global) generate_random_block_q2_K(&block, rng);

    std::vector<block_iq2_xxs> gate_compact(
        (size_t)compact_experts * iq2_blocks_per_expert);
    std::vector<block_iq2_xxs> up_compact(
        (size_t)compact_experts * iq2_blocks_per_expert);
    std::vector<block_q2_K> down_compact(
        (size_t)compact_experts * q2_blocks_per_expert);
    int global_to_compact[global_experts];
    std::fill(global_to_compact, global_to_compact + global_experts, -1);
    for (int compact = 0; compact < compact_experts; compact++) {
        const int global = compact_to_global[compact];
        global_to_compact[global] = compact;
        std::memcpy(gate_compact.data() + (size_t)compact * iq2_blocks_per_expert,
                    gate_global.data() + (size_t)global * iq2_blocks_per_expert,
                    iq2_blocks_per_expert * sizeof(block_iq2_xxs));
        std::memcpy(up_compact.data() + (size_t)compact * iq2_blocks_per_expert,
                    up_global.data() + (size_t)global * iq2_blocks_per_expert,
                    iq2_blocks_per_expert * sizeof(block_iq2_xxs));
        std::memcpy(down_compact.data() + (size_t)compact * q2_blocks_per_expert,
                    down_global.data() + (size_t)global * q2_blocks_per_expert,
                    q2_blocks_per_expert * sizeof(block_q2_K));
    }

    const size_t assignments = (size_t)n_tokens * n_expert_used;
    std::vector<int32_t> global_ids(assignments);
    std::vector<int32_t> remapped_ids(assignments);
    std::vector<float> router_weights(assignments);
    for (int token = 0; token < n_tokens; token++) {
        bool seen[compact_experts] = {};
        float router_sum = 0.0f;
        for (int slot = 0; slot < n_expert_used; slot++) {
            // Three is coprime with eight, so the six positions are unique;
            // the global round-trip makes this an explicit compact-remap test.
            const int compact = (token * 5 + slot * 3) % compact_experts;
            const int global = compact_to_global[compact];
            if (seen[compact] || global_to_compact[global] != compact) {
                fprintf(stderr, "invalid compact routing fixture\n");
                return false;
            }
            seen[compact] = true;
            const size_t pair = (size_t)token * n_expert_used + slot;
            global_ids[pair] = global;
            remapped_ids[pair] = global_to_compact[global];
            // Token- and slot-varying values make a pair-stride bug visible;
            // normalize per token to retain the production router contract.
            const float raw_weight =
                (float)(1 + ((token * 11 + slot * 7) % 23));
            router_weights[pair] = raw_weight;
            router_sum += raw_weight;
        }
        for (int slot = 0; slot < n_expert_used; slot++) {
            const size_t pair = (size_t)token * n_expert_used + slot;
            router_weights[pair] /= router_sum;
        }
    }
    std::vector<float> X((size_t)n_tokens * expert_in_dim);
    for (float & value : X) value = activation(rng);

    const size_t mid_count = assignments * expert_mid_dim;
    const size_t down_count = assignments * out_dim;
    cudaStream_t stream = nullptr;
    void *d_gate_w = nullptr, *d_up_w = nullptr, *d_down_w = nullptr;
    void *d_gate_global_w = nullptr, *d_up_global_w = nullptr,
         *d_down_global_w = nullptr;
    float *d_x = nullptr, *d_router = nullptr;
    int32_t *d_ids = nullptr, *d_global_ids = nullptr;
    float *d_gate_ref = nullptr, *d_up_ref = nullptr, *d_mid_ref = nullptr,
          *d_down_ref = nullptr;
    float *d_gate_got = nullptr, *d_up_got = nullptr, *d_mid_got = nullptr,
          *d_down_got = nullptr;
    float *d_gate_global = nullptr, *d_up_global = nullptr,
          *d_mid_global = nullptr, *d_down_global = nullptr;

    bool allocated = (persistent_q81 ||
                      cudaStreamCreate(&stream) == cudaSuccess) &&
        cudaMalloc(&d_gate_w, gate_compact.size() * sizeof(block_iq2_xxs)) == cudaSuccess &&
        cudaMalloc(&d_up_w, up_compact.size() * sizeof(block_iq2_xxs)) == cudaSuccess &&
        cudaMalloc(&d_down_w, down_compact.size() * sizeof(block_q2_K)) == cudaSuccess &&
        cudaMalloc(&d_gate_global_w, gate_global.size() * sizeof(block_iq2_xxs)) == cudaSuccess &&
        cudaMalloc(&d_up_global_w, up_global.size() * sizeof(block_iq2_xxs)) == cudaSuccess &&
        cudaMalloc(&d_down_global_w, down_global.size() * sizeof(block_q2_K)) == cudaSuccess &&
        cudaMalloc(&d_x, X.size() * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_ids, remapped_ids.size() * sizeof(int32_t)) == cudaSuccess &&
        cudaMalloc(&d_global_ids, global_ids.size() * sizeof(int32_t)) == cudaSuccess &&
        cudaMalloc(&d_router, router_weights.size() * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_gate_ref, mid_count * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_up_ref, mid_count * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_mid_ref, mid_count * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_down_ref, down_count * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_gate_got, mid_count * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_up_got, mid_count * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_mid_got, mid_count * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_down_got, down_count * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_gate_global, mid_count * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_up_global, mid_count * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_mid_global, mid_count * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_down_global, down_count * sizeof(float)) == cudaSuccess;

    auto cleanup = [&]() {
        if (d_down_global) cudaFree(d_down_global);
        if (d_mid_global) cudaFree(d_mid_global);
        if (d_up_global) cudaFree(d_up_global);
        if (d_gate_global) cudaFree(d_gate_global);
        if (d_down_got) cudaFree(d_down_got);
        if (d_mid_got) cudaFree(d_mid_got);
        if (d_up_got) cudaFree(d_up_got);
        if (d_gate_got) cudaFree(d_gate_got);
        if (d_down_ref) cudaFree(d_down_ref);
        if (d_mid_ref) cudaFree(d_mid_ref);
        if (d_up_ref) cudaFree(d_up_ref);
        if (d_gate_ref) cudaFree(d_gate_ref);
        if (d_router) cudaFree(d_router);
        if (d_global_ids) cudaFree(d_global_ids);
        if (d_ids) cudaFree(d_ids);
        if (d_x) cudaFree(d_x);
        if (d_down_global_w) cudaFree(d_down_global_w);
        if (d_up_global_w) cudaFree(d_up_global_w);
        if (d_gate_global_w) cudaFree(d_gate_global_w);
        if (d_down_w) cudaFree(d_down_w);
        if (d_up_w) cudaFree(d_up_w);
        if (d_gate_w) cudaFree(d_gate_w);
        if (stream) cudaStreamDestroy(stream);
        if (persistent_q81) {
            unsetenv("DS4_CUDA_MMQ_Q81_PERSISTENT");
            ds4_mmq_set_gb10_optimizations(0);
        }
    };
    if (!allocated) {
        fprintf(stderr, "fused raw parity allocation failed\nFAIL\n\n");
        cleanup();
        return false;
    }

    cudaMemcpyAsync(d_gate_w, gate_compact.data(),
                    gate_compact.size() * sizeof(block_iq2_xxs),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_up_w, up_compact.data(),
                    up_compact.size() * sizeof(block_iq2_xxs),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_down_w, down_compact.data(),
                    down_compact.size() * sizeof(block_q2_K),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_gate_global_w, gate_global.data(),
                    gate_global.size() * sizeof(block_iq2_xxs),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_up_global_w, up_global.data(),
                    up_global.size() * sizeof(block_iq2_xxs),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_down_global_w, down_global.data(),
                    down_global.size() * sizeof(block_q2_K),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x, X.data(), X.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ids, remapped_ids.data(),
                    remapped_ids.size() * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_global_ids, global_ids.data(),
                    global_ids.size() * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_router, router_weights.data(),
                    router_weights.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // A rejected shape must be retryable and must not enqueue writes.  Check
    // every materialized output, not just the final down buffer, with canaries.
    cudaMemsetAsync(d_gate_got, 0xA5, mid_count * sizeof(float), stream);
    cudaMemsetAsync(d_up_got, 0xA5, mid_count * sizeof(float), stream);
    cudaMemsetAsync(d_mid_got, 0xA5, mid_count * sizeof(float), stream);
    cudaMemsetAsync(d_down_got, 0xA5, down_count * sizeof(float), stream);
    const int rc_na = ds4_mmq_iq2_xxs_q2_K_moe_fused_raw(
        d_gate_w, d_up_w, d_down_w, d_x, d_ids, d_router,
        d_gate_got, d_up_got, d_mid_got, d_down_got,
        expert_mid_dim, expert_in_dim, /*out_dim=*/0,
        n_tokens, compact_experts, n_expert_used, clamp, stream);
    std::vector<uint8_t> gate_canary(mid_count * sizeof(float));
    std::vector<uint8_t> up_canary(mid_count * sizeof(float));
    std::vector<uint8_t> mid_canary(mid_count * sizeof(float));
    std::vector<uint8_t> down_canary(down_count * sizeof(float));
    cudaMemcpyAsync(gate_canary.data(), d_gate_got, gate_canary.size(),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(up_canary.data(), d_up_got, up_canary.size(),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(mid_canary.data(), d_mid_got, mid_canary.size(),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(down_canary.data(), d_down_got, down_canary.size(),
                    cudaMemcpyDeviceToHost, stream);
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    const auto canary_intact = [](const std::vector<uint8_t> & bytes) {
        return std::all_of(bytes.begin(), bytes.end(),
                           [](uint8_t value) { return value == 0xA5; });
    };
    const bool na_ok = rc_na == DS4_MMQ_NOT_APPLICABLE &&
        sync_err == cudaSuccess && canary_intact(gate_canary) &&
        canary_intact(up_canary) && canary_intact(mid_canary) &&
        canary_intact(down_canary);

    cudaMemsetAsync(d_gate_ref, 0, mid_count * sizeof(float), stream);
    cudaMemsetAsync(d_up_ref, 0, mid_count * sizeof(float), stream);
    cudaMemsetAsync(d_mid_ref, 0, mid_count * sizeof(float), stream);
    cudaMemsetAsync(d_down_ref, 0, down_count * sizeof(float), stream);
    cudaMemsetAsync(d_gate_got, 0, mid_count * sizeof(float), stream);
    cudaMemsetAsync(d_up_got, 0, mid_count * sizeof(float), stream);
    cudaMemsetAsync(d_mid_got, 0, mid_count * sizeof(float), stream);
    cudaMemsetAsync(d_down_got, 0, down_count * sizeof(float), stream);
    cudaMemsetAsync(d_gate_global, 0, mid_count * sizeof(float), stream);
    cudaMemsetAsync(d_up_global, 0, mid_count * sizeof(float), stream);
    cudaMemsetAsync(d_mid_global, 0, mid_count * sizeof(float), stream);
    cudaMemsetAsync(d_down_global, 0, down_count * sizeof(float), stream);

    const int rc_pair = ds4_mmq_iq2_xxs_moe_pair(
        d_gate_w, d_up_w, d_x, d_ids, d_gate_ref, d_up_ref,
        expert_mid_dim, expert_in_dim, n_tokens, compact_experts,
        n_expert_used, stream);
    test_swiglu_weighted_f32<<<
        (unsigned)((mid_count + 255u) / 256u), 256, 0, stream>>>(
            d_gate_ref, d_up_ref, d_router, d_mid_ref, mid_count,
            expert_mid_dim, clamp);
    const cudaError_t swiglu_err = cudaGetLastError();
    const int rc_down = ds4_mmq_q2_K_moe(
        d_down_w, d_mid_ref, d_ids, d_down_ref,
        out_dim, expert_mid_dim, (int)assignments, compact_experts,
        /*n_expert_used=*/1, stream);
    uint64_t q81_candidates0 = 0, q81_uses0 = 0, q81_hits0 = 0;
    uint64_t q81_fallbacks0 = 0, q81_allocations0 = 0, q81_resizes0 = 0;
    size_t q81_arena0 = 0, q81_high_water0 = 0;
    uint64_t q81_candidates_off = 0, q81_uses_off = 0, q81_hits_off = 0;
    uint64_t q81_fallbacks_off = 0, q81_allocations_off = 0;
    uint64_t q81_resizes_off = 0;
    size_t q81_arena_off = 0, q81_high_water_off = 0;
    if (persistent_q81) {
        ds4_mmq_q81_persistent_counters(
            &q81_candidates0, &q81_uses0, &q81_hits0,
            &q81_fallbacks0, &q81_allocations0, &q81_resizes0,
            &q81_arena0, &q81_high_water0);
    }
    const int rc_fused = ds4_mmq_iq2_xxs_q2_K_moe_fused_raw(
        d_gate_w, d_up_w, d_down_w, d_x, d_ids, d_router,
        d_gate_got, d_up_got, d_mid_got, d_down_got,
        expert_mid_dim, expert_in_dim, out_dim,
        n_tokens, compact_experts, n_expert_used, clamp, stream);
    if (persistent_q81) {
        ds4_mmq_q81_persistent_counters(
            &q81_candidates_off, &q81_uses_off, &q81_hits_off,
            &q81_fallbacks_off, &q81_allocations_off, &q81_resizes_off,
            &q81_arena_off, &q81_high_water_off);
        setenv("DS4_CUDA_MMQ_Q81_PERSISTENT", "1", 1);
    }
    // Run the same fused path against the original global expert table and
    // unremapped ids.  Bitwise equality with the compact result validates the
    // full-expert copies and, with the multi-block shape above, both raw
    // channel strides independently of the materialized compact reference.
    const int rc_global = ds4_mmq_iq2_xxs_q2_K_moe_fused_raw(
        d_gate_global_w, d_up_global_w, d_down_global_w,
        d_x, d_global_ids, d_router,
        d_gate_global, d_up_global, d_mid_global, d_down_global,
        expert_mid_dim, expert_in_dim, out_dim,
        n_tokens, global_experts, n_expert_used, clamp, stream);
    // Traverse the real acquire/grow path with a deterministic requirement
    // just above the lazy minimum.  This drains the first dispatch before
    // retiring its arena without making the parity fixture itself enormous.
    constexpr size_t q81_growth_required =
        4u * 1024u * 1024u + 257u;
    const int rc_q81_grow = persistent_q81
        ? ds4_mmq_q81_persistent_preflight_for_test(
              persistent_device, q81_growth_required)
        : 0;
    // The second identical dispatch must reuse the same owned arena and raises
    // the hit counter; default-stream order makes overwriting its Q8 input safe.
    const int rc_global_reuse = persistent_q81
        ? ds4_mmq_iq2_xxs_q2_K_moe_fused_raw(
              d_gate_global_w, d_up_global_w, d_down_global_w,
              d_x, d_global_ids, d_router,
              d_gate_global, d_up_global, d_mid_global, d_down_global,
              expert_mid_dim, expert_in_dim, out_dim,
              n_tokens, global_experts, n_expert_used, clamp, stream)
        : 0;

    std::vector<float> gate_ref(mid_count), up_ref(mid_count),
                       mid_ref(mid_count), down_ref(down_count);
    std::vector<float> gate_got(mid_count), up_got(mid_count),
                       mid_got(mid_count), down_got(down_count);
    std::vector<float> gate_global_out(mid_count), up_global_out(mid_count),
                       mid_global_out(mid_count), down_global_out(down_count);
    cudaMemcpyAsync(gate_ref.data(), d_gate_ref, mid_count * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(up_ref.data(), d_up_ref, mid_count * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(mid_ref.data(), d_mid_ref, mid_count * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(down_ref.data(), d_down_ref, down_count * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(gate_got.data(), d_gate_got, mid_count * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(up_got.data(), d_up_got, mid_count * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(mid_got.data(), d_mid_got, mid_count * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(down_got.data(), d_down_got, down_count * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(gate_global_out.data(), d_gate_global,
                    mid_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(up_global_out.data(), d_up_global,
                    mid_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(mid_global_out.data(), d_mid_global,
                    mid_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(down_global_out.data(), d_down_global,
                    down_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    sync_err = cudaStreamSynchronize(stream);

    uint64_t q81_candidates1 = 0, q81_uses1 = 0, q81_hits1 = 0;
    uint64_t q81_fallbacks1 = 0, q81_allocations1 = 0, q81_resizes1 = 0;
    size_t q81_arena1 = 0, q81_high_water1 = 0;
    if (persistent_q81) {
        ds4_mmq_q81_persistent_counters(
            &q81_candidates1, &q81_uses1, &q81_hits1,
            &q81_fallbacks1, &q81_allocations1, &q81_resizes1,
            &q81_arena1, &q81_high_water1);
        ds4_mmq_q81_persistent_report();
    }
    const size_t q81_growth_aligned =
        (q81_growth_required + 255u) & ~(size_t)255u;
    const bool q81_counters_ok = !persistent_q81 ||
        (q81_lazy_init_rc == 0 &&
         q81_init_allocations1 == q81_init_allocations0 &&
         q81_init_resizes1 == q81_init_resizes0 &&
         q81_init_arena0 == 0 && q81_init_arena1 == 0 &&
         q81_candidates_off == q81_candidates0 &&
         q81_uses_off == q81_uses0 && q81_hits_off == q81_hits0 &&
         q81_fallbacks_off == q81_fallbacks0 &&
         q81_allocations_off == q81_allocations0 &&
         q81_resizes_off == q81_resizes0 &&
         q81_arena_off == 0 &&
         q81_candidates1 - q81_candidates_off == 3u &&
         q81_uses1 - q81_uses_off == 3u &&
         q81_hits1 - q81_hits_off == 1u &&
         q81_fallbacks1 == q81_fallbacks_off &&
         q81_allocations1 - q81_allocations_off == 2u &&
         q81_resizes1 - q81_resizes_off == 1u &&
         q81_arena1 == q81_growth_aligned &&
         q81_high_water1 >= q81_growth_required &&
         q81_high_water1 >= q81_high_water0);
    if (persistent_q81) {
        fprintf(stderr,
                "q81 persistent lazy_init=%d arena=%zu->%zu alloc=%llu->%llu "
                "resize=%llu->%llu "
                "delta(off c/u/h/f/a/r)=%llu/%llu/%llu/%llu/%llu/%llu "
                "delta(on c/u/h/f/a/r)=%llu/%llu/%llu/%llu/%llu/%llu "
                "arena=%zu high=%zu\n",
                q81_lazy_init_rc, q81_init_arena0, q81_init_arena1,
                (unsigned long long)q81_init_allocations0,
                (unsigned long long)q81_init_allocations1,
                (unsigned long long)q81_init_resizes0,
                (unsigned long long)q81_init_resizes1,
                (unsigned long long)(q81_candidates_off - q81_candidates0),
                (unsigned long long)(q81_uses_off - q81_uses0),
                (unsigned long long)(q81_hits_off - q81_hits0),
                (unsigned long long)(q81_fallbacks_off - q81_fallbacks0),
                (unsigned long long)(q81_allocations_off - q81_allocations0),
                (unsigned long long)(q81_resizes_off - q81_resizes0),
                (unsigned long long)(q81_candidates1 - q81_candidates_off),
                (unsigned long long)(q81_uses1 - q81_uses_off),
                (unsigned long long)(q81_hits1 - q81_hits_off),
                (unsigned long long)(q81_fallbacks1 - q81_fallbacks_off),
                (unsigned long long)(q81_allocations1 - q81_allocations_off),
                (unsigned long long)(q81_resizes1 - q81_resizes_off),
                q81_arena1, q81_high_water1);
    }
    int final_arena_cleanup = 0;
    if (persistent_q81) {
        unsetenv("DS4_CUDA_MMQ_Q81_PERSISTENT");
        final_arena_cleanup = ds4_mmq_q81_persistent_cleanup();
        ds4_mmq_set_gb10_optimizations(0);
    }

    const auto mismatches = [](const std::vector<float> & a,
                               const std::vector<float> & b) {
        size_t bad = 0;
        for (size_t i = 0; i < a.size(); i++) {
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) bad++;
        }
        return bad;
    };
    const size_t gate_bad = mismatches(gate_got, gate_ref);
    const size_t up_bad = mismatches(up_got, up_ref);
    const size_t mid_bad = mismatches(mid_got, mid_ref);
    const size_t down_bad = mismatches(down_got, down_ref);
    const size_t gate_remap_bad = mismatches(gate_got, gate_global_out);
    const size_t up_remap_bad = mismatches(up_got, up_global_out);
    const size_t mid_remap_bad = mismatches(mid_got, mid_global_out);
    const size_t down_remap_bad = mismatches(down_got, down_global_out);
    const bool ok = na_ok && rc_pair == 0 && swiglu_err == cudaSuccess &&
        rc_down == 0 && rc_fused == 0 && rc_global == 0 &&
        rc_global_reuse == 0 && rc_q81_grow == 0 &&
        initial_arena_cleanup == 0 &&
        final_arena_cleanup == 0 && q81_counters_ok &&
        sync_err == cudaSuccess && gate_bad == 0 && up_bad == 0 &&
        mid_bad == 0 && down_bad == 0 && gate_remap_bad == 0 &&
        up_remap_bad == 0 && mid_remap_bad == 0 && down_remap_bad == 0;
    fprintf(stderr,
            "rc_na=%d canary=%s rc_pair=%d swiglu=%s rc_down=%d "
            "rc_fused=%d rc_global=%d rc_grow=%d rc_reuse=%d "
            "mismatches(g/u/m/d)=%zu/%zu/%zu/%zu "
            "remap_mismatches(g/u/m/d)=%zu/%zu/%zu/%zu sync=%s\n%s\n\n",
        rc_na, na_ok ? "intact" : "FAILED", rc_pair,
            cudaGetErrorString(swiglu_err), rc_down, rc_fused, rc_global,
            rc_q81_grow, rc_global_reuse,
            gate_bad, up_bad, mid_bad, down_bad,
            gate_remap_bad, up_remap_bad, mid_remap_bad, down_remap_bad,
            cudaGetErrorString(sync_err), ok ? "PASS" : "FAIL");

    cleanup();
    return ok;
}

bool run_q4_K_moe(int M, int K, int nt, int ne, int nu, uint32_t seed) {
    auto fn = [](block_q4_K * blk, float * out,
                 int n_experts, int M, int K, int blocks_per_expert,
                 std::mt19937 & rng) {
        const int blocks_per_row = K / QK_K_LOCAL;
        for (int e = 0; e < n_experts; e++) {
            block_q4_K * eblk = blk + (size_t)e * blocks_per_expert;
            for (int row = 0; row < M; row++) {
                for (int b = 0; b < blocks_per_row; b++) {
                    generate_random_block_q4_K(&eblk[row * blocks_per_row + b], rng);
                }
                dequantize_row_q4_K_cpu(&eblk[row * blocks_per_row],
                                        out + ((size_t)e * M + row) * K, K);
            }
        }
    };
    // Q4_K's 6-bit scale * 4-bit quant accumulator path agrees with the CPU
    // reference to within ~0.20*sqrt(K), same envelope as IQ2_XXS.
    return run_moe_generic<block_q4_K>(
        "Q4_K/MOE", QK_K_LOCAL, M, K, nt, ne, nu, seed, 0.20f, fn, ds4_mmq_q4_K_moe);
}

// --------------------------------------------------------------------------
// Step 6 - mmvq vector matmul parity runners.
//
// The single-W _moe_vec entries match ds4_mmq_<type>_moe exactly in shape
// and semantics (output is the dequantized W @ X matmul, no fusion); they
// route through mmvq instead of mmq. Reuse run_moe_generic with the same
// tolerance bounds.
//
// The pair-fused _moe_pair_vec entries produce silu(W_a @ X) * (W_b @ X)
// in a single launch via mmvq's built-in fusion. The verifier runs two
// back-to-back single-W _moe calls on the same activation and ids, applies
// silu on the host, then compares the elementwise product against the
// fused output. Tolerance is the sum-of-roundings bound.
// --------------------------------------------------------------------------

bool run_q8_0_moe_vec(int M, int K, int nt, int ne, int nu, uint32_t seed) {
    auto fn = [](cpu_block_q8_0 * blk, float * out,
                 int n_experts, int M, int K, int blocks_per_expert,
                 std::mt19937 & rng) {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        const int blocks_per_row = K / QK8_0;
        for (int e = 0; e < n_experts; e++) {
            cpu_block_q8_0 * eblk = blk + (size_t)e * blocks_per_expert;
            std::vector<float> src((size_t)M * K);
            for (auto & v : src) v = nd(rng);
            for (int row = 0; row < M; row++) {
                quantize_row_q8_0_cpu(&src[(size_t)row * K], &eblk[row * blocks_per_row], K);
                const cpu_block_q8_0 * rowblk = &eblk[row * blocks_per_row];
                float * orow = out + ((size_t)e * M + row) * K;
                for (int b = 0; b < blocks_per_row; b++) {
                    const float d = fp16_to_float(rowblk[b].d);
                    for (int j = 0; j < QK8_0; j++) {
                        orow[b * QK8_0 + j] = d * rowblk[b].qs[j];
                    }
                }
            }
        }
    };
    return run_moe_generic<cpu_block_q8_0>(
        "Q8_0/MOE_VEC", QK8_0, M, K, nt, ne, nu, seed, 0.05f, fn, ds4_mmq_q8_0_moe_vec);
}

bool run_q2_K_moe_vec(int M, int K, int nt, int ne, int nu, uint32_t seed) {
    auto fn = [](block_q2_K * blk, float * out,
                 int n_experts, int M, int K, int blocks_per_expert,
                 std::mt19937 & rng) {
        const int blocks_per_row = K / QK_K_LOCAL;
        for (int e = 0; e < n_experts; e++) {
            block_q2_K * eblk = blk + (size_t)e * blocks_per_expert;
            for (int row = 0; row < M; row++) {
                for (int b = 0; b < blocks_per_row; b++) {
                    generate_random_block_q2_K(&eblk[row * blocks_per_row + b], rng);
                }
                dequantize_row_q2_K_cpu(&eblk[row * blocks_per_row],
                                        out + ((size_t)e * M + row) * K, K);
            }
        }
    };
    return run_moe_generic<block_q2_K>(
        "Q2_K/MOE_VEC", QK_K_LOCAL, M, K, nt, ne, nu, seed, 0.05f, fn, ds4_mmq_q2_K_moe_vec);
}

bool run_iq2_xxs_moe_vec(int M, int K, int nt, int ne, int nu, uint32_t seed) {
    auto fn = [](block_iq2_xxs * blk, float * out,
                 int n_experts, int M, int K, int blocks_per_expert,
                 std::mt19937 & rng) {
        const int blocks_per_row = K / QK_K_LOCAL;
        for (int e = 0; e < n_experts; e++) {
            block_iq2_xxs * eblk = blk + (size_t)e * blocks_per_expert;
            for (int row = 0; row < M; row++) {
                for (int b = 0; b < blocks_per_row; b++) {
                    generate_random_block_iq2_xxs(&eblk[row * blocks_per_row + b], rng);
                }
                dequantize_row_iq2_xxs_cpu(&eblk[row * blocks_per_row],
                                           out + ((size_t)e * M + row) * K, K);
            }
        }
    };
    return run_moe_generic<block_iq2_xxs>(
        "IQ2_XXS/MOE_VEC", QK_K_LOCAL, M, K, nt, ne, nu, seed, 0.20f, fn, ds4_mmq_iq2_xxs_moe_vec);
}

bool run_q4_K_moe_vec(int M, int K, int nt, int ne, int nu, uint32_t seed) {
    auto fn = [](block_q4_K * blk, float * out,
                 int n_experts, int M, int K, int blocks_per_expert,
                 std::mt19937 & rng) {
        const int blocks_per_row = K / QK_K_LOCAL;
        for (int e = 0; e < n_experts; e++) {
            block_q4_K * eblk = blk + (size_t)e * blocks_per_expert;
            for (int row = 0; row < M; row++) {
                for (int b = 0; b < blocks_per_row; b++) {
                    generate_random_block_q4_K(&eblk[row * blocks_per_row + b], rng);
                }
                dequantize_row_q4_K_cpu(&eblk[row * blocks_per_row],
                                        out + ((size_t)e * M + row) * K, K);
            }
        }
    };
    return run_moe_generic<block_q4_K>(
        "Q4_K/MOE_VEC", QK_K_LOCAL, M, K, nt, ne, nu, seed, 0.20f, fn, ds4_mmq_q4_K_moe_vec);
}

// Pair-fused mmvq verifier. Runs two single-W _moe calls (reference) and
// the fused pair_vec call (candidate). Compares
//   silu(ref_a[t,s,row]) * ref_b[t,s,row]   vs   got[t,s,row]
// elementwise. n_tokens fixed to 1 (fusion only supported at ncols_dst=1).
template <typename BlockT, typename DequantFn>
bool run_moe_pair_vec_generic(
        const char * tag, int blck_size,
        int M, int K, int n_experts, int n_expert_used,
        uint32_t seed, float abs_scale, float rel_tol,
        DequantFn gen_and_dequant,
        int (*pair_vec_entry)(const void *, const void *, const float *,
                              const int32_t *, float *,
                              int, int, int, int, cudaStream_t),
        int (*single_entry)(const void *, const float *, const int32_t *,
                            float *, int, int, int, int, int, cudaStream_t)) {
    const int n_tokens = 1;
    fprintf(stderr, "=== %s/PAIR_VEC  M=%d K=%d nexp=%d nused=%d  seed=%u ===\n",
            tag, M, K, n_experts, n_expert_used, seed);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const int blocks_per_row    = K / blck_size;
    const int blocks_per_expert = M * blocks_per_row;

    std::vector<BlockT> W_a((size_t)n_experts * blocks_per_expert);
    std::vector<BlockT> W_b((size_t)n_experts * blocks_per_expert);
    std::vector<float>  unused((size_t)n_experts * M * K);
    gen_and_dequant(W_a.data(), unused.data(), n_experts, M, K, blocks_per_expert, rng);
    gen_and_dequant(W_b.data(), unused.data(), n_experts, M, K, blocks_per_expert, rng);

    // Single-token routing: pick n_expert_used distinct experts.
    std::vector<int32_t> ids((size_t)n_tokens * n_expert_used);
    std::uniform_int_distribution<int> uexp(0, n_experts - 1);
    for (int t = 0; t < n_tokens; t++) {
        std::vector<int> picked;
        while ((int)picked.size() < n_expert_used) {
            int e = uexp(rng);
            if (std::find(picked.begin(), picked.end(), e) == picked.end()) picked.push_back(e);
        }
        for (int s = 0; s < n_expert_used; s++) ids[t * n_expert_used + s] = picked[s];
    }

    std::vector<float> X((size_t)n_tokens * K);
    for (auto & v : X) v = nd(rng);

    const int64_t ne_get_rows = (int64_t)n_tokens * n_expert_used;
    const size_t  out_count   = (size_t)M * ne_get_rows;

    cudaStream_t stream; cudaStreamCreate(&stream);
    void * dWa = nullptr; void * dWb = nullptr;
    float * dX = nullptr; int32_t * dIds = nullptr;
    float * dYa_ref = nullptr; float * dYb_ref = nullptr;
    float * dYfused = nullptr;
    cudaMalloc(&dWa, W_a.size() * sizeof(BlockT));
    cudaMalloc(&dWb, W_b.size() * sizeof(BlockT));
    cudaMalloc(&dX,  X.size() * sizeof(float));
    cudaMalloc(&dIds, ids.size() * sizeof(int32_t));
    cudaMalloc(&dYa_ref, out_count * sizeof(float));
    cudaMalloc(&dYb_ref, out_count * sizeof(float));
    cudaMalloc(&dYfused, out_count * sizeof(float));
    cudaMemcpyAsync(dWa, W_a.data(), W_a.size() * sizeof(BlockT), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dWb, W_b.data(), W_b.size() * sizeof(BlockT), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dX,  X.data(),   X.size()  * sizeof(float),   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dIds, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(dYa_ref, 0, out_count * sizeof(float), stream);
    cudaMemsetAsync(dYb_ref, 0, out_count * sizeof(float), stream);
    cudaMemsetAsync(dYfused, 0, out_count * sizeof(float), stream);

    // Reference: two single-W mmq calls.
    int rc_sa = single_entry(dWa, dX, dIds, dYa_ref, M, K, n_tokens, n_experts, n_expert_used, stream);
    int rc_sb = single_entry(dWb, dX, dIds, dYb_ref, M, K, n_tokens, n_experts, n_expert_used, stream);
    // Candidate: fused pair-vec.
    int rc_p  = pair_vec_entry(dWa, dWb, dX, dIds, dYfused,
                               M, K, n_experts, n_expert_used, stream);
    if (rc_sa != 0 || rc_sb != 0 || rc_p != 0) {
        fprintf(stderr, "%s pair_vec entry: rc_sa=%d rc_sb=%d rc_p=%d\n", tag, rc_sa, rc_sb, rc_p);
        cudaFree(dWa); cudaFree(dWb); cudaFree(dX); cudaFree(dIds);
        cudaFree(dYa_ref); cudaFree(dYb_ref); cudaFree(dYfused);
        cudaStreamDestroy(stream);
        return false;
    }

    std::vector<float> ya_ref(out_count, 0.0f), yb_ref(out_count, 0.0f);
    std::vector<float> yfused(out_count, 0.0f);
    cudaMemcpyAsync(ya_ref.data(), dYa_ref, out_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(yb_ref.data(), dYb_ref, out_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(yfused.data(), dYfused, out_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(dWa); cudaFree(dWb); cudaFree(dX); cudaFree(dIds);
    cudaFree(dYa_ref); cudaFree(dYb_ref); cudaFree(dYfused);
    cudaStreamDestroy(stream);

    // mmvq's fusion computes  out = main_matmul * silu(gate_matmul).
    // With fusion.gate = W_b passed to pair_vec_entry, the kernel computes
    //   out[i] = (W_a @ x)[i] * silu((W_b @ x)[i])
    // i.e. W_a plays the role of "up" and W_b plays the role of "gate".
    std::vector<float> expected(out_count, 0.0f);
    for (size_t i = 0; i < out_count; i++) {
        const float a = ya_ref[i];                   // W_a @ x
        const float b = yb_ref[i];                   // W_b @ x
        const float sb = b / (1.0f + std::exp(-b));  // silu(W_b @ x)
        expected[i] = a * sb;
    }

    const float abs_tol = abs_scale * std::sqrt((float)K);
    const bool ok = check_close(yfused, expected, abs_tol, rel_tol);
    fprintf(stderr, "%s\n\n", ok ? "PASS" : "FAIL");
    return ok;
}

bool run_q8_0_dense_vec(int M, int N, int K, uint32_t seed) {
    fprintf(stderr, "=== Q8_0/DENSE_VEC  M=%d N=%d K=%d  seed=%u ===\n", M, N, K, seed);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const int blocks_per_row = K / QK8_0;
    std::vector<cpu_block_q8_0> W_blk((size_t)M * blocks_per_row);
    std::vector<float>          W_deq((size_t)M * K);
    std::vector<float>          src((size_t)M * K);
    for (auto & v : src) v = nd(rng);
    for (int row = 0; row < M; row++) {
        quantize_row_q8_0_cpu(&src[(size_t)row * K], &W_blk[(size_t)row * blocks_per_row], K);
        for (int b = 0; b < blocks_per_row; b++) {
            const float d = fp16_to_float(W_blk[(size_t)row * blocks_per_row + b].d);
            for (int j = 0; j < QK8_0; j++) {
                W_deq[(size_t)row * K + b * QK8_0 + j] =
                    d * W_blk[(size_t)row * blocks_per_row + b].qs[j];
            }
        }
    }

    std::vector<float> X((size_t)N * K);
    for (auto & v : X) v = nd(rng);

    // Reference: naive GEMM with dequantized W. out[col*M + row] = W[row,:] . X[col,:]
    std::vector<float> ref_out((size_t)M * N, 0.0f);
    for (int col = 0; col < N; col++) {
        for (int row = 0; row < M; row++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += W_deq[(size_t)row * K + k] * X[(size_t)col * K + k];
            }
            ref_out[(size_t)col * M + row] = acc;
        }
    }

    cudaStream_t stream; cudaStreamCreate(&stream);
    void * dW = nullptr; float * dX = nullptr; float * dY = nullptr;
    cudaMalloc(&dW, W_blk.size() * sizeof(cpu_block_q8_0));
    cudaMalloc(&dX, X.size() * sizeof(float));
    cudaMalloc(&dY, (size_t)M * N * sizeof(float));
    cudaMemcpyAsync(dW, W_blk.data(), W_blk.size() * sizeof(cpu_block_q8_0), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dX, X.data(),     X.size() * sizeof(float),              cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(dY, 0, (size_t)M * N * sizeof(float), stream);

    int rc = ds4_mmq_q8_0_dense_vec(dW, dX, dY, M, N, K, stream);
    if (rc != 0) {
        fprintf(stderr, "ds4_mmq_q8_0_dense_vec returned %d\n", rc);
        cudaFree(dW); cudaFree(dX); cudaFree(dY); cudaStreamDestroy(stream);
        return false;
    }

    std::vector<float> got_out((size_t)M * N, 0.0f);
    cudaMemcpyAsync(got_out.data(), dY, got_out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(dW); cudaFree(dX); cudaFree(dY); cudaStreamDestroy(stream);

    const float abs_tol = 0.05f * std::sqrt((float)K);
    const bool ok = check_close(got_out, ref_out, abs_tol, 0.05f);
    fprintf(stderr, "%s\n\n", ok ? "PASS" : "FAIL");
    return ok;
}

bool run_q4_K_dense_vec_gb10_parity(
        int M, int N, int K, uint32_t seed, bool persistent_k1024) {
    fprintf(stderr,
            "=== Q4_K/DENSE_VEC_%s  M=%d N=%d K=%d  seed=%u ===\n",
            persistent_k1024 ? "PERSISTENT" : "SCRATCH",
            M, N, K, seed);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int blocks_per_row = K / QK_K_LOCAL;
    std::vector<block_q4_K> W((size_t)M * blocks_per_row);
    for (auto &blk : W) generate_random_block_q4_K(&blk, rng);
    std::vector<float> X((size_t)N * K);
    for (float &v : X) v = nd(rng);

    cudaStream_t stream = nullptr;
    void *dW = nullptr;
    void *scratch = nullptr;
    float *dX = nullptr;
    float *dRef = nullptr;
    float *dGot = nullptr;
    bool ok = cudaStreamCreate(&stream) == cudaSuccess &&
              cudaMalloc(&dW, W.size() * sizeof(block_q4_K)) == cudaSuccess &&
              cudaMalloc(&dX, X.size() * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&dRef, (size_t)M * N * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&dGot, (size_t)M * N * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&scratch, 256u * 1024u) == cudaSuccess;
    if (!ok) {
        fprintf(stderr, "Q4_K dense vec parity allocation failed\n");
        if (scratch) cudaFree(scratch);
        if (dGot) cudaFree(dGot);
        if (dRef) cudaFree(dRef);
        if (dX) cudaFree(dX);
        if (dW) cudaFree(dW);
        if (stream) cudaStreamDestroy(stream);
        return false;
    }
    cudaMemcpyAsync(dW, W.data(), W.size() * sizeof(block_q4_K),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dX, X.data(), X.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    uint64_t candidates0 = 0, uses0 = 0, fallbacks0 = 0;
    uint64_t require_failures0 = 0, oracle_calls0 = 0;
    uint64_t oracle_mismatches0 = 0, oracle_skips0 = 0;
    ds4_mmq_q4_K_k1024_persistent_counters(
        &candidates0, &uses0, &fallbacks0, &require_failures0,
        &oracle_calls0, &oracle_mismatches0, &oracle_skips0);

    unsetenv("DS4_CUDA_NO_Q4_GB10_FAST");
    unsetenv("DS4_CUDA_NO_Q4_DENSE_SCRATCH");
    unsetenv("DS4_CUDA_NO_Q4_K1024_PERSISTENT");
    unsetenv("DS4_CUDA_ENABLE_Q4_K1024_PERSISTENT");
    unsetenv("DS4_CUDA_REQUIRE_Q4_K1024_PERSISTENT");
    unsetenv("DS4_CUDA_Q4_K1024_PERSISTENT_ORACLE");
    ds4_mmq_set_gb10_optimizations(persistent_k1024 ? 1 : 0);
    ds4_mmq_set_aligned_q81_scratch(
        persistent_k1024 ? scratch : nullptr,
        persistent_k1024 ? 256u * 1024u : 0u);
    /* The reference must remain canonical even after a future default-on
     * promotion.  The authoritative kill switch makes this a real
     * reference-vs-candidate comparison rather than candidate-vs-candidate. */
    if (persistent_k1024) {
        setenv("DS4_CUDA_NO_Q4_K1024_PERSISTENT", "1", 1);
    }
    const int rc_ref = ds4_mmq_q4_K_dense_vec(
        dW, dX, dRef, M, N, K, stream);
    unsetenv("DS4_CUDA_NO_Q4_K1024_PERSISTENT");

    ds4_mmq_set_gb10_optimizations(1);
    ds4_mmq_set_aligned_q81_scratch(scratch, 256u * 1024u);
    if (persistent_k1024) {
        setenv("DS4_CUDA_ENABLE_Q4_K1024_PERSISTENT", "1", 1);
        setenv("DS4_CUDA_REQUIRE_Q4_K1024_PERSISTENT", "1", 1);
    }
    const int rc_got = ds4_mmq_q4_K_dense_vec(
        dW, dX, dGot, M, N, K, stream);

    std::vector<float> ref((size_t)M * N);
    std::vector<float> got((size_t)M * N);
    cudaMemcpyAsync(ref.data(), dRef, ref.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(got.data(), dGot, got.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    int rc_required_disabled = 0;
    int rc_oracle = 0;
    std::vector<float> oracle((size_t)M * N);
    if (persistent_k1024) {
        setenv("DS4_CUDA_NO_Q4_K1024_PERSISTENT", "1", 1);
        rc_required_disabled = ds4_mmq_q4_K_dense_vec(
            dW, dX, dGot, M, N, K, stream);
        unsetenv("DS4_CUDA_NO_Q4_K1024_PERSISTENT");
        setenv("DS4_CUDA_Q4_K1024_PERSISTENT_ORACLE", "1", 1);
        rc_oracle = ds4_mmq_q4_K_dense_vec(
            dW, dX, dGot, M, N, K, stream);
        unsetenv("DS4_CUDA_Q4_K1024_PERSISTENT_ORACLE");
        cudaMemcpyAsync(oracle.data(), dGot, oracle.size() * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    }
    const cudaError_t sync_err = cudaStreamSynchronize(stream);
    size_t mismatches = 0;
    size_t oracle_output_mismatches = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        if (std::memcmp(&ref[i], &got[i], sizeof(float)) != 0) mismatches++;
        if (persistent_k1024 &&
            std::memcmp(&ref[i], &oracle[i], sizeof(float)) != 0) {
            oracle_output_mismatches++;
        }
    }

    uint64_t candidates1 = 0, uses1 = 0, fallbacks1 = 0;
    uint64_t require_failures1 = 0, oracle_calls1 = 0;
    uint64_t oracle_mismatches1 = 0, oracle_skips1 = 0;
    ds4_mmq_q4_K_k1024_persistent_counters(
        &candidates1, &uses1, &fallbacks1, &require_failures1,
        &oracle_calls1, &oracle_mismatches1, &oracle_skips1);
    const bool counter_ok = !persistent_k1024 ||
        (candidates1 - candidates0 >= 4u &&
         uses1 - uses0 >= 2u &&
         fallbacks1 - fallbacks0 >= 2u &&
         require_failures1 - require_failures0 >= 1u &&
         oracle_calls1 - oracle_calls0 >= 1u &&
         oracle_mismatches1 == oracle_mismatches0 &&
         oracle_skips1 == oracle_skips0);
    ok = rc_ref == 0 && rc_got == 0 &&
         (!persistent_k1024 || rc_required_disabled != 0) &&
         (!persistent_k1024 || rc_oracle == 0) &&
         sync_err == cudaSuccess &&
         mismatches == 0 && oracle_output_mismatches == 0 && counter_ok;
    fprintf(stderr,
            "rc_ref=%d rc_candidate=%d rc_required_disabled=%d "
            "rc_oracle=%d mismatches=%zu oracle_output_mismatches=%zu "
            "counter_delta=%llu/%llu/%llu/%llu/%llu/%llu/%llu sync=%s\n%s\n\n",
            rc_ref, rc_got, rc_required_disabled, rc_oracle, mismatches,
            oracle_output_mismatches,
            (unsigned long long)(candidates1 - candidates0),
            (unsigned long long)(uses1 - uses0),
            (unsigned long long)(fallbacks1 - fallbacks0),
            (unsigned long long)(require_failures1 - require_failures0),
            (unsigned long long)(oracle_calls1 - oracle_calls0),
            (unsigned long long)(oracle_mismatches1 - oracle_mismatches0),
            (unsigned long long)(oracle_skips1 - oracle_skips0),
            cudaGetErrorString(sync_err),
            ok ? "PASS" : "FAIL");

    unsetenv("DS4_CUDA_ENABLE_Q4_K1024_PERSISTENT");
    unsetenv("DS4_CUDA_REQUIRE_Q4_K1024_PERSISTENT");
    unsetenv("DS4_CUDA_Q4_K1024_PERSISTENT_ORACLE");
    ds4_mmq_set_aligned_q81_scratch(nullptr, 0u);
    ds4_mmq_set_gb10_optimizations(0);
    cudaFree(scratch);
    cudaFree(dGot);
    cudaFree(dRef);
    cudaFree(dX);
    cudaFree(dW);
    cudaStreamDestroy(stream);
    return ok;
}

bool run_q4_K_grouped_vec_parity(
        int M, int N, int K, int n_groups, uint32_t seed) {
    fprintf(stderr,
            "=== Q4_K/GROUPED_VEC  M=%d N=%d K=%d groups=%d seed=%u ===\n",
            M, N, K, n_groups, seed);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int blocks_per_row = K / QK_K_LOCAL;
    const size_t blocks_per_group = (size_t)M * blocks_per_row;
    std::vector<block_q4_K> W((size_t)n_groups * blocks_per_group);
    for (auto &blk : W) generate_random_block_q4_K(&blk, rng);
    std::vector<float> X((size_t)N * n_groups * K);
    for (float &v : X) v = nd(rng);

    cudaStream_t stream = nullptr;
    void *dW = nullptr;
    void *scratch = nullptr;
    float *dX = nullptr;
    float *dRef = nullptr;
    float *dGot = nullptr;
    const size_t output_count = (size_t)N * n_groups * M;
    /* Covers the N=8, G=16, K=4096 parity envelope with room for ids. */
    const size_t scratch_bytes = 1024u * 1024u;
    bool ok = cudaStreamCreate(&stream) == cudaSuccess &&
              cudaMalloc(&dW, W.size() * sizeof(block_q4_K)) == cudaSuccess &&
              cudaMalloc(&dX, X.size() * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&dRef, output_count * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&dGot, output_count * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&scratch, scratch_bytes) == cudaSuccess;
    if (!ok) {
        fprintf(stderr, "Q4_K grouped vec parity allocation failed\n");
        if (scratch) cudaFree(scratch);
        if (dGot) cudaFree(dGot);
        if (dRef) cudaFree(dRef);
        if (dX) cudaFree(dX);
        if (dW) cudaFree(dW);
        if (stream) cudaStreamDestroy(stream);
        return false;
    }
    cudaMemcpyAsync(dW, W.data(), W.size() * sizeof(block_q4_K),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dX, X.data(), X.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    unsetenv("DS4_CUDA_NO_Q4_GB10_FAST");
    unsetenv("DS4_CUDA_NO_Q4_GROUPED_ATTN_A");
    unsetenv("DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH");
    unsetenv("DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_BATCH");
    ds4_mmq_set_gb10_optimizations(1);
    ds4_mmq_set_aligned_q81_scratch(scratch, scratch_bytes);

    int rc_ref = 0;
    for (int t = 0; t < N && rc_ref == 0; t++) {
        for (int g = 0; g < n_groups && rc_ref == 0; g++) {
            const size_t channel = (size_t)t * n_groups + g;
            rc_ref = ds4_mmq_q4_K_dense_vec(
                (const char *)dW + (size_t)g * blocks_per_group *
                    sizeof(block_q4_K),
                dX + channel * K,
                dRef + channel * M,
                M, 1, K, stream);
        }
    }

    int rc_opt_out = DS4_MMQ_NOT_APPLICABLE;
    int rc_short_scratch = DS4_MMQ_NOT_APPLICABLE;
    if (N > 1) {
        rc_opt_out = ds4_mmq_q4_K_grouped_batch_vec(
            dW, dX, dGot, M, K, N, n_groups, stream);
        setenv("DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_BATCH", "1", 1);
        const size_t padded_k = ((size_t)K + 511u) & ~(size_t)511u;
        const size_t q8_row_bytes =
            padded_k * sizeof(block_q8_1) / QK8_1;
        const size_t q8_bytes = (size_t)N * n_groups * q8_row_bytes;
        const size_t ids_offset = (q8_bytes + 15u) & ~(size_t)15u;
        const size_t required_bytes =
            ids_offset + (size_t)N * n_groups * sizeof(int32_t);
        if (required_bytes > 0u) {
            ds4_mmq_set_aligned_q81_scratch(scratch, required_bytes - 1u);
            rc_short_scratch = ds4_mmq_q4_K_grouped_batch_vec(
                dW, dX, dGot, M, K, N, n_groups, stream);
            ds4_mmq_set_aligned_q81_scratch(scratch, scratch_bytes);
        }
    }
    const int rc_got = N == 1
        ? ds4_mmq_q4_K_grouped_vec(
              dW, dX, dGot, M, K, n_groups, stream)
        : ds4_mmq_q4_K_grouped_batch_vec(
              dW, dX, dGot, M, K, N, n_groups, stream);
    if (N == 1) {
        setenv("DS4_CUDA_NO_Q4_GB10_FAST", "1", 1);
    } else {
        setenv("DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH", "1", 1);
    }
    const int rc_disabled = N == 1
        ? ds4_mmq_q4_K_grouped_vec(
              dW, dX, dGot, M, K, n_groups, stream)
        : ds4_mmq_q4_K_grouped_batch_vec(
              dW, dX, dGot, M, K, N, n_groups, stream);
    unsetenv("DS4_CUDA_NO_Q4_GB10_FAST");
    unsetenv("DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH");

    std::vector<float> ref(output_count);
    std::vector<float> got(output_count);
    cudaMemcpyAsync(ref.data(), dRef, ref.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(got.data(), dGot, got.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    const cudaError_t sync_err = cudaStreamSynchronize(stream);
    size_t mismatches = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        if (std::memcmp(&ref[i], &got[i], sizeof(float)) != 0) mismatches++;
    }
    ok = rc_ref == 0 && rc_got == 0 &&
         rc_opt_out == DS4_MMQ_NOT_APPLICABLE &&
         rc_short_scratch == DS4_MMQ_NOT_APPLICABLE &&
         rc_disabled == DS4_MMQ_NOT_APPLICABLE &&
         sync_err == cudaSuccess &&
         mismatches == 0;
    fprintf(stderr,
            "rc_ref=%d rc_grouped=%d rc_opt_out=%d rc_short_scratch=%d "
            "rc_disabled=%d "
            "mismatches=%zu sync=%s\n%s\n\n",
            rc_ref, rc_got, rc_opt_out, rc_short_scratch, rc_disabled, mismatches,
            cudaGetErrorString(sync_err),
            ok ? "PASS" : "FAIL");

    unsetenv("DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_BATCH");
    ds4_mmq_set_aligned_q81_scratch(nullptr, 0u);
    ds4_mmq_set_gb10_optimizations(0);
    cudaFree(scratch);
    cudaFree(dGot);
    cudaFree(dRef);
    cudaFree(dX);
    cudaFree(dW);
    cudaStreamDestroy(stream);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    (void)argc; (void)argv;
    int rc = ds4_mmq_init(0);
    if (rc != 0) { fprintf(stderr, "ds4_mmq_init failed: %d\n", rc); return 1; }

    bool all_ok = true;

    // Q8_0
    all_ok &= run_q8_0(/*M=*/64,   /*N=*/4,   /*K=*/256,  0xC0FFEE);
    all_ok &= run_q8_0(/*M=*/128,  /*N=*/8,   /*K=*/512,  0xDEADBEE);
    all_ok &= run_q8_0(/*M=*/64,   /*N=*/1,   /*K=*/256,  0x12345);
    all_ok &= run_q8_0(/*M=*/1024, /*N=*/16,  /*K=*/4096, 0xBAD7E11);

    // Q2_K - V4 Flash ffn_down_exps per-expert shape is (K=2048, N=4096).
    all_ok &= run_q2_K(/*M=*/64,   /*N=*/4,   /*K=*/256,  0x02C0FFEE);
    all_ok &= run_q2_K(/*M=*/128,  /*N=*/8,   /*K=*/512,  0x0205BEEF);
    all_ok &= run_q2_K(/*M=*/256,  /*N=*/1,   /*K=*/2048, 0x0206A000);
    all_ok &= run_q2_K(/*M=*/4096, /*N=*/16,  /*K=*/2048, 0x0207B000);

    // IQ2_XXS - V4 Flash ffn_gate_exps per-expert shape is (K=4096, N=2048).
    all_ok &= run_iq2_xxs(/*M=*/64,   /*N=*/4,   /*K=*/256,  0xCAFE2);
    all_ok &= run_iq2_xxs(/*M=*/128,  /*N=*/8,   /*K=*/512,  0xCAFE3);
    all_ok &= run_iq2_xxs(/*M=*/256,  /*N=*/1,   /*K=*/4096, 0xCAFE4);
    all_ok &= run_iq2_xxs(/*M=*/2048, /*N=*/16,  /*K=*/4096, 0xCAFE5);

    // Q4_K - covers gate/up + down shapes for hypothetical Q4_K MoE GGUFs.
    all_ok &= run_q4_K(/*M=*/64,   /*N=*/4,   /*K=*/256,  0xC4FE1);
    all_ok &= run_q4_K(/*M=*/128,  /*N=*/8,   /*K=*/512,  0xC4FE2);
    all_ok &= run_q4_K(/*M=*/256,  /*N=*/1,   /*K=*/2048, 0xC4FE3);
    all_ok &= run_q4_K(/*M=*/2048, /*N=*/16,  /*K=*/4096, 0xC4FE4);

    // MoE (_id) path.  Small expert counts + small shapes for fast verification.
    // Per-token-distinct routing with top_k=2 or 6.
    all_ok &= run_q8_0_moe   (/*M=*/64,   /*K=*/256,  /*nt=*/8,  /*nexp=*/4,   /*nused=*/2, 0xC0FE01);
    all_ok &= run_q8_0_moe   (/*M=*/128,  /*K=*/512,  /*nt=*/16, /*nexp=*/8,   /*nused=*/2, 0xC0FE02);
    all_ok &= run_q8_0_moe   (/*M=*/256,  /*K=*/256,  /*nt=*/4,  /*nexp=*/8,   /*nused=*/4, 0xC0FE03);
    all_ok &= run_q2_K_moe   (/*M=*/64,   /*K=*/256,  /*nt=*/8,  /*nexp=*/4,   /*nused=*/2, 0xC0FE04);
    all_ok &= run_q2_K_moe   (/*M=*/128,  /*K=*/512,  /*nt=*/16, /*nexp=*/8,   /*nused=*/2, 0xC0FE05);
    all_ok &= run_iq2_xxs_moe(/*M=*/64,   /*K=*/256,  /*nt=*/8,  /*nexp=*/4,   /*nused=*/2, 0xC0FE06);
    all_ok &= run_iq2_xxs_moe(/*M=*/128,  /*K=*/512,  /*nt=*/16, /*nexp=*/8,   /*nused=*/2, 0xC0FE07);
    // V4 Flash-realistic top-6 routing (n_expert_used=6 hits the templated
    // mm_ids_helper specialisation at mmid.cu:148).  Use a small expert
    // population so the CPU reference stays fast.
    all_ok &= run_q8_0_moe   (/*M=*/256,  /*K=*/256,  /*nt=*/8,  /*nexp=*/16,  /*nused=*/6, 0xC0FE08);
    all_ok &= run_q2_K_moe   (/*M=*/256,  /*K=*/512,  /*nt=*/8,  /*nexp=*/16,  /*nused=*/6, 0xC0FE09);
    all_ok &= run_iq2_xxs_moe(/*M=*/256,  /*K=*/512,  /*nt=*/8,  /*nexp=*/16,  /*nused=*/6, 0xC0FE0A);
    // Q4_K MoE - new in Step 2. Three shapes mirror the IQ2_XXS coverage.
    all_ok &= run_q4_K_moe   (/*M=*/64,   /*K=*/256,  /*nt=*/8,  /*nexp=*/4,   /*nused=*/2, 0xC4FE05);
    all_ok &= run_q4_K_moe   (/*M=*/128,  /*K=*/512,  /*nt=*/16, /*nexp=*/8,   /*nused=*/2, 0xC4FE06);
    all_ok &= run_q4_K_moe   (/*M=*/256,  /*K=*/512,  /*nt=*/8,  /*nexp=*/16,  /*nused=*/6, 0xC4FE07);

    // Step 3 - paired MoE (one quantize, two matmuls).  Each call asserts
    // bit-identity vs two back-to-back single-W moe calls over the same
    // ids + activation.  Lambdas capture the same random-block generators
    // used by run_*_moe above.
    auto gen_iq2 = [](block_iq2_xxs * blk, float * out,
                      int n_experts, int M, int K, int blocks_per_expert,
                      std::mt19937 & rng) {
        const int blocks_per_row = K / QK_K_LOCAL;
        for (int e = 0; e < n_experts; e++) {
            block_iq2_xxs * eblk = blk + (size_t)e * blocks_per_expert;
            for (int row = 0; row < M; row++) {
                for (int b = 0; b < blocks_per_row; b++) {
                    generate_random_block_iq2_xxs(&eblk[row * blocks_per_row + b], rng);
                }
                dequantize_row_iq2_xxs_cpu(&eblk[row * blocks_per_row],
                                           out + ((size_t)e * M + row) * K, K);
            }
        }
    };
    auto gen_q4k = [](block_q4_K * blk, float * out,
                      int n_experts, int M, int K, int blocks_per_expert,
                      std::mt19937 & rng) {
        const int blocks_per_row = K / QK_K_LOCAL;
        for (int e = 0; e < n_experts; e++) {
            block_q4_K * eblk = blk + (size_t)e * blocks_per_expert;
            for (int row = 0; row < M; row++) {
                for (int b = 0; b < blocks_per_row; b++) {
                    generate_random_block_q4_K(&eblk[row * blocks_per_row + b], rng);
                }
                dequantize_row_q4_K_cpu(&eblk[row * blocks_per_row],
                                        out + ((size_t)e * M + row) * K, K);
            }
        }
    };
    all_ok &= run_moe_pair_generic<block_iq2_xxs>(
        "IQ2_XXS", QK_K_LOCAL, /*M=*/256, /*K=*/512, /*nt=*/8,
        /*ne=*/16, /*nu=*/6, 0xC0FE10, gen_iq2,
        ds4_mmq_iq2_xxs_moe_pair, ds4_mmq_iq2_xxs_moe);
    all_ok &= run_moe_pair_generic<block_q4_K>(
        "Q4_K", QK_K_LOCAL, /*M=*/256, /*K=*/512, /*nt=*/8,
        /*ne=*/16, /*nu=*/6, 0xC4FE10, gen_q4k,
        ds4_mmq_q4_K_moe_pair, ds4_mmq_q4_K_moe);

    // SSD compact-table raw fusion: one expert map/activation quantize for
    // IQ2 gate+up and Q2 down.  Cover the production top-6 routing shape at
    // each target prefill width.
    all_ok &= run_iq2_xxs_q2_K_fused_raw_parity(/*nt=*/8,   0xC2F008);
    all_ok &= run_iq2_xxs_q2_K_fused_raw_parity(/*nt=*/32,  0xC2F020);
    all_ok &= run_iq2_xxs_q2_K_fused_raw_parity(/*nt=*/128, 0xC2F080);
    all_ok &= run_iq2_xxs_q2_K_fused_raw_parity(
        /*nt=*/32, 0xC2F021, /*persistent_q81=*/true);

    // Step 6 - mmvq vector matmul tests.
    //
    // Single-W _moe_vec tests. Two shape classes per type:
    //   - Decode gate/up: n_tokens=1, n_expert_used=6
    //   - Decode down  : n_tokens=6, n_expert_used=1 (reinterpretation
    //                    used by routed_moe_launch's mmvq branch)
    // The down reinterpretation hits mmvq's multi-token MoE kernel
    // (mul_mat_vec_q_moe) which we want to exercise.
    all_ok &= run_q8_0_moe_vec   (/*M=*/64,   /*K=*/256,  /*nt=*/1,  /*nexp=*/16,  /*nused=*/6, 0xC0FE20);
    all_ok &= run_q8_0_moe_vec   (/*M=*/256,  /*K=*/256,  /*nt=*/6,  /*nexp=*/16,  /*nused=*/1, 0xC0FE21);
    all_ok &= run_q2_K_moe_vec   (/*M=*/64,   /*K=*/256,  /*nt=*/1,  /*nexp=*/16,  /*nused=*/6, 0xC0FE22);
    all_ok &= run_q2_K_moe_vec   (/*M=*/256,  /*K=*/512,  /*nt=*/6,  /*nexp=*/16,  /*nused=*/1, 0xC0FE23);
    all_ok &= run_iq2_xxs_moe_vec(/*M=*/64,   /*K=*/256,  /*nt=*/1,  /*nexp=*/16,  /*nused=*/6, 0xC0FE24);
    all_ok &= run_iq2_xxs_moe_vec(/*M=*/256,  /*K=*/512,  /*nt=*/6,  /*nexp=*/16,  /*nused=*/1, 0xC0FE25);
    all_ok &= run_q4_K_moe_vec   (/*M=*/64,   /*K=*/256,  /*nt=*/1,  /*nexp=*/16,  /*nused=*/6, 0xC0FE26);
    all_ok &= run_q4_K_moe_vec   (/*M=*/256,  /*K=*/512,  /*nt=*/6,  /*nexp=*/16,  /*nused=*/1, 0xC0FE27);

    // Pair-fused mmvq tests. n_tokens=1 only (mmvq fusion is ncols_dst=1).
    // rel_tol is loosened from 0.05 because silu near a sign change
    // amplifies fp noise from the matmul accumulator: a small absolute
    // difference in the gate matmul output can produce a large relative
    // difference in silu(gate)*up when gate transitions sign.  Q4_K has
    // higher dynamic range than IQ2_XXS so its envelope is wider.
    all_ok &= run_moe_pair_vec_generic<block_iq2_xxs>(
        "IQ2_XXS", QK_K_LOCAL, /*M=*/256, /*K=*/512,
        /*ne=*/16, /*nu=*/6, 0xC0FE30, /*abs=*/0.30f, /*rel=*/0.10f, gen_iq2,
        ds4_mmq_iq2_xxs_moe_pair_vec, ds4_mmq_iq2_xxs_moe);
    all_ok &= run_moe_pair_vec_generic<block_q4_K>(
        "Q4_K", QK_K_LOCAL, /*M=*/256, /*K=*/512,
        /*ne=*/16, /*nu=*/6, 0xC4FE30, /*abs=*/0.30f, /*rel=*/0.40f, gen_q4k,
        ds4_mmq_q4_K_moe_pair_vec, ds4_mmq_q4_K_moe);

    // Dense Q8_0 mmvq (decode attention projection shape).
    all_ok &= run_q8_0_dense_vec(/*M=*/64,   /*N=*/1, /*K=*/256,  0xC0FE40);
    all_ok &= run_q8_0_dense_vec(/*M=*/256,  /*N=*/1, /*K=*/512,  0xC0FE41);
    all_ok &= run_q8_0_dense_vec(/*M=*/1024, /*N=*/1, /*K=*/4096, 0xC0FE42);
    all_ok &= run_q4_K_dense_vec_gb10_parity(
        /*M=*/1024, /*N=*/5, /*K=*/4096, 0xC4FE40, false);
    all_ok &= run_q4_K_dense_vec_gb10_parity(
        /*M=*/32768, /*N=*/1, /*K=*/1024, 0xC4FE41, true);
    // Preserve coverage of the original one-token grouped ABI.
    all_ok &= run_q4_K_grouped_vec_parity(
        /*M=*/64, /*N=*/1, /*K=*/512, /*groups=*/4, 0xC4FE45);
    all_ok &= run_q4_K_grouped_vec_parity(
        /*M=*/256, /*N=*/2, /*K=*/8192, /*groups=*/4, 0xC4FE42);
    // Small synthetic Flash-like case for quick token-aware coverage.
    all_ok &= run_q4_K_grouped_vec_parity(
        /*M=*/128, /*N=*/5, /*K=*/4096, /*groups=*/8, 0xC4FE43);
    // DeepSeek-V4 Flash AProjQ4 attention-A production shape: each of the
    // eight output groups owns a [rank=1024, group_dim=4096] Q4_K matrix.
    // Cover both ordinary one-token decode and the maximum DSpark proposal
    // width used by the verifier.
    all_ok &= run_q4_K_grouped_vec_parity(
        /*M=*/1024, /*N=*/1, /*K=*/4096, /*groups=*/8, 0xC4FE46);
    all_ok &= run_q4_K_grouped_vec_parity(
        /*M=*/1024, /*N=*/5, /*K=*/4096, /*groups=*/8, 0xC4FE47);
    // Pro-style maximum group count and the N=8 API ceiling.
    all_ok &= run_q4_K_grouped_vec_parity(
        /*M=*/64, /*N=*/8, /*K=*/4096, /*groups=*/16, 0xC4FE44);
    // DeepSeek-V4 Pro has the same rank/group_dim as Flash and 16 groups.
    all_ok &= run_q4_K_grouped_vec_parity(
        /*M=*/1024, /*N=*/1, /*K=*/4096, /*groups=*/16, 0xC4FE48);

    fprintf(stderr, "===================\n");
    fprintf(stderr, "%s\n", all_ok ? "ALL PASS" : "SOME FAILED");
    return all_ok ? 0 : 1;
}
