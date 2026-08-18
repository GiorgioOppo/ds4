// SPDX-License-Identifier: MIT
// Deterministic ROCm Q4_K dense/pair oracle.
//
// The test deliberately goes through the public tensor/model-map API.  Weight
// rows use the raw 144-byte GGUF Q4_K layout, while the CPU reference mirrors
// the backend's F32 -> Q8_K quantizer and Q4_K x Q8_K integer dot product.

#include "ds4_gpu.h"

#if defined(__has_include)
#  if __has_include(<hip/hip_runtime.h>)
#    include <hip/hip_runtime.h>
#    define DS4_TEST_HAS_HIP_RUNTIME 1
#  endif
#endif
#ifndef DS4_TEST_HAS_HIP_RUNTIME
#  define DS4_TEST_HAS_HIP_RUNTIME 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kQkK = 256u;
constexpr uint32_t kK = 4096u;
constexpr uint32_t kM0 = 65u;
constexpr uint32_t kM1 = 33u;
constexpr uint32_t kQ4Type = 12u;
constexpr float kCpuAbsTolerance = 2.0e-3f;
constexpr float kCpuRelTolerance = 3.0e-5f;
constexpr int kSkip = 77;

struct block_q4_K_test {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[kQkK / 2u];
};

struct block_q8_K_test {
    float d;
    int8_t qs[kQkK];
    int16_t bsums[kQkK / 16u];
};

static_assert(sizeof(block_q4_K_test) == 144u,
              "Q4_K fixture must match the raw GGUF layout");
static_assert(sizeof(block_q8_K_test) == 292u,
              "Q8_K oracle must match the ROCm activation layout");

struct tensor_owner {
    ds4_gpu_tensor *ptr = nullptr;

    explicit tensor_owner(uint64_t bytes) : ptr(ds4_gpu_tensor_alloc(bytes)) {}
    ~tensor_owner() { ds4_gpu_tensor_free(ptr); }

    tensor_owner(const tensor_owner &) = delete;
    tensor_owner &operator=(const tensor_owner &) = delete;
};

struct aligned_model {
    uint8_t *data = nullptr;
    uint64_t size = 0;
    uint64_t weight0_offset = 0;
    uint64_t weight1_offset = 0;

    ~aligned_model() { std::free(data); }

    aligned_model(const aligned_model &) = delete;
    aligned_model &operator=(const aligned_model &) = delete;
    aligned_model() = default;
};

struct env_snapshot {
    const char *name;
    bool was_set;
    std::string value;

    explicit env_snapshot(const char *key)
        : name(key), was_set(std::getenv(key) != nullptr),
          value(was_set ? std::getenv(key) : "") {}
    ~env_snapshot() {
        if (was_set) {
            (void)setenv(name, value.c_str(), 1);
        } else {
            (void)unsetenv(name);
        }
    }

    env_snapshot(const env_snapshot &) = delete;
    env_snapshot &operator=(const env_snapshot &) = delete;
};

uint64_t round_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

uint32_t lcg_next(uint32_t &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

float fp16_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16u;
    uint32_t exp = (h >> 10u) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mant & 0x400u) == 0u) {
                mant <<= 1u;
                shift++;
            }
            mant &= 0x3ffu;
            bits = sign | (uint32_t)(127 - 14 - shift) << 23u | mant << 13u;
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | mant << 13u;
    } else {
        bits = sign | (exp + 112u) << 23u | mant << 13u;
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t float_to_fp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits >> 31u;
    int32_t exp = (int32_t)((bits >> 23u) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp >= 31) return (uint16_t)((sign << 15u) | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)(sign << 15u);
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t rounded = mant >> shift;
        const uint32_t halfway = 1u << (shift - 1u);
        if ((mant & halfway) &&
            ((mant & (halfway - 1u)) || (rounded & 1u))) {
            rounded++;
        }
        return (uint16_t)((sign << 15u) | rounded);
    }
    uint32_t rounded = mant + 0x0fffu + ((mant >> 13u) & 1u);
    if (rounded & 0x800000u) {
        rounded = 0u;
        exp++;
        if (exp >= 31) return (uint16_t)((sign << 15u) | 0x7c00u);
    }
    return (uint16_t)((sign << 15u) | (uint32_t)exp << 10u |
                      rounded >> 13u);
}

void q4_scale_min(uint32_t j, const uint8_t *scales,
                  uint8_t *scale, uint8_t *minimum) {
    if (j < 4u) {
        *scale = scales[j] & 63u;
        *minimum = scales[j + 4u] & 63u;
    } else {
        *scale = (scales[j + 4u] & 0x0fu) |
                 (uint8_t)((scales[j - 4u] >> 6u) << 4u);
        *minimum = (scales[j + 4u] >> 4u) |
                   (uint8_t)((scales[j] >> 6u) << 4u);
    }
}

void fill_q4_rows(block_q4_K_test *rows, uint32_t n_rows, uint32_t seed) {
    uint32_t state = seed;
    const uint32_t blocks_per_row = kK / kQkK;
    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            block_q4_K_test &block = rows[(uint64_t)row * blocks_per_row + b];
            const float d = 0.0025f +
                0.00025f * (float)(1u + (lcg_next(state) % 23u));
            const float dmin = 0.0010f +
                0.00020f * (float)(1u + (lcg_next(state) % 19u));
            block.d = float_to_fp16(d);
            block.dmin = float_to_fp16(dmin);
            for (uint8_t &v : block.scales) v = (uint8_t)(lcg_next(state) >> 24u);
            for (uint8_t &v : block.qs) v = (uint8_t)(lcg_next(state) >> 24u);
        }
    }
}

bool make_model(aligned_model *model) {
    constexpr uint64_t page = 4096u;
    const uint64_t row_bytes = (kK / kQkK) * sizeof(block_q4_K_test);
    const uint64_t weight0_bytes = kM0 * row_bytes;
    const uint64_t weight1_bytes = kM1 * row_bytes;
    model->weight0_offset = 0u;
    model->weight1_offset = round_up(weight0_bytes, page);
    model->size = round_up(model->weight1_offset + weight1_bytes, page);
    void *storage = nullptr;
    if (posix_memalign(&storage, (size_t)page, (size_t)model->size) != 0) {
        return false;
    }
    model->data = static_cast<uint8_t *>(storage);
    std::memset(model->data, 0xa5, (size_t)model->size);
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->weight0_offset),
                 kM0, 0x41c64e6du);
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->weight1_offset),
                 kM1, 0x9e3779b9u);
    return true;
}

void fill_activation(std::vector<float> *x, uint32_t n_tokens) {
    x->resize((uint64_t)n_tokens * kK);
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t b = 0; b < kK / kQkK; b++) {
            float *block = x->data() + (uint64_t)token * kK + b * kQkK;
            for (uint32_t i = 0; i < kQkK; i++) {
                const int q = (int)((i * 73u + token * 37u + b * 19u) % 241u) - 120;
                block[i] = (float)q / 32.0f;
            }
            // A unique, exactly representable maximum makes the CPU and GPU
            // quantizers select the same signed scale without tie ambiguity.
            block[0] = ((token + b) & 1u) ? 127.0f / 32.0f
                                          : -127.0f / 32.0f;
        }
    }
}

void quantize_q8_K_cpu(const float *x, block_q8_K_test *out) {
    float amax = 0.0f;
    float maxv = 0.0f;
    for (uint32_t i = 0; i < kQkK; i++) {
        const float av = std::fabs(x[i]);
        if (av > amax) {
            amax = av;
            maxv = x[i];
        }
    }
    if (amax == 0.0f) {
        std::memset(out, 0, sizeof(*out));
        return;
    }
    const float iscale = -127.0f / maxv;
    for (uint32_t i = 0; i < kQkK; i++) {
        int q = (int)std::lrint(iscale * x[i]);
        q = std::max(-128, std::min(127, q));
        out->qs[i] = (int8_t)q;
    }
    for (uint32_t group = 0; group < kQkK / 16u; group++) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; i++) {
            sum += out->qs[group * 16u + i];
        }
        out->bsums[group] = (int16_t)sum;
    }
    out->d = 1.0f / iscale;
}

float dot_q4_q8_raw(const block_q4_K_test &weight,
                    const block_q8_K_test &activation) {
    int isum = 0;
    int summs = 0;
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t scale = 0;
        uint8_t minimum = 0;
        q4_scale_min(j, weight.scales, &scale, &minimum);
        summs += (int)minimum *
                 ((int)activation.bsums[2u * j] +
                  (int)activation.bsums[2u * j + 1u]);
        const uint32_t byte_offset = (j >> 1u) * 32u;
        const uint32_t shift = (j & 1u) ? 4u : 0u;
        int group_dot = 0;
        for (uint32_t i = 0; i < 32u; i++) {
            const int q4 = (weight.qs[byte_offset + i] >> shift) & 0x0f;
            group_dot += q4 * (int)activation.qs[j * 32u + i];
        }
        isum += (int)scale * group_dot;
    }
    const float d = fp16_to_float(weight.d);
    const float dmin = fp16_to_float(weight.dmin);
    return activation.d * d * (float)isum -
           activation.d * dmin * (float)summs;
}

std::vector<float> dense_reference(const uint8_t *weight_base,
                                   const std::vector<float> &x,
                                   uint32_t out_dim,
                                   uint32_t n_tokens) {
    const auto *weights = reinterpret_cast<const block_q4_K_test *>(weight_base);
    constexpr uint32_t blocks_per_row = kK / kQkK;
    std::vector<block_q8_K_test> xq((uint64_t)n_tokens * blocks_per_row);
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            quantize_q8_K_cpu(x.data() + (uint64_t)token * kK + b * kQkK,
                              &xq[(uint64_t)token * blocks_per_row + b]);
        }
    }
    std::vector<float> result((uint64_t)n_tokens * out_dim, 0.0f);
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            // Mirror the kernel's b=lane; b+=8 walk and width-8 shuffle tree.
            // This makes raw-bit diagnostics meaningful even at the real
            // K=4096 shape while the tolerance remains the promotion gate.
            float lane_sum[8] = {};
            for (uint32_t lane = 0; lane < 8u; lane++) {
                for (uint32_t b = lane; b < blocks_per_row; b += 8u) {
                    lane_sum[lane] += dot_q4_q8_raw(
                        weights[(uint64_t)row * blocks_per_row + b],
                        xq[(uint64_t)token * blocks_per_row + b]);
                }
            }
            for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
                for (uint32_t lane = 0; lane + offset < 8u; lane++) {
                    lane_sum[lane] += lane_sum[lane + offset];
                }
            }
            result[(uint64_t)token * out_dim + row] = lane_sum[0];
        }
    }
    return result;
}

bool close_to_cpu(const std::vector<float> &got,
                  const std::vector<float> &expected,
                  const char *label) {
    uint64_t raw_mismatches = 0;
    uint64_t tolerance_failures = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < got.size(); i++) {
        if (std::memcmp(&got[i], &expected[i], sizeof(float)) != 0) {
            raw_mismatches++;
        }
        const float diff = std::fabs(got[i] - expected[i]);
        const float rel = diff / std::max(1.0f, std::fabs(expected[i]));
        if (diff > max_abs) {
            max_abs = diff;
            worst = i;
        }
        max_rel = std::max(max_rel, rel);
        const float limit = kCpuAbsTolerance +
                            kCpuRelTolerance * std::fabs(expected[i]);
        if (!std::isfinite(got[i]) || diff > limit) tolerance_failures++;
    }
    std::fprintf(stderr,
                 "%s: raw_mismatches=%llu/%zu max_abs=%g max_rel=%g "
                 "worst=%zu tolerance(abs=%g rel=%g) %s\n",
                 label, (unsigned long long)raw_mismatches, got.size(),
                 max_abs, max_rel, worst, kCpuAbsTolerance,
                 kCpuRelTolerance, tolerance_failures == 0 ? "PASS" : "FAIL");
    if (tolerance_failures != 0 && worst < got.size()) {
        std::fprintf(stderr, "  worst got=%g cpu=%g delta=%g\n",
                     got[worst], expected[worst], got[worst] - expected[worst]);
    }
    return tolerance_failures == 0;
}

bool bitwise_equal(const std::vector<float> &got,
                   const std::vector<float> &expected,
                   const char *label) {
    uint64_t mismatches = 0;
    size_t first = 0;
    for (size_t i = 0; i < got.size(); i++) {
        if (std::memcmp(&got[i], &expected[i], sizeof(float)) != 0) {
            if (mismatches == 0) first = i;
            mismatches++;
        }
    }
    std::fprintf(stderr, "%s: raw_mismatches=%llu/%zu %s\n",
                 label, (unsigned long long)mismatches, got.size(),
                 mismatches == 0 ? "PASS" : "FAIL");
    if (mismatches != 0) {
        uint32_t got_bits = 0;
        uint32_t expected_bits = 0;
        std::memcpy(&got_bits, &got[first], sizeof(got_bits));
        std::memcpy(&expected_bits, &expected[first], sizeof(expected_bits));
        std::fprintf(stderr,
                     "  first=%zu got=%g/0x%08x expected=%g/0x%08x\n",
                     first, got[first], got_bits, expected[first], expected_bits);
    }
    return mismatches == 0;
}

bool write_tensor(ds4_gpu_tensor *tensor, const std::vector<float> &values) {
    return tensor && ds4_gpu_tensor_write(
        tensor, 0, values.data(), values.size() * sizeof(float)) != 0;
}

bool read_tensor(const ds4_gpu_tensor *tensor, std::vector<float> *values) {
    return tensor && ds4_gpu_tensor_read(
        tensor, 0, values->data(), values->size() * sizeof(float)) != 0;
}

std::vector<float> sentinel_values(size_t count) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; i++) {
        const uint32_t bits = 0x4b000000u + (uint32_t)i;
        std::memcpy(&result[i], &bits, sizeof(bits));
    }
    return result;
}

bool run_dense_case(const aligned_model &model, uint32_t n_tokens,
                    uint64_t offset, uint32_t out_dim, const char *label) {
    std::vector<float> x;
    fill_activation(&x, n_tokens);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner out_gpu((uint64_t)n_tokens * out_dim * sizeof(float));
    if (!x_gpu.ptr || !out_gpu.ptr || !write_tensor(x_gpu.ptr, x)) {
        std::fprintf(stderr, "%s: tensor allocation/write FAIL\n", label);
        return false;
    }
    const int rc = ds4_gpu_matmul_quant_tensor(
        out_gpu.ptr, model.data, model.size, offset, kQ4Type,
        kK, out_dim, x_gpu.ptr, n_tokens);
    std::vector<float> got((uint64_t)n_tokens * out_dim);
    if (rc == 0 || !read_tensor(out_gpu.ptr, &got)) {
        std::fprintf(stderr, "%s: dense dispatch rc=%d FAIL\n", label, rc);
        return false;
    }
    const std::vector<float> cpu = dense_reference(
        model.data + offset, x, out_dim, n_tokens);
    return close_to_cpu(got, cpu, label);
}

bool run_pair_case(const aligned_model &model, uint32_t n_tokens,
                   const char *label) {
    std::vector<float> x;
    fill_activation(&x, n_tokens);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner dense0((uint64_t)n_tokens * kM0 * sizeof(float));
    tensor_owner dense1((uint64_t)n_tokens * kM1 * sizeof(float));
    tensor_owner pair0((uint64_t)n_tokens * kM0 * sizeof(float));
    tensor_owner pair1((uint64_t)n_tokens * kM1 * sizeof(float));
    if (!x_gpu.ptr || !dense0.ptr || !dense1.ptr || !pair0.ptr || !pair1.ptr ||
        !write_tensor(x_gpu.ptr, x)) {
        std::fprintf(stderr, "%s: tensor allocation/write FAIL\n", label);
        return false;
    }
    const int dense_rc0 = ds4_gpu_matmul_quant_tensor(
        dense0.ptr, model.data, model.size, model.weight0_offset, kQ4Type,
        kK, kM0, x_gpu.ptr, n_tokens);
    const int dense_rc1 = ds4_gpu_matmul_quant_tensor(
        dense1.ptr, model.data, model.size, model.weight1_offset, kQ4Type,
        kK, kM1, x_gpu.ptr, n_tokens);
    const int pair_rc = ds4_gpu_matmul_q4_K_pair_tensor(
        pair0.ptr, pair1.ptr, model.data, model.size,
        model.weight0_offset, model.weight1_offset,
        kK, kM0, kM1, x_gpu.ptr, n_tokens);
    std::vector<float> dense0_host((uint64_t)n_tokens * kM0);
    std::vector<float> dense1_host((uint64_t)n_tokens * kM1);
    std::vector<float> pair0_host(dense0_host.size());
    std::vector<float> pair1_host(dense1_host.size());
    if (dense_rc0 == 0 || dense_rc1 == 0 || pair_rc == 0 ||
        !read_tensor(dense0.ptr, &dense0_host) ||
        !read_tensor(dense1.ptr, &dense1_host) ||
        !read_tensor(pair0.ptr, &pair0_host) ||
        !read_tensor(pair1.ptr, &pair1_host)) {
        std::fprintf(stderr,
                     "%s: dispatch/read dense=(%d,%d) pair=%d FAIL\n",
                     label, dense_rc0, dense_rc1, pair_rc);
        return false;
    }
    const std::vector<float> cpu0 = dense_reference(
        model.data + model.weight0_offset, x, kM0, n_tokens);
    const std::vector<float> cpu1 = dense_reference(
        model.data + model.weight1_offset, x, kM1, n_tokens);
    bool ok = close_to_cpu(dense0_host, cpu0, "pair control dense0 vs CPU");
    ok = close_to_cpu(dense1_host, cpu1, "pair control dense1 vs CPU") && ok;
    ok = bitwise_equal(pair0_host, dense0_host, "pair0 vs standalone dense0") && ok;
    ok = bitwise_equal(pair1_host, dense1_host, "pair1 vs standalone dense1") && ok;
    std::fprintf(stderr, "%s: %s\n", label, ok ? "PASS" : "FAIL");
    return ok;
}

bool unchanged_after_rejected_call(ds4_gpu_tensor *tensor,
                                   const std::vector<float> &sentinel,
                                   const char *label) {
    std::vector<float> after(sentinel.size());
    if (!read_tensor(tensor, &after)) {
        std::fprintf(stderr, "%s: readback FAIL\n", label);
        return false;
    }
    return bitwise_equal(after, sentinel, label);
}

bool run_dense_guards(const aligned_model &model) {
    std::vector<float> x;
    fill_activation(&x, 1u);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner out_gpu((uint64_t)kM0 * sizeof(float));
    const std::vector<float> sentinel = sentinel_values(kM0);
    if (!x_gpu.ptr || !out_gpu.ptr || !write_tensor(x_gpu.ptr, x) ||
        !write_tensor(out_gpu.ptr, sentinel)) {
        std::fprintf(stderr, "dense guards: setup FAIL\n");
        return false;
    }
    const int bad_k_rc = ds4_gpu_matmul_quant_tensor(
        out_gpu.ptr, model.data, model.size, model.weight0_offset, kQ4Type,
        kK - 1u, kM0, x_gpu.ptr, 1u);
    bool ok = bad_k_rc == 0 && unchanged_after_rejected_call(
        out_gpu.ptr, sentinel, "dense K%256 guard preserves output");
    if (bad_k_rc != 0) {
        std::fprintf(stderr, "dense K%%256 guard: expected rc=0 got=%d FAIL\n",
                     bad_k_rc);
    }
    if (!write_tensor(out_gpu.ptr, sentinel)) return false;
    const int range_rc = ds4_gpu_matmul_quant_tensor(
        out_gpu.ptr, model.data, model.size, model.size - 16u, kQ4Type,
        kK, kM0, x_gpu.ptr, 1u);
    ok = (range_rc == 0) && unchanged_after_rejected_call(
        out_gpu.ptr, sentinel, "dense model-range guard preserves output") && ok;
    if (range_rc != 0) {
        std::fprintf(stderr, "dense model-range guard: expected rc=0 got=%d FAIL\n",
                     range_rc);
    }
    return ok;
}

bool run_pair_guards(const aligned_model &model) {
    constexpr uint32_t n_tokens = 9u;
    std::vector<float> x;
    fill_activation(&x, n_tokens);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner out0((uint64_t)n_tokens * kM0 * sizeof(float));
    tensor_owner out1((uint64_t)n_tokens * kM1 * sizeof(float));
    const std::vector<float> sentinel0 = sentinel_values((uint64_t)n_tokens * kM0);
    const std::vector<float> sentinel1 = sentinel_values((uint64_t)n_tokens * kM1);
    if (!x_gpu.ptr || !out0.ptr || !out1.ptr || !write_tensor(x_gpu.ptr, x) ||
        !write_tensor(out0.ptr, sentinel0) || !write_tensor(out1.ptr, sentinel1)) {
        std::fprintf(stderr, "pair guards: setup FAIL\n");
        return false;
    }
    const int rc = ds4_gpu_matmul_q4_K_pair_tensor(
        out0.ptr, out1.ptr, model.data, model.size,
        model.weight0_offset, model.weight1_offset,
        kK, kM0, kM1, x_gpu.ptr, n_tokens);
    bool ok = rc == 0;
    if (rc != 0) {
        std::fprintf(stderr, "pair n_tok=9 guard: expected rc=0 got=%d FAIL\n", rc);
    }
    ok = unchanged_after_rejected_call(
             out0.ptr, sentinel0, "pair n_tok=9 preserves out0") && ok;
    ok = unchanged_after_rejected_call(
             out1.ptr, sentinel1, "pair n_tok=9 preserves out1") && ok;
    return ok;
}

bool run_pair_opt_in_guards(const aligned_model &model) {
    constexpr uint32_t n_tokens = 1u;
    std::vector<float> x;
    fill_activation(&x, n_tokens);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner out0((uint64_t)n_tokens * kM0 * sizeof(float));
    tensor_owner out1((uint64_t)n_tokens * kM1 * sizeof(float));
    const std::vector<float> sentinel0 = sentinel_values(kM0);
    const std::vector<float> sentinel1 = sentinel_values(kM1);
    if (!x_gpu.ptr || !out0.ptr || !out1.ptr || !write_tensor(x_gpu.ptr, x)) {
        std::fprintf(stderr, "pair opt-in guards: setup FAIL\n");
        return false;
    }

    auto rejected_call = [&](const char *label) {
        if (!write_tensor(out0.ptr, sentinel0) ||
            !write_tensor(out1.ptr, sentinel1)) {
            return false;
        }
        const int rc = ds4_gpu_matmul_q4_K_pair_tensor(
            out0.ptr, out1.ptr, model.data, model.size,
            model.weight0_offset, model.weight1_offset,
            kK, kM0, kM1, x_gpu.ptr, n_tokens);
        bool guard_ok = rc == 0;
        if (rc != 0) {
            std::fprintf(stderr, "%s: expected rc=0 got=%d FAIL\n", label, rc);
        }
        guard_ok = unchanged_after_rejected_call(out0.ptr, sentinel0, label) &&
                   guard_ok;
        guard_ok = unchanged_after_rejected_call(out1.ptr, sentinel1, label) &&
                   guard_ok;
        return guard_ok;
    };

    env_snapshot enable("DS4_ROCM_ENABLE_Q4_DENSE_PAIR");
    env_snapshot disable("DS4_ROCM_DISABLE_Q4_DENSE_PAIR");
    (void)unsetenv("DS4_ROCM_ENABLE_Q4_DENSE_PAIR");
    (void)unsetenv("DS4_ROCM_DISABLE_Q4_DENSE_PAIR");
    bool ok = rejected_call("pair disabled-by-default preserves outputs");
    (void)setenv("DS4_ROCM_ENABLE_Q4_DENSE_PAIR", "1", 1);
    (void)setenv("DS4_ROCM_DISABLE_Q4_DENSE_PAIR", "1", 1);
    ok = rejected_call("pair DISABLE dominates ENABLE") && ok;
    return ok;
}

int detect_rocm_device() {
#if DS4_TEST_HAS_HIP_RUNTIME
    int count = 0;
    const hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count <= 0) {
        std::fprintf(stderr,
                     "ROCm Q4 dense/pair: SKIP (HIP runtime has no visible device: %s)\n",
                     err == hipSuccess ? "device count is zero" : hipGetErrorString(err));
        return 0;
    }
    return count;
#else
    std::fprintf(stderr,
                 "ROCm Q4 dense/pair: SKIP (compiled without HIP runtime headers)\n");
    return 0;
#endif
}

}  // namespace

int main(int argc, char **argv) {
    bool run_dense = true;
    bool run_pair = true;
    if (argc == 2 && std::strcmp(argv[1], "--dense") == 0) {
        run_pair = false;
    } else if (argc == 2 && std::strcmp(argv[1], "--pair") == 0) {
        run_dense = false;
    } else if (argc > 1 &&
               !(argc == 2 && std::strcmp(argv[1], "--all") == 0)) {
        std::fprintf(stderr, "usage: %s [--all|--dense|--pair]\n", argv[0]);
        return 2;
    }

    if (detect_rocm_device() <= 0) {
        const char *require_device =
            std::getenv("DS4_TEST_REQUIRE_ROCM_DEVICE");
        const bool required = require_device && require_device[0] != '\0' &&
                              std::strcmp(require_device, "0") != 0;
        return required ? 1 : kSkip;
    }
    if (!ds4_gpu_init()) {
        std::fprintf(stderr,
                     "ROCm Q4 dense/pair: FAIL (device is visible but ds4_gpu_init failed)\n");
        return 1;
    }

    aligned_model model;
    bool ok = make_model(&model);
    if (!ok) {
        std::fprintf(stderr, "ROCm Q4 dense/pair: fixture allocation FAIL\n");
    } else if (!ds4_gpu_set_model_map(model.data, model.size)) {
        std::fprintf(stderr, "ROCm Q4 dense/pair: model-map registration FAIL\n");
        ok = false;
    }

    const bool model_ready = ok;
    if (model_ready && run_dense) {
        std::fprintf(stderr, "ROCm Q4 dense oracle (raw GGUF Q4_K x Q8_K):\n");
        ok = run_dense_case(model, 1u, model.weight0_offset, kM0,
                            "dense n_tok=1") && ok;
        ok = run_dense_case(model, 3u, model.weight1_offset, kM1,
                            "dense n_tok=3") && ok;
        ok = run_dense_case(model, 9u, model.weight1_offset, kM1,
                            "dense n_tok=9") && ok;
        ok = run_dense_case(model, 128u, model.weight1_offset, kM1,
                            "dense n_tok=128") && ok;
        ok = run_dense_guards(model) && ok;
    }
    if (model_ready && run_pair) {
        std::fprintf(stderr, "ROCm Q4 pair parity (pair vs two dense):\n");
        env_snapshot enable("DS4_ROCM_ENABLE_Q4_DENSE_PAIR");
        env_snapshot disable("DS4_ROCM_DISABLE_Q4_DENSE_PAIR");
        (void)setenv("DS4_ROCM_ENABLE_Q4_DENSE_PAIR", "1", 1);
        (void)unsetenv("DS4_ROCM_DISABLE_Q4_DENSE_PAIR");
        const bool pair1_ok = run_pair_case(model, 1u, "pair n_tok=1");
        const bool pair3_ok = run_pair_case(model, 3u, "pair n_tok=3");
        const bool pair8_ok = run_pair_case(model, 8u, "pair n_tok=8");
        const bool pair_guard_ok = run_pair_guards(model);
        const bool pair_opt_in_ok = run_pair_opt_in_guards(model);
        ok = pair1_ok && pair3_ok && pair8_ok && pair_guard_ok &&
             pair_opt_in_ok && ok;
    }

    // Registered host ranges must be released before their aligned backing
    // allocation is destroyed.
    ds4_gpu_cleanup();
    std::fprintf(stderr, "ROCm Q4 dense/pair oracle: %s\n",
                 ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
