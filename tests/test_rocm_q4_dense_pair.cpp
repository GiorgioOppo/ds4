// SPDX-License-Identifier: MIT
// Deterministic ROCm Q4_K dense/pair/tiled-prefill oracle.
//
// The test deliberately goes through the public tensor/model-map API.  Weight
// rows use the raw 144-byte GGUF Q4_K layout, while the CPU reference mirrors
// the backend's F32 -> Q8_K quantizer and Q4_K x Q8_K integer dot product.
// Prefill controls are forced through the rollback path before the TILE8
// REQUIRE path so a future default promotion cannot turn parity into a
// candidate-vs-candidate false green.

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
constexpr uint32_t kQ8Type = 8u;
constexpr uint32_t kTailK = 1024u;
constexpr uint32_t kAttnGroupDim = 4096u;
constexpr uint32_t kAttnRank = 32u;
constexpr uint32_t kAttnGroups = 8u;
constexpr uint32_t kAttnLowDim = kAttnGroups * kAttnRank;
constexpr uint32_t kAttnOutDim = 65u;
constexpr uint32_t kDecodeAttnGroupDim = 4096u;
constexpr uint32_t kDecodeAttnRank = 1024u;
constexpr uint32_t kDecodeAttnGroups = 8u;
constexpr uint32_t kDecodeAttnLowDim =
    kDecodeAttnGroups * kDecodeAttnRank;
constexpr size_t kOutputGuardFloats = 257u;
constexpr float kCpuAbsTolerance = 2.0e-3f;
constexpr float kCpuRelTolerance = 3.0e-5f;
constexpr int kSkip = 77;

constexpr const char *kPrefillEnable =
    "DS4_ROCM_ENABLE_Q4_PREFILL_TILE8";
constexpr const char *kPrefillDisable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_TILE8";
constexpr const char *kPrefillRequire =
    "DS4_ROCM_REQUIRE_Q4_PREFILL_TILE8";
constexpr const char *kGroupedDecodeEnable =
    "DS4_ROCM_ENABLE_Q4_GROUPED_ATTN_A";
constexpr const char *kGroupedDecodeDisable =
    "DS4_ROCM_DISABLE_Q4_GROUPED_ATTN_A";
constexpr const char *kGroupedDecodeRequire =
    "DS4_ROCM_REQUIRE_Q4_GROUPED_ATTN_A";
constexpr const char *kGroupedDecodeStats =
    "DS4_ROCM_Q4_GROUPED_ATTN_A_STATS";

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

struct block_q8_0_test {
    uint16_t d;
    int8_t qs[32];
};

static_assert(sizeof(block_q4_K_test) == 144u,
              "Q4_K fixture must match the raw GGUF layout");
static_assert(sizeof(block_q8_K_test) == 292u,
              "Q8_K oracle must match the ROCm activation layout");
static_assert(sizeof(block_q8_0_test) == 34u,
              "Q8_0 fixture must match the raw GGUF layout");

struct tensor_owner {
    ds4_gpu_tensor *ptr = nullptr;

    explicit tensor_owner(uint64_t bytes) : ptr(ds4_gpu_tensor_alloc(bytes)) {}
    explicit tensor_owner(ds4_gpu_tensor *owned) : ptr(owned) {}
    ~tensor_owner() { ds4_gpu_tensor_free(ptr); }

    tensor_owner(const tensor_owner &) = delete;
    tensor_owner &operator=(const tensor_owner &) = delete;
};

struct aligned_model {
    uint8_t *data = nullptr;
    uint64_t size = 0;
    uint64_t weight0_offset = 0;
    uint64_t weight1_offset = 0;
    uint64_t attn_a_offset = 0;
    uint64_t decode_attn_a_offset = 0;
    uint64_t attn_b_offset = 0;
    uint64_t attn_b_q8_offset = 0;
    uint64_t tail_k1024_offset = 0;

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

void fill_q4_rows(block_q4_K_test *rows, uint32_t n_rows,
                  uint32_t in_dim, uint32_t seed) {
    uint32_t state = seed;
    const uint32_t blocks_per_row = in_dim / kQkK;
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

void fill_q8_0_rows(block_q8_0_test *rows, uint32_t n_rows,
                    uint32_t in_dim, uint32_t seed) {
    uint32_t state = seed;
    const uint32_t blocks_per_row = in_dim / 32u;
    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            block_q8_0_test &block =
                rows[(uint64_t)row * blocks_per_row + b];
            const float scale = 0.0015f +
                0.000125f * (float)(1u + (lcg_next(state) % 29u));
            block.d = float_to_fp16(scale);
            for (int8_t &q : block.qs) {
                q = (int8_t)((int)(lcg_next(state) % 255u) - 127);
            }
        }
    }
}

bool make_model(aligned_model *model) {
    constexpr uint64_t page = 4096u;
    const uint64_t row_bytes = (kK / kQkK) * sizeof(block_q4_K_test);
    const uint64_t weight0_bytes = kM0 * row_bytes;
    const uint64_t weight1_bytes = kM1 * row_bytes;
    const uint64_t attn_a_row_bytes =
        (kAttnGroupDim / kQkK) * sizeof(block_q4_K_test);
    const uint64_t attn_a_bytes =
        (uint64_t)kAttnGroups * kAttnRank * attn_a_row_bytes;
    const uint64_t decode_attn_a_row_bytes =
        (kDecodeAttnGroupDim / kQkK) * sizeof(block_q4_K_test);
    const uint64_t decode_attn_a_group_bytes =
        (uint64_t)kDecodeAttnRank * decode_attn_a_row_bytes;
    const uint64_t decode_attn_a_bytes =
        (uint64_t)kDecodeAttnGroups * decode_attn_a_group_bytes;
    const uint64_t attn_b_row_bytes =
        (kAttnLowDim / kQkK) * sizeof(block_q4_K_test);
    const uint64_t attn_b_bytes =
        (uint64_t)kAttnOutDim * attn_b_row_bytes;
    const uint64_t tail_row_bytes =
        (kTailK / kQkK) * sizeof(block_q4_K_test);
    const uint64_t tail_bytes = (uint64_t)kM0 * tail_row_bytes;
    const uint64_t attn_b_q8_row_bytes =
        (kAttnLowDim / 32u) * sizeof(block_q8_0_test);
    const uint64_t attn_b_q8_bytes =
        (uint64_t)kAttnOutDim * attn_b_q8_row_bytes;
    model->weight0_offset = 0u;
    model->weight1_offset = round_up(weight0_bytes, page);
    model->attn_a_offset = round_up(
        model->weight1_offset + weight1_bytes, page);
    model->decode_attn_a_offset = round_up(
        model->attn_a_offset + attn_a_bytes, page);
    model->attn_b_offset = round_up(
        model->decode_attn_a_offset + decode_attn_a_bytes, page);
    model->tail_k1024_offset = round_up(
        model->attn_b_offset + attn_b_bytes, page);
    model->attn_b_q8_offset = round_up(
        model->tail_k1024_offset + tail_bytes, page);
    model->size = round_up(
        model->attn_b_q8_offset + attn_b_q8_bytes, page);
    void *storage = nullptr;
    if (posix_memalign(&storage, (size_t)page, (size_t)model->size) != 0) {
        return false;
    }
    model->data = static_cast<uint8_t *>(storage);
    std::memset(model->data, 0xa5, (size_t)model->size);
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->weight0_offset),
                 kM0, kK, 0x41c64e6du);
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->weight1_offset),
                 kM1, kK, 0x9e3779b9u);
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->attn_a_offset),
                 kAttnGroups * kAttnRank, kAttnGroupDim, 0x243f6a88u);
    for (uint32_t group = 0; group < kDecodeAttnGroups; group++) {
        fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                         model->data + model->decode_attn_a_offset +
                         (uint64_t)group * decode_attn_a_group_bytes),
                     kDecodeAttnRank, kDecodeAttnGroupDim,
                     0xd1b54a35u ^ (group * 0x9e3779b9u));
    }
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->attn_b_offset),
                 kAttnOutDim, kAttnLowDim, 0x85a308d3u);
    fill_q4_rows(reinterpret_cast<block_q4_K_test *>(
                     model->data + model->tail_k1024_offset),
                 kM0, kTailK, 0x13198a2eu);
    fill_q8_0_rows(reinterpret_cast<block_q8_0_test *>(
                       model->data + model->attn_b_q8_offset),
                   kAttnOutDim, kAttnLowDim, 0x03707344u);
    return true;
}

void fill_activation(std::vector<float> *x, uint32_t n_tokens,
                     uint32_t in_dim = kK) {
    x->resize((uint64_t)n_tokens * in_dim);
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t b = 0; b < in_dim / kQkK; b++) {
            float *block =
                x->data() + (uint64_t)token * in_dim + b * kQkK;
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
                                   uint32_t n_tokens,
                                   uint32_t in_dim = kK) {
    const auto *weights = reinterpret_cast<const block_q4_K_test *>(weight_base);
    const uint32_t blocks_per_row = in_dim / kQkK;
    std::vector<block_q8_K_test> xq((uint64_t)n_tokens * blocks_per_row);
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            quantize_q8_K_cpu(
                x.data() + (uint64_t)token * in_dim + b * kQkK,
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

bool output_guard_unchanged(const std::vector<float> &values,
                            const std::vector<float> &sentinel,
                            size_t logical_count,
                            const char *label) {
    if (values.size() != sentinel.size() ||
        logical_count > values.size()) {
        std::fprintf(stderr, "%s: invalid guard geometry FAIL\n", label);
        return false;
    }
    uint64_t mismatches = 0;
    size_t first = logical_count;
    for (size_t i = logical_count; i < values.size(); i++) {
        if (std::memcmp(&values[i], &sentinel[i], sizeof(float)) != 0) {
            if (mismatches == 0) first = i;
            mismatches++;
        }
    }
    std::fprintf(stderr, "%s: mismatches=%llu/%zu %s\n",
                 label, (unsigned long long)mismatches,
                 values.size() - logical_count,
                 mismatches == 0 ? "PASS" : "FAIL");
    if (mismatches != 0) {
        std::fprintf(stderr, "  first guard overwrite at float %zu\n", first);
    }
    return mismatches == 0;
}

bool run_prefill_parity_case(const aligned_model &model, uint32_t n_tokens,
                             uint64_t offset, uint32_t out_dim,
                             bool compare_cpu, const char *label,
                             uint32_t in_dim = kK) {
    std::vector<float> x;
    fill_activation(&x, n_tokens, in_dim);
    const size_t logical_count = (size_t)n_tokens * out_dim;
    const size_t allocation_count = logical_count + kOutputGuardFloats;
    const std::vector<float> sentinel = sentinel_values(allocation_count);

    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner legacy_gpu(allocation_count * sizeof(float));
    tensor_owner candidate_gpu(allocation_count * sizeof(float));
    if (!x_gpu.ptr || !legacy_gpu.ptr || !candidate_gpu.ptr ||
        !write_tensor(x_gpu.ptr, x) ||
        !write_tensor(legacy_gpu.ptr, sentinel) ||
        !write_tensor(candidate_gpu.ptr, sentinel)) {
        std::fprintf(stderr, "%s: tensor allocation/write FAIL\n", label);
        return false;
    }

    env_snapshot enable(kPrefillEnable);
    env_snapshot disable(kPrefillDisable);
    env_snapshot require(kPrefillRequire);

    // The authoritative rollback remains the reference now that the tiled
    // path is default-on.
    (void)unsetenv(kPrefillEnable);
    (void)setenv(kPrefillDisable, "1", 1);
    (void)unsetenv(kPrefillRequire);
    const int legacy_rc = ds4_gpu_matmul_quant_tensor(
        legacy_gpu.ptr, model.data, model.size, offset, kQ4Type,
        in_dim, out_dim, x_gpu.ptr, n_tokens);

    // TILE8 is default-on.  Leave the legacy ENABLE unset and use REQUIRE so
    // a silently ineligible default cannot compare the legacy kernel with
    // itself.
    (void)unsetenv(kPrefillEnable);
    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    const int candidate_rc = ds4_gpu_matmul_quant_tensor(
        candidate_gpu.ptr, model.data, model.size, offset, kQ4Type,
        in_dim, out_dim, x_gpu.ptr, n_tokens);

    std::vector<float> legacy_all(allocation_count);
    std::vector<float> candidate_all(allocation_count);
    if (legacy_rc == 0 || candidate_rc == 0 ||
        !read_tensor(legacy_gpu.ptr, &legacy_all) ||
        !read_tensor(candidate_gpu.ptr, &candidate_all)) {
        std::fprintf(stderr,
                     "%s: dispatch/read legacy=%d candidate=%d FAIL\n",
                     label, legacy_rc, candidate_rc);
        return false;
    }

    bool ok = output_guard_unchanged(
        legacy_all, sentinel, logical_count, "prefill legacy output canary");
    ok = output_guard_unchanged(
             candidate_all, sentinel, logical_count,
             "prefill candidate output canary") && ok;

    legacy_all.resize(logical_count);
    candidate_all.resize(logical_count);
    ok = bitwise_equal(candidate_all, legacy_all,
                       "prefill candidate vs forced legacy") && ok;
    if (compare_cpu) {
        const std::vector<float> cpu = dense_reference(
            model.data + offset, x, out_dim, n_tokens, in_dim);
        ok = close_to_cpu(legacy_all, cpu,
                          "prefill forced legacy vs CPU") && ok;
        ok = close_to_cpu(candidate_all, cpu,
                          "prefill candidate vs CPU") && ok;
    }
    std::fprintf(stderr,
                 "%s: legacy_rc=%d candidate_rc=%d logical=%zu guard=%zu %s\n",
                 label, legacy_rc, candidate_rc, logical_count,
                 kOutputGuardFloats, ok ? "PASS" : "FAIL");
    return ok;
}

bool run_prefill_gate_guards(const aligned_model &model) {
    constexpr uint32_t n_tokens = 9u;
    std::vector<float> x;
    fill_activation(&x, n_tokens);
    const size_t output_count = (size_t)n_tokens * kM1 + kOutputGuardFloats;
    const std::vector<float> sentinel = sentinel_values(output_count);
    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner out_gpu(output_count * sizeof(float));
    if (!x_gpu.ptr || !out_gpu.ptr || !write_tensor(x_gpu.ptr, x) ||
        !write_tensor(out_gpu.ptr, sentinel)) {
        std::fprintf(stderr, "prefill gate guards: setup FAIL\n");
        return false;
    }

    env_snapshot enable(kPrefillEnable);
    env_snapshot disable(kPrefillDisable);
    env_snapshot require(kPrefillRequire);

    /* REQUIRE with neither ENABLE nor DISABLE proves that TILE8 is selected
     * by the default policy. */
    (void)unsetenv(kPrefillEnable);
    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    const int default_rc = ds4_gpu_matmul_quant_tensor(
        out_gpu.ptr, model.data, model.size, model.weight1_offset, kQ4Type,
        kK, kM1, x_gpu.ptr, n_tokens);
    bool ok = default_rc != 0;
    if (default_rc == 0) {
        std::fprintf(stderr,
                     "prefill default-on REQUIRE: expected success got=%d FAIL\n",
                     default_rc);
    }
    std::vector<float> default_out(output_count);
    if (!read_tensor(out_gpu.ptr, &default_out)) return false;
    ok = output_guard_unchanged(
             default_out, sentinel, (size_t)n_tokens * kM1,
             "prefill default-on output canary") && ok;
    if (!write_tensor(out_gpu.ptr, sentinel)) return false;

    (void)setenv(kPrefillEnable, "1", 1);
    (void)setenv(kPrefillDisable, "1", 1);
    (void)setenv(kPrefillRequire, "1", 1);
    const int rc = ds4_gpu_matmul_quant_tensor(
        out_gpu.ptr, model.data, model.size, model.weight1_offset, kQ4Type,
        kK, kM1, x_gpu.ptr, n_tokens);
    ok = rc == 0 && ok;
    if (rc != 0) {
        std::fprintf(stderr,
                     "prefill DISABLE+REQUIRE: expected rc=0 got=%d FAIL\n",
                     rc);
    }
    ok = unchanged_after_rejected_call(
             out_gpu.ptr, sentinel,
             "prefill DISABLE dominates REQUIRE and preserves output") && ok;
    return ok;
}

bool run_prefill_pair_case(const aligned_model &model) {
    constexpr uint32_t n_tokens = 128u;
    std::vector<float> x;
    fill_activation(&x, n_tokens);
    const size_t count0 = (size_t)n_tokens * kM0;
    const size_t count1 = (size_t)n_tokens * kM1;
    const std::vector<float> sentinel0 =
        sentinel_values(count0 + kOutputGuardFloats);
    const std::vector<float> sentinel1 =
        sentinel_values(count1 + kOutputGuardFloats);

    tensor_owner x_gpu(x.size() * sizeof(float));
    tensor_owner legacy0(sentinel0.size() * sizeof(float));
    tensor_owner legacy1(sentinel1.size() * sizeof(float));
    tensor_owner pair0(sentinel0.size() * sizeof(float));
    tensor_owner pair1(sentinel1.size() * sizeof(float));
    if (!x_gpu.ptr || !legacy0.ptr || !legacy1.ptr || !pair0.ptr ||
        !pair1.ptr || !write_tensor(x_gpu.ptr, x) ||
        !write_tensor(legacy0.ptr, sentinel0) ||
        !write_tensor(legacy1.ptr, sentinel1) ||
        !write_tensor(pair0.ptr, sentinel0) ||
        !write_tensor(pair1.ptr, sentinel1)) {
        std::fprintf(stderr, "prefill pair n_tok=128: setup FAIL\n");
        return false;
    }

    env_snapshot prefill_enable(kPrefillEnable);
    env_snapshot prefill_disable(kPrefillDisable);
    env_snapshot prefill_require(kPrefillRequire);
    env_snapshot pair_enable("DS4_ROCM_ENABLE_Q4_DENSE_PAIR");
    env_snapshot pair_disable("DS4_ROCM_DISABLE_Q4_DENSE_PAIR");

    (void)unsetenv(kPrefillEnable);
    (void)setenv(kPrefillDisable, "1", 1);
    (void)unsetenv(kPrefillRequire);
    const int legacy_rc0 = ds4_gpu_matmul_quant_tensor(
        legacy0.ptr, model.data, model.size, model.weight0_offset, kQ4Type,
        kK, kM0, x_gpu.ptr, n_tokens);
    const int legacy_rc1 = ds4_gpu_matmul_quant_tensor(
        legacy1.ptr, model.data, model.size, model.weight1_offset, kQ4Type,
        kK, kM1, x_gpu.ptr, n_tokens);

    // The prefill pair is a distinct path: it must not depend on the legacy
    // decode-pair opt-in, whose <=8-token behavior is tested separately.
    (void)unsetenv(kPrefillEnable);
    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    (void)unsetenv("DS4_ROCM_ENABLE_Q4_DENSE_PAIR");
    (void)unsetenv("DS4_ROCM_DISABLE_Q4_DENSE_PAIR");
    const int pair_rc = ds4_gpu_matmul_q4_K_pair_tensor(
        pair0.ptr, pair1.ptr, model.data, model.size,
        model.weight0_offset, model.weight1_offset,
        kK, kM0, kM1, x_gpu.ptr, n_tokens);

    std::vector<float> legacy0_host(sentinel0.size());
    std::vector<float> legacy1_host(sentinel1.size());
    std::vector<float> pair0_host(sentinel0.size());
    std::vector<float> pair1_host(sentinel1.size());
    if (legacy_rc0 == 0 || legacy_rc1 == 0 || pair_rc == 0 ||
        !read_tensor(legacy0.ptr, &legacy0_host) ||
        !read_tensor(legacy1.ptr, &legacy1_host) ||
        !read_tensor(pair0.ptr, &pair0_host) ||
        !read_tensor(pair1.ptr, &pair1_host)) {
        std::fprintf(stderr,
                     "prefill pair n_tok=128: dispatch/read legacy=(%d,%d) "
                     "pair=%d FAIL\n",
                     legacy_rc0, legacy_rc1, pair_rc);
        return false;
    }

    bool ok = output_guard_unchanged(
        pair0_host, sentinel0, count0, "prefill pair0 output canary");
    ok = output_guard_unchanged(
             pair1_host, sentinel1, count1,
             "prefill pair1 output canary") && ok;
    pair0_host.resize(count0);
    pair1_host.resize(count1);
    legacy0_host.resize(count0);
    legacy1_host.resize(count1);
    ok = bitwise_equal(pair0_host, legacy0_host,
                       "prefill pair0 vs forced legacy dense0") && ok;
    ok = bitwise_equal(pair1_host, legacy1_host,
                       "prefill pair1 vs forced legacy dense1") && ok;
    std::fprintf(stderr,
                 "prefill pair K=4096 M=(65,33) n_tok=128 "
                 "legacy=(%d,%d) pair=%d %s\n",
                 legacy_rc0, legacy_rc1, pair_rc, ok ? "PASS" : "FAIL");
    return ok;
}

bool run_attention_rowwise_reference(const aligned_model &model,
                                     const ds4_gpu_tensor *heads,
                                     ds4_gpu_tensor *low,
                                     ds4_gpu_tensor *out,
                                     uint32_t n_tokens,
                                     uint64_t out_b_offset,
                                     uint32_t out_b_type) {
    const uint64_t heads_group_bytes =
        (uint64_t)kAttnGroupDim * sizeof(float);
    const uint64_t heads_token_bytes =
        (uint64_t)kAttnGroups * heads_group_bytes;
    const uint64_t low_group_bytes =
        (uint64_t)kAttnRank * sizeof(float);
    const uint64_t low_token_bytes =
        (uint64_t)kAttnLowDim * sizeof(float);
    const uint64_t out_token_bytes =
        (uint64_t)kAttnOutDim * sizeof(float);
    const uint64_t row_a_bytes =
        (kAttnGroupDim / kQkK) * sizeof(block_q4_K_test);
    const uint64_t group_a_bytes = (uint64_t)kAttnRank * row_a_bytes;

    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t group = 0; group < kAttnGroups; group++) {
            tensor_owner heads_group(ds4_gpu_tensor_view(
                heads,
                (uint64_t)token * heads_token_bytes +
                    (uint64_t)group * heads_group_bytes,
                heads_group_bytes));
            tensor_owner low_group(ds4_gpu_tensor_view(
                low,
                (uint64_t)token * low_token_bytes +
                    (uint64_t)group * low_group_bytes,
                low_group_bytes));
            if (!heads_group.ptr || !low_group.ptr ||
                ds4_gpu_matmul_quant_tensor(
                    low_group.ptr, model.data, model.size,
                    model.attn_a_offset + (uint64_t)group * group_a_bytes,
                    kQ4Type, kAttnGroupDim, kAttnRank,
                    heads_group.ptr, 1u) == 0) {
                std::fprintf(stderr,
                             "attention row reference A token=%u group=%u FAIL\n",
                             token, group);
                return false;
            }
        }
        tensor_owner low_row(ds4_gpu_tensor_view(
            low, (uint64_t)token * low_token_bytes, low_token_bytes));
        tensor_owner out_row(ds4_gpu_tensor_view(
            out, (uint64_t)token * out_token_bytes, out_token_bytes));
        if (!low_row.ptr || !out_row.ptr ||
            ds4_gpu_matmul_quant_tensor(
                out_row.ptr, model.data, model.size, out_b_offset,
                out_b_type, kAttnLowDim, kAttnOutDim,
                low_row.ptr, 1u) == 0) {
            std::fprintf(stderr,
                         "attention row reference B token=%u FAIL\n", token);
            return false;
        }
    }
    return true;
}

bool run_attention_prefill_case(const aligned_model &model,
                                uint32_t n_tokens,
                                const char *label,
                                uint32_t out_b_type = kQ4Type) {
    if (out_b_type != kQ4Type && out_b_type != kQ8Type) {
        std::fprintf(stderr, "%s: unsupported output-B type %u FAIL\n",
                     label, out_b_type);
        return false;
    }
    const uint64_t out_b_offset = out_b_type == kQ8Type
        ? model.attn_b_q8_offset : model.attn_b_offset;
    const size_t heads_count =
        (size_t)n_tokens * kAttnGroups * kAttnGroupDim;
    const size_t low_count = (size_t)n_tokens * kAttnLowDim;
    const size_t out_count = (size_t)n_tokens * kAttnOutDim;
    const size_t group_tmp_count = (size_t)n_tokens * kAttnGroupDim;
    const size_t low_tmp_count = (size_t)n_tokens * kAttnRank;
    std::vector<float> heads_host;
    fill_activation(&heads_host, n_tokens * kAttnGroups);
    const std::vector<float> low_sentinel =
        sentinel_values(low_count + kOutputGuardFloats);
    const std::vector<float> out_sentinel =
        sentinel_values(out_count + kOutputGuardFloats);
    const std::vector<float> group_tmp_sentinel =
        sentinel_values(group_tmp_count + kOutputGuardFloats);
    const std::vector<float> low_tmp_sentinel =
        sentinel_values(low_tmp_count + kOutputGuardFloats);

    tensor_owner heads_gpu(heads_count * sizeof(float));
    tensor_owner reference_low(low_sentinel.size() * sizeof(float));
    tensor_owner reference_out(out_sentinel.size() * sizeof(float));
    tensor_owner candidate_low(low_sentinel.size() * sizeof(float));
    tensor_owner candidate_out(out_sentinel.size() * sizeof(float));
    tensor_owner group_tmp(group_tmp_sentinel.size() * sizeof(float));
    tensor_owner low_tmp(low_tmp_sentinel.size() * sizeof(float));
    if (!heads_gpu.ptr || !reference_low.ptr || !reference_out.ptr ||
        !candidate_low.ptr || !candidate_out.ptr || !group_tmp.ptr ||
        !low_tmp.ptr || !write_tensor(heads_gpu.ptr, heads_host) ||
        !write_tensor(reference_low.ptr, low_sentinel) ||
        !write_tensor(reference_out.ptr, out_sentinel) ||
        !write_tensor(candidate_low.ptr, low_sentinel) ||
        !write_tensor(candidate_out.ptr, out_sentinel) ||
        !write_tensor(group_tmp.ptr, group_tmp_sentinel) ||
        !write_tensor(low_tmp.ptr, low_tmp_sentinel)) {
        std::fprintf(stderr, "%s: setup FAIL\n", label);
        return false;
    }

    env_snapshot enable(kPrefillEnable);
    env_snapshot disable(kPrefillDisable);
    env_snapshot require(kPrefillRequire);
    (void)unsetenv(kPrefillEnable);
    (void)setenv(kPrefillDisable, "1", 1);
    (void)unsetenv(kPrefillRequire);
    if (!run_attention_rowwise_reference(
            model, heads_gpu.ptr, reference_low.ptr, reference_out.ptr,
            n_tokens, out_b_offset, out_b_type)) {
        std::fprintf(stderr, "%s: row-wise reference FAIL\n", label);
        return false;
    }

    (void)unsetenv(kPrefillEnable);
    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    const int candidate_rc = ds4_gpu_attention_output_q4_K_batch_tensor(
        candidate_out.ptr, candidate_low.ptr, group_tmp.ptr, low_tmp.ptr,
        model.data, model.size, model.attn_a_offset, out_b_offset,
        out_b_type, kAttnGroupDim, kAttnRank, kAttnGroups, kAttnOutDim,
        heads_gpu.ptr, n_tokens);

    std::vector<float> reference_low_host(low_sentinel.size());
    std::vector<float> reference_out_host(out_sentinel.size());
    std::vector<float> candidate_low_host(low_sentinel.size());
    std::vector<float> candidate_out_host(out_sentinel.size());
    std::vector<float> group_tmp_host(group_tmp_sentinel.size());
    std::vector<float> low_tmp_host(low_tmp_sentinel.size());
    if (candidate_rc != 1 ||
        !read_tensor(reference_low.ptr, &reference_low_host) ||
        !read_tensor(reference_out.ptr, &reference_out_host) ||
        !read_tensor(candidate_low.ptr, &candidate_low_host) ||
        !read_tensor(candidate_out.ptr, &candidate_out_host) ||
        !read_tensor(group_tmp.ptr, &group_tmp_host) ||
        !read_tensor(low_tmp.ptr, &low_tmp_host)) {
        std::fprintf(stderr, "%s: candidate dispatch/read rc=%d FAIL\n",
                     label, candidate_rc);
        return false;
    }

    bool ok = output_guard_unchanged(
        candidate_low_host, low_sentinel, low_count,
        "attention candidate low canary");
    ok = output_guard_unchanged(
             candidate_out_host, out_sentinel, out_count,
             "attention candidate out canary") && ok;
    ok = output_guard_unchanged(
             group_tmp_host, group_tmp_sentinel, group_tmp_count,
             "attention group scratch canary") && ok;
    ok = output_guard_unchanged(
             low_tmp_host, low_tmp_sentinel, low_tmp_count,
             "attention low scratch canary") && ok;
    reference_low_host.resize(low_count);
    reference_out_host.resize(out_count);
    candidate_low_host.resize(low_count);
    candidate_out_host.resize(out_count);
    ok = bitwise_equal(candidate_low_host, reference_low_host,
                       "attention candidate low vs 8x row-wise A") && ok;
    ok = bitwise_equal(candidate_out_host, reference_out_host,
                       "attention candidate out vs row-wise B") && ok;

    // The batch API promises -1 for a REQUIRE diagnostic so the graph does
    // not replay its row fallback after a forced-candidate failure.
    if (!write_tensor(candidate_low.ptr, low_sentinel) ||
        !write_tensor(candidate_out.ptr, out_sentinel) ||
        !write_tensor(group_tmp.ptr, group_tmp_sentinel) ||
        !write_tensor(low_tmp.ptr, low_tmp_sentinel)) {
        return false;
    }
    (void)setenv(kPrefillDisable, "1", 1);
    const int rejected_rc = ds4_gpu_attention_output_q4_K_batch_tensor(
        candidate_out.ptr, candidate_low.ptr, group_tmp.ptr, low_tmp.ptr,
        model.data, model.size, model.attn_a_offset, out_b_offset,
        out_b_type, kAttnGroupDim, kAttnRank, kAttnGroups, kAttnOutDim,
        heads_gpu.ptr, n_tokens);
    if (rejected_rc != -1) {
        std::fprintf(stderr,
                     "%s: DISABLE+REQUIRE expected rc=-1 got=%d FAIL\n",
                     label, rejected_rc);
        ok = false;
    }
    ok = unchanged_after_rejected_call(
             candidate_low.ptr, low_sentinel,
             "attention rejected call preserves low") && ok;
    ok = unchanged_after_rejected_call(
             candidate_out.ptr, out_sentinel,
             "attention rejected call preserves out") && ok;
    ok = unchanged_after_rejected_call(
             group_tmp.ptr, group_tmp_sentinel,
             "attention rejected call preserves group scratch") && ok;
    ok = unchanged_after_rejected_call(
             low_tmp.ptr, low_tmp_sentinel,
             "attention rejected call preserves low scratch") && ok;
    std::fprintf(stderr,
                 "%s: candidate_rc=%d rejected_rc=%d %s\n",
                 label, candidate_rc, rejected_rc, ok ? "PASS" : "FAIL");
    return ok;
}

bool run_grouped_attention_decode_case(const aligned_model &model) {
    const uint64_t row_bytes =
        (kDecodeAttnGroupDim / kQkK) * sizeof(block_q4_K_test);
    const uint64_t group_weight_bytes =
        (uint64_t)kDecodeAttnRank * row_bytes;
    const size_t heads_count =
        (size_t)kDecodeAttnGroups * kDecodeAttnGroupDim;
    const size_t logical_count = kDecodeAttnLowDim;
    const size_t allocation_count = logical_count + kOutputGuardFloats;
    std::vector<float> heads_host;
    fill_activation(&heads_host, kDecodeAttnGroups, kDecodeAttnGroupDim);
    const std::vector<float> sentinel = sentinel_values(allocation_count);

    tensor_owner heads_gpu(heads_count * sizeof(float));
    tensor_owner legacy_gpu(allocation_count * sizeof(float));
    tensor_owner candidate_gpu(allocation_count * sizeof(float));
    if (!heads_gpu.ptr || !legacy_gpu.ptr || !candidate_gpu.ptr ||
        !write_tensor(heads_gpu.ptr, heads_host) ||
        !write_tensor(legacy_gpu.ptr, sentinel) ||
        !write_tensor(candidate_gpu.ptr, sentinel)) {
        std::fprintf(stderr, "grouped attention-A decode: setup FAIL\n");
        return false;
    }

    /* Eight standalone decode calls are the bitwise oracle.  Per-group seeds
     * and activation rows make a wrong weight/input group immediately visible. */
    for (uint32_t group = 0; group < kDecodeAttnGroups; group++) {
        tensor_owner head_group(ds4_gpu_tensor_view(
            heads_gpu.ptr,
            (uint64_t)group * kDecodeAttnGroupDim * sizeof(float),
            (uint64_t)kDecodeAttnGroupDim * sizeof(float)));
        tensor_owner low_group(ds4_gpu_tensor_view(
            legacy_gpu.ptr,
            (uint64_t)group * kDecodeAttnRank * sizeof(float),
            (uint64_t)kDecodeAttnRank * sizeof(float)));
        if (!head_group.ptr || !low_group.ptr ||
            ds4_gpu_matmul_quant_tensor(
                low_group.ptr, model.data, model.size,
                model.decode_attn_a_offset +
                    (uint64_t)group * group_weight_bytes,
                kQ4Type, kDecodeAttnGroupDim, kDecodeAttnRank,
                head_group.ptr, 1u) == 0) {
            std::fprintf(stderr,
                         "grouped attention-A legacy group=%u FAIL\n", group);
            return false;
        }
    }

    env_snapshot enable(kGroupedDecodeEnable);
    env_snapshot disable(kGroupedDecodeDisable);
    env_snapshot require(kGroupedDecodeRequire);
    env_snapshot stats(kGroupedDecodeStats);
    (void)setenv(kGroupedDecodeStats, "1", 1);
    (void)unsetenv(kGroupedDecodeEnable);
    (void)unsetenv(kGroupedDecodeDisable);
    (void)unsetenv(kGroupedDecodeRequire);

    const int default_rc = ds4_gpu_attention_output_low_q4_K_slice_tensor(
        candidate_gpu.ptr, model.data, model.size,
        model.decode_attn_a_offset, kDecodeAttnGroupDim, kDecodeAttnRank,
        0u, kDecodeAttnGroups, heads_gpu.ptr);
    bool ok = default_rc == 0;
    if (default_rc != 0) {
        std::fprintf(stderr,
                     "grouped attention-A default gate: expected rc=0 got=%d FAIL\n",
                     default_rc);
    }
    ok = unchanged_after_rejected_call(
             candidate_gpu.ptr, sentinel,
             "grouped attention-A disabled-by-default preserves output") && ok;

    (void)setenv(kGroupedDecodeEnable, "1", 1);
    (void)setenv(kGroupedDecodeDisable, "1", 1);
    const int disabled_only_rc =
        ds4_gpu_attention_output_low_q4_K_slice_tensor(
            candidate_gpu.ptr, model.data, model.size,
            model.decode_attn_a_offset, kDecodeAttnGroupDim, kDecodeAttnRank,
            0u, kDecodeAttnGroups, heads_gpu.ptr);
    if (disabled_only_rc != 0) {
        std::fprintf(stderr,
                     "grouped attention-A ENABLE+DISABLE: expected rc=0 got=%d FAIL\n",
                     disabled_only_rc);
        ok = false;
    }
    ok = unchanged_after_rejected_call(
             candidate_gpu.ptr, sentinel,
             "grouped attention-A DISABLE dominates ENABLE") && ok;

    (void)setenv(kGroupedDecodeRequire, "1", 1);
    const int disabled_rc =
        ds4_gpu_attention_output_low_q4_K_slice_tensor(
            candidate_gpu.ptr, model.data, model.size,
            model.decode_attn_a_offset, kDecodeAttnGroupDim, kDecodeAttnRank,
            0u, kDecodeAttnGroups, heads_gpu.ptr);
    if (disabled_rc != -1) {
        std::fprintf(stderr,
                     "grouped attention-A DISABLE+REQUIRE: expected rc=-1 got=%d FAIL\n",
                     disabled_rc);
        ok = false;
    }
    ok = unchanged_after_rejected_call(
             candidate_gpu.ptr, sentinel,
             "grouped attention-A DISABLE dominates REQUIRE") && ok;

    (void)unsetenv(kGroupedDecodeDisable);
    const int invalid_rc = ds4_gpu_attention_output_low_q4_K_slice_tensor(
        candidate_gpu.ptr, model.data, model.size, model.size - 16u,
        kDecodeAttnGroupDim, kDecodeAttnRank, 0u, kDecodeAttnGroups,
        heads_gpu.ptr);
    if (invalid_rc != -1) {
        std::fprintf(stderr,
                     "grouped attention-A REQUIRE range guard: expected rc=-1 got=%d FAIL\n",
                     invalid_rc);
        ok = false;
    }
    ok = unchanged_after_rejected_call(
             candidate_gpu.ptr, sentinel,
             "grouped attention-A rejected range preserves output") && ok;

    const int candidate_rc = ds4_gpu_attention_output_low_q4_K_slice_tensor(
        candidate_gpu.ptr, model.data, model.size,
        model.decode_attn_a_offset, kDecodeAttnGroupDim, kDecodeAttnRank,
        0u, kDecodeAttnGroups, heads_gpu.ptr);
    std::vector<float> legacy_host(allocation_count);
    std::vector<float> candidate_host(allocation_count);
    if (candidate_rc != 1 || !read_tensor(legacy_gpu.ptr, &legacy_host) ||
        !read_tensor(candidate_gpu.ptr, &candidate_host)) {
        std::fprintf(stderr,
                     "grouped attention-A candidate dispatch/read rc=%d FAIL\n",
                     candidate_rc);
        return false;
    }
    ok = output_guard_unchanged(
             legacy_host, sentinel, logical_count,
             "grouped attention-A legacy output canary") && ok;
    ok = output_guard_unchanged(
             candidate_host, sentinel, logical_count,
             "grouped attention-A candidate output canary") && ok;
    legacy_host.resize(logical_count);
    candidate_host.resize(logical_count);
    ok = bitwise_equal(candidate_host, legacy_host,
                       "grouped attention-A candidate vs 8 legacy calls") && ok;

    /* A non-zero weight-group origin consumes a compact input/output slice.
     * Reuse groups 3 and 4 from the full fixture to verify both the weight
     * skip and local grouped layout. */
    constexpr uint32_t subset_group0 = 3u;
    constexpr uint32_t subset_group_cnt = 2u;
    const size_t subset_logical_count =
        (size_t)subset_group_cnt * kDecodeAttnRank;
    const std::vector<float> subset_sentinel =
        sentinel_values(subset_logical_count + kOutputGuardFloats);
    tensor_owner subset_heads(ds4_gpu_tensor_view(
        heads_gpu.ptr,
        (uint64_t)subset_group0 * kDecodeAttnGroupDim * sizeof(float),
        (uint64_t)subset_group_cnt * kDecodeAttnGroupDim * sizeof(float)));
    tensor_owner subset_legacy(subset_sentinel.size() * sizeof(float));
    tensor_owner subset_candidate(subset_sentinel.size() * sizeof(float));
    if (!subset_heads.ptr || !subset_legacy.ptr || !subset_candidate.ptr ||
        !write_tensor(subset_legacy.ptr, subset_sentinel) ||
        !write_tensor(subset_candidate.ptr, subset_sentinel)) {
        std::fprintf(stderr, "grouped attention-A subset: setup FAIL\n");
        return false;
    }
    for (uint32_t i = 0; i < subset_group_cnt; i++) {
        tensor_owner head_group(ds4_gpu_tensor_view(
            subset_heads.ptr,
            (uint64_t)i * kDecodeAttnGroupDim * sizeof(float),
            (uint64_t)kDecodeAttnGroupDim * sizeof(float)));
        tensor_owner low_group(ds4_gpu_tensor_view(
            subset_legacy.ptr,
            (uint64_t)i * kDecodeAttnRank * sizeof(float),
            (uint64_t)kDecodeAttnRank * sizeof(float)));
        if (!head_group.ptr || !low_group.ptr ||
            ds4_gpu_matmul_quant_tensor(
                low_group.ptr, model.data, model.size,
                model.decode_attn_a_offset +
                    (uint64_t)(subset_group0 + i) * group_weight_bytes,
                kQ4Type, kDecodeAttnGroupDim, kDecodeAttnRank,
                head_group.ptr, 1u) == 0) {
            std::fprintf(stderr,
                         "grouped attention-A subset legacy group=%u FAIL\n",
                         subset_group0 + i);
            return false;
        }
    }
    const int subset_rc = ds4_gpu_attention_output_low_q4_K_slice_tensor(
        subset_candidate.ptr, model.data, model.size,
        model.decode_attn_a_offset, kDecodeAttnGroupDim, kDecodeAttnRank,
        subset_group0, subset_group_cnt, subset_heads.ptr);
    std::vector<float> subset_legacy_host(subset_sentinel.size());
    std::vector<float> subset_candidate_host(subset_sentinel.size());
    if (subset_rc != 1 ||
        !read_tensor(subset_legacy.ptr, &subset_legacy_host) ||
        !read_tensor(subset_candidate.ptr, &subset_candidate_host)) {
        std::fprintf(stderr,
                     "grouped attention-A subset dispatch/read rc=%d FAIL\n",
                     subset_rc);
        return false;
    }
    ok = output_guard_unchanged(
             subset_legacy_host, subset_sentinel, subset_logical_count,
             "grouped attention-A subset legacy canary") && ok;
    ok = output_guard_unchanged(
             subset_candidate_host, subset_sentinel, subset_logical_count,
             "grouped attention-A subset candidate canary") && ok;
    subset_legacy_host.resize(subset_logical_count);
    subset_candidate_host.resize(subset_logical_count);
    ok = bitwise_equal(
             subset_candidate_host, subset_legacy_host,
             "grouped attention-A subset group0=3 count=2 vs legacy") && ok;
    std::fprintf(stderr,
                 "grouped attention-A decode groups=8 K=4096 rank=1024: "
                 "default=%d disabled=%d disabled_required=%d invalid=%d "
                 "candidate=%d subset=%d %s\n",
                 default_rc, disabled_only_rc, disabled_rc, invalid_rc,
                 candidate_rc, subset_rc,
                 ok ? "PASS" : "FAIL");
    return ok;
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
                     "ROCm Q4 dense/pair/prefill: SKIP "
                     "(HIP runtime has no visible device: %s)\n",
                     err == hipSuccess ? "device count is zero" : hipGetErrorString(err));
        return 0;
    }
    return count;
#else
    std::fprintf(stderr,
                 "ROCm Q4 dense/pair/prefill: SKIP "
                 "(compiled without HIP runtime headers)\n");
    return 0;
#endif
}

}  // namespace

int main(int argc, char **argv) {
    bool run_dense = true;
    bool run_pair = true;
    bool run_prefill = true;
    bool run_grouped_decode = true;
    bool run_prefill_long = false;
    if (argc == 2 && std::strcmp(argv[1], "--dense") == 0) {
        run_pair = false;
        run_prefill = false;
        run_grouped_decode = false;
    } else if (argc == 2 && std::strcmp(argv[1], "--pair") == 0) {
        run_dense = false;
        run_prefill = false;
        run_grouped_decode = false;
    } else if (argc == 2 && std::strcmp(argv[1], "--prefill") == 0) {
        run_dense = false;
        run_pair = false;
        run_grouped_decode = false;
    } else if (argc == 2 &&
               std::strcmp(argv[1], "--grouped-decode") == 0) {
        run_dense = false;
        run_pair = false;
        run_prefill = false;
    } else if (argc == 2 &&
               std::strcmp(argv[1], "--prefill-long") == 0) {
        run_dense = false;
        run_pair = false;
        run_grouped_decode = false;
        run_prefill_long = true;
    } else if (argc > 1 &&
               !(argc == 2 && std::strcmp(argv[1], "--all") == 0)) {
        std::fprintf(stderr,
                     "usage: %s [--all|--dense|--pair|--grouped-decode|"
                     "--prefill|--prefill-long]\n",
                     argv[0]);
        return 2;
    }

    env_snapshot prefill_enable(kPrefillEnable);
    env_snapshot prefill_disable(kPrefillDisable);
    env_snapshot prefill_require(kPrefillRequire);
    env_snapshot grouped_enable(kGroupedDecodeEnable);
    env_snapshot grouped_disable(kGroupedDecodeDisable);
    env_snapshot grouped_require(kGroupedDecodeRequire);
    env_snapshot grouped_stats(kGroupedDecodeStats);
    (void)unsetenv(kPrefillEnable);
    (void)unsetenv(kPrefillDisable);
    (void)unsetenv(kPrefillRequire);
    (void)unsetenv(kGroupedDecodeEnable);
    (void)unsetenv(kGroupedDecodeDisable);
    (void)unsetenv(kGroupedDecodeRequire);
    (void)unsetenv(kGroupedDecodeStats);

    if (detect_rocm_device() <= 0) {
        const char *require_device =
            std::getenv("DS4_TEST_REQUIRE_ROCM_DEVICE");
        const bool required = require_device && require_device[0] != '\0' &&
                              std::strcmp(require_device, "0") != 0;
        return required ? 1 : kSkip;
    }
    if (!ds4_gpu_init()) {
        std::fprintf(stderr,
                     "ROCm Q4 dense/pair/prefill: FAIL "
                     "(device is visible but ds4_gpu_init failed)\n");
        return 1;
    }

    aligned_model model;
    bool ok = make_model(&model);
    if (!ok) {
        std::fprintf(stderr,
                     "ROCm Q4 dense/pair/prefill: fixture allocation FAIL\n");
    } else if (!ds4_gpu_set_model_map(model.data, model.size)) {
        std::fprintf(stderr,
                     "ROCm Q4 dense/pair/prefill: model-map registration FAIL\n");
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
    if (model_ready && run_grouped_decode) {
        std::fprintf(stderr,
                     "ROCm Q4 grouped attention-A decode parity "
                     "(one grouped dispatch vs eight legacy calls):\n");
        ok = run_grouped_attention_decode_case(model) && ok;
    }
    if (model_ready && run_prefill) {
        std::fprintf(stderr,
                     "ROCm Q4 tiled prefill parity "
                     "(forced DISABLE vs default+REQUIRE):\n");
        const bool prefill9_ok = run_prefill_parity_case(
            model, 9u, model.weight0_offset, kM0, true,
            "prefill K=4096 M=65 n_tok=9");
        const bool prefill30_ok = run_prefill_parity_case(
            model, 30u, model.weight0_offset, kM0, true,
            "prefill K=4096 M=65 n_tok=30 (token-tail nt=6)");
        const bool prefill128_ok = run_prefill_parity_case(
            model, 128u, model.weight1_offset, kM1, true,
            "prefill K=4096 M=33 n_tok=128");
        const bool prefill_tail9_ok = run_prefill_parity_case(
            model, 9u, model.tail_k1024_offset, kM0, true,
            "prefill K=1024 M=65 n_tok=9 (K-tail nb=4)", kTailK);
        const bool prefill_tail128_ok = run_prefill_parity_case(
            model, 128u, model.tail_k1024_offset, kM0, true,
            "prefill K=1024 M=65 n_tok=128 (K-tail nb=4)", kTailK);
        const bool prefill_single9_ok = run_prefill_parity_case(
            model, 9u, model.attn_b_offset, kAttnOutDim, true,
            "prefill K=256 M=65 n_tok=9 (K-tail nb=1)", kAttnLowDim);
        const bool prefill_single128_ok = run_prefill_parity_case(
            model, 128u, model.attn_b_offset, kAttnOutDim, true,
            "prefill K=256 M=65 n_tok=128 (K-tail nb=1)", kAttnLowDim);
        const bool prefill_pair_ok = run_prefill_pair_case(model);
        const bool attention9_ok = run_attention_prefill_case(
            model, 9u,
            "attention prefill groups=8 K=4096 rank=32 M=65 n_tok=9");
        const bool attention30_ok = run_attention_prefill_case(
            model, 30u,
            "attention prefill groups=8 K=4096 rank=32 M=65 "
            "n_tok=30 (token-tail nt=6)");
        const bool attention128_ok = run_attention_prefill_case(
            model, 128u,
            "attention prefill groups=8 K=4096 rank=32 M=65 n_tok=128");
        const bool attention_q8_9_ok = run_attention_prefill_case(
            model, 9u,
            "attention prefill Q4-A/Q8-B groups=8 K=4096 rank=32 M=65 "
            "n_tok=9",
            kQ8Type);
        const bool attention_q8_30_ok = run_attention_prefill_case(
            model, 30u,
            "attention prefill Q4-A/Q8-B groups=8 K=4096 rank=32 M=65 "
            "n_tok=30 (token-tail nt=6)",
            kQ8Type);
        const bool gate_ok = run_prefill_gate_guards(model);
        ok = prefill9_ok && prefill30_ok && prefill128_ok &&
             prefill_tail9_ok &&
             prefill_tail128_ok && prefill_single9_ok &&
             prefill_single128_ok && prefill_pair_ok && attention9_ok &&
             attention30_ok && attention128_ok && attention_q8_9_ok &&
             attention_q8_30_ok && gate_ok && ok;
        if (run_prefill_long) {
            // A 64 MiB activation and a roughly 0.5 Gi-op projection stress
            // arbitrary token-grid tails without the much slower CPU oracle.
            const bool long_ok = run_prefill_parity_case(
                model, 4096u, model.weight1_offset, kM1, false,
                "prefill stress K=4096 M=33 n_tok=4096");
            ok = long_ok && ok;
        }
    }

    // Registered host ranges must be released before their aligned backing
    // allocation is destroyed.
    ds4_gpu_cleanup();
    std::fprintf(stderr, "ROCm Q4 dense/pair/prefill oracle: %s\n",
                 ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
