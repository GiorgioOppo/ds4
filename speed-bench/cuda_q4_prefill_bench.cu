// SPDX-License-Identifier: MIT
// Resident, CUDA-event-only Q4_K prefill microbenchmark.

#include "ds4_gpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kQ4Type = 12u;
constexpr uint32_t kQkK = 256u;
constexpr uint32_t kDenseK = 4096u;
constexpr uint32_t kDenseM = 1024u;
constexpr uint32_t kKvM = 512u;
constexpr uint32_t kQbK = 1024u;
constexpr uint32_t kQbM = 32768u;
constexpr uint32_t kDefaultSets = 4u;
constexpr uint32_t kDefaultSamples = 8u;
constexpr uint32_t kDefaultWarmup = 2u;
constexpr uint32_t kGuardWords = 64u;
constexpr uint64_t kCompareChunk = 4u * 1024u * 1024u;

struct block_q4_K_host {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[kQkK / 2u];
};

static_assert(sizeof(block_q4_K_host) == 144u,
              "Q4_K fixture must match the raw GGUF layout");

enum class bench_case {
    all,
    dense,
    pair,
    qb,
};

enum class cuda_path {
    mmq,
    legacy,
};

struct config {
    bench_case selected = bench_case::all;
    cuda_path path = cuda_path::mmq;
    std::vector<uint32_t> tokens = {9u, 17u, 33u, 128u, 512u};
    uint32_t sets = kDefaultSets;
    uint32_t samples = kDefaultSamples;
    uint32_t warmup = kDefaultWarmup;
};

struct weight_set {
    uint64_t dense_offset = 0;
    uint64_t kv_offset = 0;
    uint64_t qb_offset = 0;
};

struct model_fixture {
    uint8_t *data = nullptr;
    uint64_t size = 0;
    uint64_t payload_bytes = 0;
    std::vector<weight_set> weights;

    ~model_fixture() { std::free(data); }
    model_fixture() = default;
    model_fixture(const model_fixture &) = delete;
    model_fixture &operator=(const model_fixture &) = delete;
};

struct tensor_owner {
    ds4_gpu_tensor *ptr = nullptr;

    explicit tensor_owner(uint64_t bytes) : ptr(ds4_gpu_tensor_alloc(bytes)) {}
    ~tensor_owner() { ds4_gpu_tensor_free(ptr); }
    tensor_owner(const tensor_owner &) = delete;
    tensor_owner &operator=(const tensor_owner &) = delete;
};

struct env_snapshot {
    const char *name;
    bool existed;
    std::string value;

    explicit env_snapshot(const char *key)
        : name(key), existed(std::getenv(key) != nullptr),
          value(existed ? std::getenv(key) : "") {}
    ~env_snapshot() {
        if (existed) {
            (void)setenv(name, value.c_str(), 1);
        } else {
            (void)unsetenv(name);
        }
    }
    env_snapshot(const env_snapshot &) = delete;
    env_snapshot &operator=(const env_snapshot &) = delete;
};

struct event_timer {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;

    event_timer() {
        if (cudaEventCreate(&begin) != cudaSuccess ||
            cudaEventCreate(&end) != cudaSuccess) {
            std::fprintf(stderr,
                         "cuda-q4-prefill-bench: CUDA event allocation failed\n");
            std::exit(1);
        }
    }
    ~event_timer() {
        if (begin) (void)cudaEventDestroy(begin);
        if (end) (void)cudaEventDestroy(end);
    }

    bool measure(const std::function<bool()> &dispatch, float *milliseconds) {
        // The harness disables decode graphs.  Production Q4 eager dispatch
        // passes cuda_decode_stream()==0 to both MMQ (including its async
        // scratch pool) and the Q8_K fallback kernels, so bracketing stream 0
        // covers every enqueue in the public call.
        if (cudaEventRecord(begin, nullptr) != cudaSuccess) return false;
        if (!dispatch()) return false;
        if (cudaEventRecord(end, nullptr) != cudaSuccess ||
            cudaEventSynchronize(end) != cudaSuccess ||
            cudaEventElapsedTime(milliseconds, begin, end) != cudaSuccess) {
            return false;
        }
        return true;
    }
};

struct arm {
    const char *name;
    std::function<bool(uint32_t)> dispatch;
};

struct stats {
    double minimum = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double mean = 0.0;
};

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0u && b > std::numeric_limits<uint64_t>::max() / a) return false;
    *out = a * b;
    return true;
}

uint32_t lcg_next(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

void fill_q4(void *storage, uint64_t bytes, uint32_t seed) {
    auto *blocks = static_cast<block_q4_K_host *>(storage);
    const uint64_t count = bytes / sizeof(*blocks);
    uint32_t state = seed;
    for (uint64_t i = 0; i < count; i++) {
        // Positive, finite FP16 scales.  Payload values are deterministic.
        blocks[i].d = static_cast<uint16_t>(0x2400u +
                                            (lcg_next(&state) & 0xffu));
        blocks[i].dmin = static_cast<uint16_t>(0x2000u +
                                               (lcg_next(&state) & 0xffu));
        for (uint8_t &v : blocks[i].scales) {
            v = static_cast<uint8_t>(lcg_next(&state) >> 24u);
        }
        for (uint8_t &v : blocks[i].qs) {
            v = static_cast<uint8_t>(lcg_next(&state) >> 24u);
        }
    }
}

uint64_t q4_weight_bytes(uint32_t in_dim, uint32_t out_dim) {
    return static_cast<uint64_t>(out_dim) * (in_dim / kQkK) *
           sizeof(block_q4_K_host);
}

bool make_model(model_fixture *model, uint32_t sets) {
    constexpr uint64_t page = 4096u;
    const uint64_t dense_bytes = q4_weight_bytes(kDenseK, kDenseM);
    const uint64_t kv_bytes = q4_weight_bytes(kDenseK, kKvM);
    const uint64_t qb_bytes = q4_weight_bytes(kQbK, kQbM);

    model->weights.resize(sets);
    uint64_t cursor = 0;
    auto append = [&](uint64_t bytes) {
        const uint64_t offset = align_up(cursor, page);
        cursor = offset + bytes;
        model->payload_bytes += bytes;
        return offset;
    };
    for (uint32_t i = 0; i < sets; i++) {
        model->weights[i].dense_offset = append(dense_bytes);
        model->weights[i].kv_offset = append(kv_bytes);
        model->weights[i].qb_offset = append(qb_bytes);
    }
    model->size = align_up(cursor, page);
    void *storage = nullptr;
    if (posix_memalign(&storage, static_cast<size_t>(page),
                       static_cast<size_t>(model->size)) != 0) {
        return false;
    }
    model->data = static_cast<uint8_t *>(storage);
    std::memset(model->data, 0, static_cast<size_t>(model->size));
    for (uint32_t i = 0; i < sets; i++) {
        fill_q4(model->data + model->weights[i].dense_offset, dense_bytes,
                0x243f6a88u ^ (i * 0x9e3779b9u));
        fill_q4(model->data + model->weights[i].kv_offset, kv_bytes,
                0x85a308d3u ^ (i * 0x7f4a7c15u));
        fill_q4(model->data + model->weights[i].qb_offset, qb_bytes,
                0x13198a2eu ^ (i * 0x94d049bbu));
    }
    return true;
}

void fill_activation(std::vector<float> *values, uint32_t n_tokens,
                     uint32_t in_dim) {
    values->resize(static_cast<uint64_t>(n_tokens) * in_dim);
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t block = 0; block < in_dim / kQkK; block++) {
            float *dst = values->data() +
                         static_cast<uint64_t>(token) * in_dim + block * kQkK;
            for (uint32_t i = 0; i < kQkK; i++) {
                const int q = static_cast<int>((i * 73u + token * 37u +
                                                block * 19u) % 241u) - 120;
                dst[i] = static_cast<float>(q) / 32.0f;
            }
            dst[0] = ((token + block) & 1u) ? 127.0f / 32.0f
                                             : -127.0f / 32.0f;
        }
    }
}

std::vector<uint32_t> guard_pattern(uint32_t salt) {
    std::vector<uint32_t> guard(kGuardWords);
    for (uint32_t i = 0; i < kGuardWords; i++) {
        guard[i] = 0x7fc12000u ^ salt ^ (i * 0x00010101u);
    }
    return guard;
}

bool prepare_guard(ds4_gpu_tensor *tensor, uint64_t logical_bytes,
                   uint32_t salt) {
    const std::vector<uint32_t> guard = guard_pattern(salt);
    return ds4_gpu_tensor_write(tensor, logical_bytes, guard.data(),
                                guard.size() * sizeof(guard[0])) != 0;
}

bool poison_output(ds4_gpu_tensor *tensor, uint64_t logical_bytes,
                   uint32_t pattern, uint32_t guard_salt) {
    if (!tensor || logical_bytes == 0u ||
        logical_bytes % sizeof(uint32_t) != 0u) {
        return false;
    }
    const uint64_t chunk_bytes = std::min(kCompareChunk, logical_bytes);
    std::vector<uint32_t> poison(
        static_cast<size_t>(chunk_bytes / sizeof(uint32_t)), pattern);
    for (uint64_t offset = 0; offset < logical_bytes; offset += chunk_bytes) {
        const uint64_t count = std::min(chunk_bytes, logical_bytes - offset);
        if (!ds4_gpu_tensor_write(tensor, offset, poison.data(), count)) {
            return false;
        }
    }
    return prepare_guard(tensor, logical_bytes, guard_salt);
}

bool check_guard(const ds4_gpu_tensor *tensor, uint64_t logical_bytes,
                 uint32_t salt, const char *label) {
    const std::vector<uint32_t> expected = guard_pattern(salt);
    std::vector<uint32_t> got(expected.size());
    if (!ds4_gpu_tensor_read(tensor, logical_bytes, got.data(),
                             got.size() * sizeof(got[0]))) {
        std::fprintf(stderr, "%s: guard read failed\n", label);
        return false;
    }
    if (got != expected) {
        const auto mismatch = std::mismatch(got.begin(), got.end(),
                                            expected.begin());
        std::fprintf(stderr, "%s: guard overwritten at word %zu\n", label,
                     static_cast<size_t>(mismatch.first - got.begin()));
        return false;
    }
    return true;
}

bool output_is_finite(const ds4_gpu_tensor *tensor, uint64_t logical_bytes,
                      const char *label) {
    const uint64_t chunk_bytes = std::min(kCompareChunk, logical_bytes);
    std::vector<float> values(
        static_cast<size_t>(chunk_bytes / sizeof(float)));
    for (uint64_t offset = 0; offset < logical_bytes; offset += chunk_bytes) {
        const uint64_t count = std::min(chunk_bytes, logical_bytes - offset);
        if (!ds4_gpu_tensor_read(tensor, offset, values.data(), count)) {
            std::fprintf(stderr, "%s: output read failed\n", label);
            return false;
        }
        const size_t n = static_cast<size_t>(count / sizeof(float));
        for (size_t i = 0; i < n; i++) {
            if (!std::isfinite(values[i])) {
                std::fprintf(stderr,
                             "%s: non-finite/unwritten output at element %llu\n",
                             label,
                             static_cast<unsigned long long>(
                                 offset / sizeof(float) + i));
                return false;
            }
        }
    }
    return true;
}

bool bitwise_equal(const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
                   uint64_t bytes, const char *label) {
    const uint64_t chunk = std::min(kCompareChunk, bytes);
    std::vector<uint8_t> lhs(static_cast<size_t>(chunk));
    std::vector<uint8_t> rhs(static_cast<size_t>(chunk));
    for (uint64_t offset = 0; offset < bytes; offset += chunk) {
        const uint64_t count = std::min(chunk, bytes - offset);
        if (!ds4_gpu_tensor_read(a, offset, lhs.data(), count) ||
            !ds4_gpu_tensor_read(b, offset, rhs.data(), count)) {
            std::fprintf(stderr, "%s: oracle read failed\n", label);
            return false;
        }
        if (std::memcmp(lhs.data(), rhs.data(), static_cast<size_t>(count)) !=
            0) {
            uint64_t first = 0;
            while (first < count && lhs[static_cast<size_t>(first)] ==
                                        rhs[static_cast<size_t>(first)]) {
                first++;
            }
            std::fprintf(stderr, "%s: bitwise mismatch at output byte %llu\n",
                         label,
                         static_cast<unsigned long long>(offset + first));
            return false;
        }
    }
    return true;
}

float fp16_to_float(uint16_t h) {
    const float sign = (h & 0x8000u) ? -1.0f : 1.0f;
    const uint32_t exponent = (h >> 10u) & 0x1fu;
    const uint32_t mantissa = h & 0x3ffu;
    if (exponent == 0u) {
        return mantissa == 0u
            ? std::copysign(0.0f, sign)
            : sign * std::ldexp(static_cast<float>(mantissa), -24);
    }
    if (exponent == 31u) {
        return mantissa == 0u
            ? std::copysign(std::numeric_limits<float>::infinity(), sign)
            : std::numeric_limits<float>::quiet_NaN();
    }
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                             static_cast<int>(exponent) - 15);
}

void get_scale_min_k4(uint32_t index, const uint8_t *packed,
                      uint8_t *scale, uint8_t *minimum) {
    if (index < 4u) {
        *scale = packed[index] & 63u;
        *minimum = packed[index + 4u] & 63u;
    } else {
        *scale = static_cast<uint8_t>((packed[index + 4u] & 0x0fu) |
                                      ((packed[index - 4u] >> 6u) << 4u));
        *minimum = static_cast<uint8_t>((packed[index + 4u] >> 4u) |
                                        ((packed[index] >> 6u) << 4u));
    }
}

void dequantize_q4_row(const block_q4_K_host *blocks, uint32_t in_dim,
                       std::vector<float> *row) {
    row->resize(in_dim);
    float *dst = row->data();
    const uint32_t n_blocks = in_dim / kQkK;
    for (uint32_t block = 0; block < n_blocks; block++) {
        const float d = fp16_to_float(blocks[block].d);
        const float dmin = fp16_to_float(blocks[block].dmin);
        const uint8_t *q = blocks[block].qs;
        uint32_t scale_index = 0;
        for (uint32_t group = 0; group < kQkK; group += 64u) {
            uint8_t sc0 = 0, min0 = 0, sc1 = 0, min1 = 0;
            get_scale_min_k4(scale_index, blocks[block].scales,
                             &sc0, &min0);
            get_scale_min_k4(scale_index + 1u, blocks[block].scales,
                             &sc1, &min1);
            const float ds0 = d * sc0;
            const float dm0 = dmin * min0;
            const float ds1 = d * sc1;
            const float dm1 = dmin * min1;
            for (uint32_t i = 0; i < 32u; i++) {
                *dst++ = ds0 * static_cast<float>(q[i] & 0x0fu) - dm0;
            }
            for (uint32_t i = 0; i < 32u; i++) {
                *dst++ = ds1 * static_cast<float>(q[i] >> 4u) - dm1;
            }
            q += 32u;
            scale_index += 2u;
            (void)group;
        }
    }
}

std::vector<uint32_t> sample_indices(uint32_t extent) {
    std::vector<uint32_t> values = {
        0u, extent / 3u, extent / 2u, extent - 1u,
    };
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

bool sampled_cpu_oracle(const ds4_gpu_tensor *output,
                        const model_fixture &model, uint64_t weight_offset,
                        const std::vector<float> &activation,
                        uint32_t n_tokens, uint32_t in_dim,
                        uint32_t out_dim, const char *label) {
    const std::vector<uint32_t> tokens = sample_indices(n_tokens);
    const std::vector<uint32_t> rows = sample_indices(out_dim);
    const uint64_t blocks_per_row = in_dim / kQkK;
    const uint64_t row_bytes = blocks_per_row * sizeof(block_q4_K_host);
    std::vector<float> weight_row;
    // cuda/mmq/test/test_mmq_parity.cu validates Q4_K MMQ against this same
    // dequantized-weight x original-F32 reference at 0.20*sqrt(K) absolute
    // and 5% relative error.  Use a small shared envelope for both that Q8_1
    // path and the legacy 256-value Q8_K activation quantizer.  Every fixture
    // block pins |x|max to 127/32, keeping the Q8_K step bounded and stable.
    const double abs_tol = 0.25 * std::sqrt(static_cast<double>(in_dim));
    constexpr double rel_tol = 0.06;

    for (uint32_t row : rows) {
        const auto *blocks = reinterpret_cast<const block_q4_K_host *>(
            model.data + weight_offset + static_cast<uint64_t>(row) * row_bytes);
        dequantize_q4_row(blocks, in_dim, &weight_row);
        for (uint32_t token : tokens) {
            const float *x = activation.data() +
                             static_cast<uint64_t>(token) * in_dim;
            float reference = 0.0f;
            for (uint32_t k = 0; k < in_dim; k++) {
                reference += weight_row[k] * x[k];
            }
            float got = 0.0f;
            const uint64_t element = static_cast<uint64_t>(token) * out_dim + row;
            if (!ds4_gpu_tensor_read(output, element * sizeof(float), &got,
                                     sizeof(got))) {
                std::fprintf(stderr, "%s: sampled output read failed\n", label);
                return false;
            }
            const double abs_error = std::fabs(static_cast<double>(got) -
                                               reference);
            const double rel_error = reference != 0.0f
                ? abs_error / std::fabs(static_cast<double>(reference))
                : (abs_error == 0.0 ? 0.0
                                    : std::numeric_limits<double>::infinity());
            if (!std::isfinite(got) ||
                (abs_error > abs_tol && rel_error > rel_tol)) {
                std::fprintf(
                    stderr,
                    "%s: CPU oracle mismatch token=%u row=%u got=%.7g "
                    "reference=%.7g abs=%.5g rel=%.5g limits=%.5g/%.3g\n",
                    label, token, row, got, reference, abs_error, rel_error,
                    abs_tol, rel_tol);
                return false;
            }
        }
    }
    return true;
}

double percentile(std::vector<double> sorted, double fraction) {
    std::sort(sorted.begin(), sorted.end());
    if (sorted.empty()) return 0.0;
    const double position = fraction * static_cast<double>(sorted.size() - 1u);
    const size_t lo = static_cast<size_t>(std::floor(position));
    const size_t hi = static_cast<size_t>(std::ceil(position));
    const double alpha = position - static_cast<double>(lo);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * alpha;
}

stats summarize(const std::vector<double> &samples) {
    stats out;
    out.minimum = *std::min_element(samples.begin(), samples.end());
    out.median = percentile(samples, 0.5);
    out.p95 = percentile(samples, 0.95);
    for (double sample : samples) out.mean += sample;
    out.mean /= static_cast<double>(samples.size());
    return out;
}

const char *path_name(cuda_path path) {
    return path == cuda_path::mmq ? "mmq" : "legacy";
}

bool benchmark_single_path(
        const char *case_name, uint32_t n_tokens, uint32_t in_dim,
        uint32_t out_dim, const config &cfg, const arm &which,
        const std::function<bool()> &oracle_prepare,
        const std::function<bool(uint32_t)> &oracle) {
    // Validate every rotating weight set and prime lazy Q8 scratch before
    // recording a CUDA event.  Upload, poison, readback, and CPU work remain
    // outside the measured interval.
    for (uint32_t set = 0; set < cfg.sets; set++) {
        if (!oracle_prepare() || !which.dispatch(set) ||
            !ds4_gpu_synchronize() || !oracle(set)) {
            std::fprintf(stderr,
                         "cuda-q4-prefill-bench: %s oracle failed for "
                         "weight set %u\n",
                         case_name, set);
            return false;
        }
    }

    for (uint32_t i = 0; i < cfg.warmup; i++) {
        if (!which.dispatch(i % cfg.sets) || !ds4_gpu_synchronize()) {
            return false;
        }
    }

    event_timer timer;
    std::vector<double> samples;
    samples.reserve(cfg.samples);
    for (uint32_t i = 0; i < cfg.samples; i++) {
        float elapsed = 0.0f;
        if (!timer.measure([&]() { return which.dispatch(i % cfg.sets); },
                           &elapsed)) {
            std::fprintf(stderr,
                         "cuda-q4-prefill-bench: %s/%s timed dispatch failed\n",
                         case_name, which.name);
            return false;
        }
        samples.push_back(static_cast<double>(elapsed));
    }

    const stats result = summarize(samples);
    const double macs = static_cast<double>(n_tokens) * in_dim * out_dim;
    const double gmac_s = macs / (result.median * 1.0e6);
    std::printf(
        "DS4_CUDA_Q4_PREFILL_BENCH case=%s path=%s N=%u K=%u M=%u "
        "variant=%s samples=%u sets=%u ms_p50=%.6f ms_min=%.6f "
        "ms_p95=%.6f ms_mean=%.6f gmac_s=%.3f\n",
        case_name, path_name(cfg.path), n_tokens, in_dim, out_dim, which.name,
        cfg.samples, cfg.sets, result.median, result.minimum, result.p95,
        result.mean, gmac_s);
    std::fflush(stdout);
    return true;
}

bool benchmark_pair_arms(
        uint32_t n_tokens, const config &cfg, const arm &baseline,
        const arm &candidate, const std::function<bool()> &oracle_prepare,
        const std::function<bool(uint32_t)> &oracle) {
    for (uint32_t set = 0; set < cfg.sets; set++) {
        if (!oracle_prepare() || !baseline.dispatch(set) ||
            !ds4_gpu_synchronize() || !candidate.dispatch(set) ||
            !ds4_gpu_synchronize() || !oracle(set)) {
            std::fprintf(stderr,
                         "cuda-q4-prefill-bench: pair oracle failed for "
                         "weight set %u\n",
                         set);
            return false;
        }
    }

    for (uint32_t i = 0; i < cfg.warmup; i++) {
        const uint32_t set = i % cfg.sets;
        if (!baseline.dispatch(set) || !ds4_gpu_synchronize() ||
            !candidate.dispatch(set) || !ds4_gpu_synchronize()) {
            return false;
        }
    }

    event_timer timer;
    std::vector<double> a_samples;
    std::vector<double> b_samples;
    a_samples.reserve(cfg.samples);
    b_samples.reserve(cfg.samples);
    auto take = [&](const arm &which, uint32_t set,
                    std::vector<double> *samples) {
        float elapsed = 0.0f;
        if (!timer.measure([&]() { return which.dispatch(set); }, &elapsed)) {
            std::fprintf(stderr,
                         "cuda-q4-prefill-bench: pair/%s timed dispatch failed\n",
                         which.name);
            return false;
        }
        samples->push_back(static_cast<double>(elapsed));
        return true;
    };

    // Two samples per arm per cycle. Alternating ABBA/BAAB balances order
    // while both arms see identical rotating resident weights.
    for (uint32_t cycle = 0; a_samples.size() < cfg.samples; cycle++) {
        const uint32_t set0 = (cycle * 2u) % cfg.sets;
        const uint32_t set1 = (cycle * 2u + 1u) % cfg.sets;
        if ((cycle & 1u) == 0u) {
            if (!take(baseline, set0, &a_samples) ||
                !take(candidate, set0, &b_samples) ||
                !take(candidate, set1, &b_samples) ||
                !take(baseline, set1, &a_samples)) return false;
        } else {
            if (!take(candidate, set0, &b_samples) ||
                !take(baseline, set0, &a_samples) ||
                !take(baseline, set1, &a_samples) ||
                !take(candidate, set1, &b_samples)) return false;
        }
    }

    const stats a = summarize(a_samples);
    const stats b = summarize(b_samples);
    std::vector<double> paired_delta;
    paired_delta.reserve(a_samples.size());
    for (size_t i = 0; i < a_samples.size(); i++) {
        paired_delta.push_back((b_samples[i] / a_samples[i] - 1.0) * 100.0);
    }
    const double paired_median = percentile(paired_delta, 0.5);
    const double median_delta = (b.median / a.median - 1.0) * 100.0;
    const double speedup = (a.median / b.median - 1.0) * 100.0;
    const double macs = static_cast<double>(n_tokens) * kDenseK *
                        (kDenseM + kKvM);
    std::printf(
        "DS4_CUDA_Q4_PREFILL_BENCH case=pair path=%s N=%u K=%u M=%u "
        "baseline=%s candidate=%s samples=%u sets=%u "
        "baseline_ms_p50=%.6f candidate_ms_p50=%.6f "
        "baseline_ms_min=%.6f candidate_ms_min=%.6f "
        "baseline_ms_p95=%.6f candidate_ms_p95=%.6f "
        "baseline_gmac_s=%.3f candidate_gmac_s=%.3f "
        "candidate_delta_pct=%.3f paired_delta_pct_p50=%.3f "
        "speedup_pct=%.3f\n",
        path_name(cfg.path), n_tokens, kDenseK, kDenseM + kKvM,
        baseline.name, candidate.name, cfg.samples, cfg.sets, a.median,
        b.median, a.minimum, b.minimum, a.p95, b.p95,
        macs / (a.median * 1.0e6), macs / (b.median * 1.0e6),
        median_delta, paired_median, speedup);
    std::fflush(stdout);
    return true;
}

bool run_dense(const model_fixture &model, const config &cfg,
               uint32_t n_tokens) {
    uint64_t out_elements = 0;
    if (!checked_mul(n_tokens, kDenseM, &out_elements)) return false;
    const uint64_t x_bytes = static_cast<uint64_t>(n_tokens) * kDenseK *
                             sizeof(float);
    const uint64_t out_bytes = out_elements * sizeof(float);
    const uint64_t guard_bytes = kGuardWords * sizeof(uint32_t);
    tensor_owner x(x_bytes + guard_bytes);
    tensor_owner output(out_bytes + guard_bytes);
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, kDenseK);
    if (!x.ptr || !output.ptr ||
        !ds4_gpu_tensor_write(x.ptr, 0, activation.data(), x_bytes) ||
        !prepare_guard(x.ptr, x_bytes, 0x1000u)) {
        std::fprintf(stderr, "dense N=%u: tensor setup failed\n", n_tokens);
        return false;
    }
    const arm path = {
        cfg.path == cuda_path::mmq ? "mmq" : "legacy_q8k",
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       output.ptr, model.data, model.size,
                       model.weights[set].dense_offset, kQ4Type, kDenseK,
                       kDenseM, x.ptr, n_tokens) != 0;
        }};
    return benchmark_single_path(
        "dense", n_tokens, kDenseK, kDenseM, cfg, path,
        [&]() {
            return poison_output(output.ptr, out_bytes, 0x7fc10001u,
                                 0x2000u);
        },
        [&](uint32_t set) {
            return output_is_finite(output.ptr, out_bytes, "dense output") &&
                   sampled_cpu_oracle(output.ptr, model,
                                      model.weights[set].dense_offset,
                                      activation, n_tokens, kDenseK, kDenseM,
                                      "dense") &&
                   check_guard(x.ptr, x_bytes, 0x1000u, "dense input") &&
                   check_guard(output.ptr, out_bytes, 0x2000u,
                               "dense output");
        }) &&
        check_guard(x.ptr, x_bytes, 0x1000u, "dense input final") &&
        check_guard(output.ptr, out_bytes, 0x2000u, "dense output final");
}

bool run_pair(const model_fixture &model, const config &cfg,
              uint32_t n_tokens) {
    if (cfg.path == cuda_path::legacy) {
        std::printf(
            "DS4_CUDA_Q4_PREFILL_SKIP case=pair path=legacy N=%u "
            "reason=fused_prefill_api_requires_mmq\n",
            n_tokens);
        std::fflush(stdout);
        return true;
    }

    const uint64_t x_bytes = static_cast<uint64_t>(n_tokens) * kDenseK *
                             sizeof(float);
    const uint64_t out0_bytes = static_cast<uint64_t>(n_tokens) * kDenseM *
                                sizeof(float);
    const uint64_t out1_bytes = static_cast<uint64_t>(n_tokens) * kKvM *
                                sizeof(float);
    const uint64_t guard_bytes = kGuardWords * sizeof(uint32_t);
    tensor_owner x(x_bytes + guard_bytes);
    tensor_owner separate0(out0_bytes + guard_bytes);
    tensor_owner separate1(out1_bytes + guard_bytes);
    tensor_owner pair0(out0_bytes + guard_bytes);
    tensor_owner pair1(out1_bytes + guard_bytes);
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, kDenseK);
    if (!x.ptr || !separate0.ptr || !separate1.ptr || !pair0.ptr ||
        !pair1.ptr ||
        !ds4_gpu_tensor_write(x.ptr, 0, activation.data(), x_bytes) ||
        !prepare_guard(x.ptr, x_bytes, 0x3000u)) {
        std::fprintf(stderr, "pair N=%u: tensor setup failed\n", n_tokens);
        return false;
    }
    const arm baseline = {
        "two_dense_mmq",
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       separate0.ptr, model.data, model.size,
                       model.weights[set].dense_offset, kQ4Type, kDenseK,
                       kDenseM, x.ptr, n_tokens) != 0 &&
                   ds4_gpu_matmul_quant_tensor(
                       separate1.ptr, model.data, model.size,
                       model.weights[set].kv_offset, kQ4Type, kDenseK,
                       kKvM, x.ptr, n_tokens) != 0;
        }};
    const arm candidate = {
        "pair_mmq",
        [&](uint32_t set) {
            return ds4_gpu_matmul_q4_K_pair_tensor(
                       pair0.ptr, pair1.ptr, model.data, model.size,
                       model.weights[set].dense_offset,
                       model.weights[set].kv_offset, kDenseK, kDenseM, kKvM,
                       x.ptr, n_tokens) != 0;
        }};
    return benchmark_pair_arms(
        n_tokens, cfg, baseline, candidate,
        [&]() {
            return poison_output(separate0.ptr, out0_bytes, 0x7fc10001u,
                                 0x4000u) &&
                   poison_output(separate1.ptr, out1_bytes, 0x7fc20002u,
                                 0x5000u) &&
                   poison_output(pair0.ptr, out0_bytes, 0x7fc30003u,
                                 0x6000u) &&
                   poison_output(pair1.ptr, out1_bytes, 0x7fc40004u,
                                 0x7000u);
        },
        [&](uint32_t set) {
            return bitwise_equal(separate0.ptr, pair0.ptr, out0_bytes,
                                 "pair q_a output") &&
                   bitwise_equal(separate1.ptr, pair1.ptr, out1_bytes,
                                 "pair kv output") &&
                   output_is_finite(separate0.ptr, out0_bytes,
                                    "pair q_a output") &&
                   output_is_finite(separate1.ptr, out1_bytes,
                                    "pair kv output") &&
                   sampled_cpu_oracle(separate0.ptr, model,
                                      model.weights[set].dense_offset,
                                      activation, n_tokens, kDenseK, kDenseM,
                                      "pair q_a") &&
                   sampled_cpu_oracle(separate1.ptr, model,
                                      model.weights[set].kv_offset,
                                      activation, n_tokens, kDenseK, kKvM,
                                      "pair kv") &&
                   check_guard(x.ptr, x_bytes, 0x3000u, "pair input") &&
                   check_guard(separate0.ptr, out0_bytes, 0x4000u,
                               "pair separate q_a") &&
                   check_guard(separate1.ptr, out1_bytes, 0x5000u,
                               "pair separate kv") &&
                   check_guard(pair0.ptr, out0_bytes, 0x6000u,
                               "pair fused q_a") &&
                   check_guard(pair1.ptr, out1_bytes, 0x7000u,
                               "pair fused kv");
        }) &&
        check_guard(x.ptr, x_bytes, 0x3000u, "pair input final") &&
        check_guard(separate0.ptr, out0_bytes, 0x4000u,
                    "pair separate q_a final") &&
        check_guard(separate1.ptr, out1_bytes, 0x5000u,
                    "pair separate kv final") &&
        check_guard(pair0.ptr, out0_bytes, 0x6000u,
                    "pair fused q_a final") &&
        check_guard(pair1.ptr, out1_bytes, 0x7000u,
                    "pair fused kv final");
}

bool run_qb(const model_fixture &model, const config &cfg,
            uint32_t n_tokens) {
    uint64_t out_elements = 0;
    if (!checked_mul(n_tokens, kQbM, &out_elements)) return false;
    const uint64_t x_bytes = static_cast<uint64_t>(n_tokens) * kQbK *
                             sizeof(float);
    const uint64_t out_bytes = out_elements * sizeof(float);
    const uint64_t guard_bytes = kGuardWords * sizeof(uint32_t);
    tensor_owner x(x_bytes + guard_bytes);
    tensor_owner output(out_bytes + guard_bytes);
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, kQbK);
    if (!x.ptr || !output.ptr ||
        !ds4_gpu_tensor_write(x.ptr, 0, activation.data(), x_bytes) ||
        !prepare_guard(x.ptr, x_bytes, 0x8000u)) {
        std::fprintf(stderr, "q_b N=%u: tensor setup failed\n", n_tokens);
        return false;
    }
    const arm path = {
        cfg.path == cuda_path::mmq ? "mmq" : "legacy_q8k",
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       output.ptr, model.data, model.size,
                       model.weights[set].qb_offset, kQ4Type, kQbK, kQbM,
                       x.ptr, n_tokens) != 0;
        }};
    return benchmark_single_path(
        "q_b", n_tokens, kQbK, kQbM, cfg, path,
        [&]() {
            return poison_output(output.ptr, out_bytes, 0x7fc10001u,
                                 0x9000u);
        },
        [&](uint32_t set) {
            return output_is_finite(output.ptr, out_bytes, "q_b output") &&
                   sampled_cpu_oracle(output.ptr, model,
                                      model.weights[set].qb_offset,
                                      activation, n_tokens, kQbK, kQbM,
                                      "q_b") &&
                   check_guard(x.ptr, x_bytes, 0x8000u, "q_b input") &&
                   check_guard(output.ptr, out_bytes, 0x9000u,
                               "q_b output");
        }) &&
        check_guard(x.ptr, x_bytes, 0x8000u, "q_b input final") &&
        check_guard(output.ptr, out_bytes, 0x9000u, "q_b output final");
}

void usage(FILE *stream, const char *argv0) {
    std::fprintf(
        stream,
        "usage: %s [options]\n\n"
        "Resident CUDA Q4_K prefill kernel benchmark (CUDA event timing).\n\n"
        "  --path mmq|legacy         process-wide path (default: mmq)\n"
        "  --case all|dense|pair|qb  case to run (default: all)\n"
        "  --tokens N[,N...]         token counts, each 9..4096\n"
        "  --full                    use 9,16,17,31,32,33,128,512,4096\n"
        "  --sets N                  rotating resident weight sets (default: %u)\n"
        "  --samples N               samples/arm, multiple of 4 (default: %u)\n"
        "  --warmup N                untimed dispatches/arm (default: %u)\n"
        "  -h, --help                show this help\n\n"
        "Dense and q_b measure one immutable process path. Run separate "
        "legacy/MMQ\nprocesses (preferably ABBA/BAAB) to compare them because "
        "the CUDA backend\ncaches DS4_CUDA_MMQ on its first dispatch. Pair is "
        "an in-process ABBA/BAAB\ncomparison of two MMQ projections against "
        "the fused public pair API.\n",
        argv0, kDefaultSets, kDefaultSamples, kDefaultWarmup);
}

uint32_t parse_u32(const char *text, const char *option, uint32_t minimum,
                   uint32_t maximum) {
    char *end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno != 0 || !text[0] || !end || *end || value < minimum ||
        value > maximum) {
        std::fprintf(stderr, "invalid %s: %s\n", option, text);
        std::exit(2);
    }
    return static_cast<uint32_t>(value);
}

const char *need_value(int *index, int argc, char **argv) {
    if (*index + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", argv[*index]);
        std::exit(2);
    }
    return argv[++*index];
}

std::vector<uint32_t> parse_tokens(const char *text) {
    std::vector<uint32_t> result;
    const char *cursor = text;
    while (*cursor) {
        const char *comma = std::strchr(cursor, ',');
        const std::string item(cursor,
                               comma ? static_cast<size_t>(comma - cursor)
                                     : std::strlen(cursor));
        result.push_back(parse_u32(item.c_str(), "--tokens", 9u, 4096u));
        if (!comma) break;
        cursor = comma + 1;
        if (!*cursor) {
            std::fprintf(stderr, "invalid --tokens: trailing comma\n");
            std::exit(2);
        }
    }
    if (result.empty()) {
        std::fprintf(stderr, "--tokens cannot be empty\n");
        std::exit(2);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

config parse_options(int argc, char **argv) {
    config cfg;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            usage(stdout, argv[0]);
            std::exit(0);
        } else if (!std::strcmp(argv[i], "--path")) {
            const char *value = need_value(&i, argc, argv);
            if (!std::strcmp(value, "mmq")) cfg.path = cuda_path::mmq;
            else if (!std::strcmp(value, "legacy")) {
                cfg.path = cuda_path::legacy;
            } else {
                std::fprintf(stderr, "invalid --path: %s\n", value);
                std::exit(2);
            }
        } else if (!std::strcmp(argv[i], "--case")) {
            const char *value = need_value(&i, argc, argv);
            if (!std::strcmp(value, "all")) cfg.selected = bench_case::all;
            else if (!std::strcmp(value, "dense")) {
                cfg.selected = bench_case::dense;
            } else if (!std::strcmp(value, "pair")) {
                cfg.selected = bench_case::pair;
            } else if (!std::strcmp(value, "qb")) {
                cfg.selected = bench_case::qb;
            } else {
                std::fprintf(stderr, "invalid --case: %s\n", value);
                std::exit(2);
            }
        } else if (!std::strcmp(argv[i], "--tokens")) {
            cfg.tokens = parse_tokens(need_value(&i, argc, argv));
        } else if (!std::strcmp(argv[i], "--full")) {
            cfg.tokens = {9u, 16u, 17u, 31u, 32u, 33u, 128u, 512u, 4096u};
        } else if (!std::strcmp(argv[i], "--sets")) {
            cfg.sets = parse_u32(need_value(&i, argc, argv), "--sets", 1u,
                                 32u);
        } else if (!std::strcmp(argv[i], "--samples")) {
            cfg.samples = parse_u32(need_value(&i, argc, argv), "--samples",
                                    4u, 1000u);
        } else if (!std::strcmp(argv[i], "--warmup")) {
            cfg.warmup = parse_u32(need_value(&i, argc, argv), "--warmup",
                                   0u, 100u);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage(stderr, argv[0]);
            std::exit(2);
        }
    }
    if ((cfg.samples % 4u) != 0u) {
        std::fprintf(stderr,
                     "--samples must be a multiple of 4 for balanced runs\n");
        std::exit(2);
    }
    if (cfg.path == cuda_path::legacy && cfg.selected == bench_case::pair) {
        std::fprintf(stderr,
                     "--case pair requires --path mmq for prefill N > 8\n");
        std::exit(2);
    }
    return cfg;
}

bool includes(bench_case selected, bench_case wanted) {
    return selected == bench_case::all || selected == wanted;
}

bool install_resident_model(const model_fixture &model,
                            size_t *resident_delta,
                            bool *resident_delta_valid) {
    size_t free_before = 0, total_before = 0;
    size_t free_after = 0, total_after = 0;
    const bool have_before =
        cudaMemGetInfo(&free_before, &total_before) == cudaSuccess;
    if (!have_before) (void)cudaGetLastError();
    if (!ds4_gpu_set_model_map(model.data, model.size) ||
        !ds4_gpu_synchronize()) {
        return false;
    }
    const bool have_after =
        cudaMemGetInfo(&free_after, &total_after) == cudaSuccess;
    if (!have_after) (void)cudaGetLastError();
    (void)total_before;
    (void)total_after;
    *resident_delta_valid = have_before && have_after;
    *resident_delta = *resident_delta_valid && free_before >= free_after
        ? free_before - free_after : 0u;
    return true;
}

bool verify_resident_weight_ranges(const model_fixture &model) {
    const uint64_t dense_bytes = q4_weight_bytes(kDenseK, kDenseM);
    const uint64_t kv_bytes = q4_weight_bytes(kDenseK, kKvM);
    const uint64_t qb_bytes = q4_weight_bytes(kQbK, kQbM);
    for (uint32_t set = 0; set < model.weights.size(); set++) {
        struct range_desc {
            const char *name;
            uint64_t offset;
            uint64_t bytes;
        };
        const range_desc ranges[] = {
            {"dense", model.weights[set].dense_offset, dense_bytes},
            {"kv", model.weights[set].kv_offset, kv_bytes},
            {"q_b", model.weights[set].qb_offset, qb_bytes},
        };
        for (const range_desc &range : ranges) {
            if (!ds4_cuda_test_model_range_is_device_resident(
                    model.data, model.size, range.offset, range.bytes, 0)) {
                std::fprintf(
                    stderr,
                    "cuda-q4-prefill-bench: nonresident %s weight range "
                    "set=%u offset=%llu bytes=%llu\n",
                    range.name, set,
                    static_cast<unsigned long long>(range.offset),
                    static_cast<unsigned long long>(range.bytes));
                return false;
            }
        }
    }
    return true;
}

bool verify_mmq_prefill_dispatch(const model_fixture &model) {
    // For N > 8 the public pair API succeeds only through MMQ.  Probe it
    // before printing any timing to initialize and attest the process-wide
    // decision.  Strict mode separately rejects fallback on every measured
    // dense, pair, and q_b dispatch.
    constexpr uint32_t n_tokens = 9u;
    const uint64_t x_bytes = static_cast<uint64_t>(n_tokens) * kDenseK *
                             sizeof(float);
    const uint64_t out0_bytes = static_cast<uint64_t>(n_tokens) * kDenseM *
                                sizeof(float);
    const uint64_t out1_bytes = static_cast<uint64_t>(n_tokens) * kKvM *
                                sizeof(float);
    tensor_owner x(x_bytes);
    tensor_owner out0(out0_bytes);
    tensor_owner out1(out1_bytes);
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, kDenseK);
    return x.ptr && out0.ptr && out1.ptr &&
           ds4_gpu_tensor_write(x.ptr, 0, activation.data(), x_bytes) &&
           ds4_gpu_matmul_q4_K_pair_tensor(
               out0.ptr, out1.ptr, model.data, model.size,
               model.weights[0].dense_offset, model.weights[0].kv_offset,
               kDenseK, kDenseM, kKvM, x.ptr, n_tokens) &&
           ds4_gpu_synchronize();
}

}  // namespace

int main(int argc, char **argv) {
    const config cfg = parse_options(argc, argv);
    env_snapshot mmq_guard("DS4_CUDA_MMQ");
    env_snapshot copy_guard("DS4_CUDA_COPY_MODEL");
    env_snapshot pair_guard("DS4_CUDA_DISABLE_Q4_DENSE_PAIR");
    env_snapshot graph_guard("DS4_CUDA_DECODE_GRAPHS");
    if (setenv("DS4_CUDA_MMQ",
               cfg.path == cuda_path::mmq ? "1" : "0", 1) != 0 ||
        setenv("DS4_CUDA_COPY_MODEL", "1", 1) != 0 ||
        setenv("DS4_CUDA_DECODE_GRAPHS", "0", 1) != 0 ||
        unsetenv("DS4_CUDA_DISABLE_Q4_DENSE_PAIR") != 0) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: environment setup failed\n");
        return 1;
    }

    int device_count = 0;
    const cudaError_t count_rc = cudaGetDeviceCount(&device_count);
    if (count_rc != cudaSuccess || device_count <= 0) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: no visible CUDA device (%s)\n",
                     count_rc == cudaSuccess ? "device count is zero"
                                             : cudaGetErrorString(count_rc));
        return 77;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: cannot query device properties\n");
        return 1;
    }
    if (!ds4_gpu_init()) {
        std::fprintf(stderr, "cuda-q4-prefill-bench: ds4_gpu_init failed\n");
        return 1;
    }
    ds4_cuda_test_set_q4_mmq_strict(
        cfg.path == cuda_path::mmq ? 1 : 0);

    bool ok = true;
    model_fixture model;
    if (!make_model(&model, cfg.sets)) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: model fixture allocation failed\n");
        ok = false;
    }
    size_t resident_delta = 0;
    bool resident_delta_valid = false;
    if (ok) {
        ds4_gpu_set_quality(false);
        ds4_gpu_set_ssd_streaming(false);
        if (!install_resident_model(
                model, &resident_delta, &resident_delta_valid)) {
            std::fprintf(
                stderr,
                "cuda-q4-prefill-bench: model-map installation failed\n");
            ok = false;
        } else if (!verify_resident_weight_ranges(model)) {
            std::fprintf(
                stderr,
                "cuda-q4-prefill-bench: explicit CUDA model provenance "
                "check failed; refusing PCIe/HMM-contaminated timings\n");
            ok = false;
        }
    }
    if (ok && cfg.path == cuda_path::mmq &&
        !verify_mmq_prefill_dispatch(model)) {
        std::fprintf(stderr,
                     "cuda-q4-prefill-bench: MMQ prefill proof probe failed; "
                     "refusing to label fallback timings as MMQ\n");
        ok = false;
    }

    if (ok) {
        std::printf(
            "DS4_CUDA_Q4_PREFILL_SETUP device=%s cc=%d.%d warp=%d path=%s "
            "sets=%u resident_payload_mib=%.2f device_free_delta_mib=%.2f "
            "device_free_delta_valid=%d timing=cuda_events "
            "ssd_streaming=off model_storage=cudaMalloc "
            "residency=backend_provenance strict_mmq=%d "
            "dispatch_stream=legacy_default\n",
            properties.name, properties.major, properties.minor,
            properties.warpSize, path_name(cfg.path), cfg.sets,
            static_cast<double>(model.payload_bytes) / 1048576.0,
            static_cast<double>(resident_delta) / 1048576.0,
            resident_delta_valid ? 1 : 0,
            cfg.path == cuda_path::mmq ? 1 : 0);
        std::fflush(stdout);
        for (uint32_t n_tokens : cfg.tokens) {
            if (includes(cfg.selected, bench_case::dense)) {
                ok = run_dense(model, cfg, n_tokens) && ok;
            }
            if (ok && includes(cfg.selected, bench_case::pair)) {
                ok = run_pair(model, cfg, n_tokens) && ok;
            }
            if (ok && includes(cfg.selected, bench_case::qb)) {
                ok = run_qb(model, cfg, n_tokens) && ok;
            }
            if (!ok) break;
        }
    }

    ds4_gpu_cleanup();
    std::fprintf(stderr, "cuda-q4-prefill-bench: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
