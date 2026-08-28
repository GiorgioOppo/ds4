// SPDX-License-Identifier: MIT
// Resident, GPU-event-only ROCm Q4_K prefill microbenchmark.

#include "ds4_gpu.h"

#include <hip/hip_runtime.h>

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
#include <sys/mman.h>
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

constexpr const char *kPrefillEnable =
    "DS4_ROCM_ENABLE_Q4_PREFILL_TILE8";
constexpr const char *kPrefillDisable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_TILE8";
constexpr const char *kPrefillRequire =
    "DS4_ROCM_REQUIRE_Q4_PREFILL_TILE8";
constexpr const char *kK1024Tile4Disable =
    "DS4_ROCM_DISABLE_Q4_PREFILL_K1024_TILE4";
constexpr const char *kK1024Tile4SsdEnable =
    "DS4_ROCM_ENABLE_Q4_PREFILL_K1024_TILE4_SSD";
constexpr const char *kK1024Tile4Require =
    "DS4_ROCM_REQUIRE_Q4_PREFILL_K1024_TILE4";

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

struct config {
    bench_case selected = bench_case::all;
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
    uint64_t resident_bytes = 0;
    FILE *file = nullptr;
    std::vector<weight_set> weights;
    std::vector<uint64_t> span_offsets;
    std::vector<uint64_t> span_sizes;

    ~model_fixture() {
        if (data && size != 0u) (void)munmap(data, static_cast<size_t>(size));
        if (file) std::fclose(file);
    }
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
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;

    event_timer() {
        if (hipEventCreate(&begin) != hipSuccess ||
            hipEventCreate(&end) != hipSuccess) {
            std::fprintf(stderr,
                         "rocm-q4-prefill-bench: HIP event allocation failed\n");
            std::exit(1);
        }
    }
    ~event_timer() {
        if (begin) (void)hipEventDestroy(begin);
        if (end) (void)hipEventDestroy(end);
    }

    bool measure(const std::function<bool()> &dispatch, float *milliseconds) {
        if (hipEventRecord(begin, 0) != hipSuccess) return false;
        if (!dispatch()) return false;
        if (hipEventRecord(end, 0) != hipSuccess ||
            hipEventSynchronize(end) != hipSuccess ||
            hipEventElapsedTime(milliseconds, begin, end) != hipSuccess) {
            return false;
        }
        return true;
    }
};

struct arm {
    const char *name;
    std::function<void()> prepare;
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
        // Positive, finite FP16 scales. The payload is deterministic but does
        // not need a CPU oracle: the benchmark compares production GPU paths.
        blocks[i].d = static_cast<uint16_t>(0x2400u + (lcg_next(&state) & 0xffu));
        blocks[i].dmin =
            static_cast<uint16_t>(0x2000u + (lcg_next(&state) & 0xffu));
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
    model->span_offsets.reserve(static_cast<size_t>(sets) * 3u);
    model->span_sizes.reserve(static_cast<size_t>(sets) * 3u);
    uint64_t cursor = 0;
    auto append = [&](uint64_t bytes) {
        const uint64_t offset = align_up(cursor, page);
        cursor = offset + bytes;
        model->span_offsets.push_back(offset);
        model->span_sizes.push_back(bytes);
        model->resident_bytes += bytes;
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
    auto *staging = static_cast<uint8_t *>(storage);
    std::memset(staging, 0, static_cast<size_t>(model->size));
    for (uint32_t i = 0; i < sets; i++) {
        fill_q4(staging + model->weights[i].dense_offset, dense_bytes,
                0x243f6a88u ^ (i * 0x9e3779b9u));
        fill_q4(staging + model->weights[i].kv_offset, kv_bytes,
                0x85a308d3u ^ (i * 0x7f4a7c15u));
        fill_q4(staging + model->weights[i].qb_offset, qb_bytes,
                0x13198a2eu ^ (i * 0x94d049bbu));
    }

    FILE *file = std::tmpfile();
    if (!file ||
        std::fwrite(staging, 1u, static_cast<size_t>(model->size), file) !=
            static_cast<size_t>(model->size) ||
        std::fflush(file) != 0) {
        std::free(staging);
        if (file) std::fclose(file);
        return false;
    }
    void *mapping = mmap(nullptr, static_cast<size_t>(model->size), PROT_READ,
                         MAP_PRIVATE, fileno(file), 0);
    std::free(staging);
    if (mapping == MAP_FAILED) {
        std::fclose(file);
        return false;
    }
    model->data = static_cast<uint8_t *>(mapping);
    model->file = file;
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

std::vector<uint32_t> guard_pattern() {
    std::vector<uint32_t> guard(kGuardWords);
    for (uint32_t i = 0; i < kGuardWords; i++) guard[i] = 0x7fc12000u + i;
    return guard;
}

bool prepare_guard(ds4_gpu_tensor *tensor, uint64_t logical_bytes) {
    const std::vector<uint32_t> guard = guard_pattern();
    return ds4_gpu_tensor_write(tensor, logical_bytes, guard.data(),
                                guard.size() * sizeof(guard[0])) != 0;
}

bool poison_output(ds4_gpu_tensor *tensor, uint64_t logical_bytes,
                   uint32_t pattern) {
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
    return prepare_guard(tensor, logical_bytes);
}

bool check_guard(const ds4_gpu_tensor *tensor, uint64_t logical_bytes,
                 const char *label) {
    const std::vector<uint32_t> expected = guard_pattern();
    std::vector<uint32_t> got(expected.size());
    if (!ds4_gpu_tensor_read(tensor, logical_bytes, got.data(),
                             got.size() * sizeof(got[0]))) {
        std::fprintf(stderr, "%s: guard read failed\n", label);
        return false;
    }
    if (got != expected) {
        const auto mismatch = std::mismatch(got.begin(), got.end(),
                                            expected.begin());
        std::fprintf(stderr, "%s: output guard overwritten at word %zu\n",
                     label,
                     static_cast<size_t>(mismatch.first - got.begin()));
        return false;
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
        if (std::memcmp(lhs.data(), rhs.data(), static_cast<size_t>(count)) != 0) {
            uint64_t first = 0;
            while (first < count && lhs[static_cast<size_t>(first)] ==
                                         rhs[static_cast<size_t>(first)]) {
                first++;
            }
            std::fprintf(stderr,
                         "%s: bitwise mismatch at output byte %llu\n", label,
                         static_cast<unsigned long long>(offset + first));
            return false;
        }
    }
    return true;
}

void select_legacy() {
    (void)unsetenv(kPrefillEnable);
    (void)setenv(kPrefillDisable, "1", 1);
    (void)unsetenv(kPrefillRequire);
    (void)unsetenv(kK1024Tile4Disable);
    (void)unsetenv(kK1024Tile4SsdEnable);
    (void)unsetenv(kK1024Tile4Require);
}

void select_tile8(bool disable_k1024_tile4) {
    (void)unsetenv(kPrefillEnable);
    (void)unsetenv(kPrefillDisable);
    (void)setenv(kPrefillRequire, "1", 1);
    (void)unsetenv(kK1024Tile4SsdEnable);
    (void)unsetenv(kK1024Tile4Require);
    if (disable_k1024_tile4) {
        (void)setenv(kK1024Tile4Disable, "1", 1);
    } else {
        (void)unsetenv(kK1024Tile4Disable);
    }
}

void select_k1024_tile4() {
    select_tile8(false);
    (void)setenv(kK1024Tile4Require, "1", 1);
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

bool benchmark_arms(const char *case_name, uint32_t n_tokens, uint32_t in_dim,
                    uint32_t out_dim, const config &cfg, const arm &baseline,
                    const arm &candidate,
                    const std::function<bool()> &oracle_prepare,
                    const std::function<bool()> &oracle) {
    // Validate every rotating weight set and prime reusable Q8_K scratch
    // before any timed event.  This catches data-dependent path errors without
    // admitting readback or comparison work into the HIP-event interval.
    for (uint32_t set = 0; set < cfg.sets; set++) {
        if (!oracle_prepare()) {
            std::fprintf(stderr,
                         "rocm-q4-prefill-bench: %s oracle poison failed "
                         "for weight set %u\n",
                         case_name, set);
            return false;
        }
        baseline.prepare();
        if (!baseline.dispatch(set) || !ds4_gpu_synchronize()) return false;
        candidate.prepare();
        if (!candidate.dispatch(set) || !ds4_gpu_synchronize()) return false;
        if (!oracle()) {
            std::fprintf(stderr,
                         "rocm-q4-prefill-bench: %s oracle failed for "
                         "weight set %u\n",
                         case_name, set);
            return false;
        }
    }

    for (uint32_t i = 0; i < cfg.warmup; i++) {
        const uint32_t set = i % cfg.sets;
        baseline.prepare();
        if (!baseline.dispatch(set) || !ds4_gpu_synchronize()) return false;
        candidate.prepare();
        if (!candidate.dispatch(set) || !ds4_gpu_synchronize()) return false;
    }

    event_timer timer;
    std::vector<double> a_samples;
    std::vector<double> b_samples;
    a_samples.reserve(cfg.samples);
    b_samples.reserve(cfg.samples);

    auto take = [&](const arm &which, uint32_t set,
                    std::vector<double> *samples) {
        which.prepare();
        float elapsed = 0.0f;
        const bool ok = timer.measure(
            [&]() { return which.dispatch(set); }, &elapsed);
        if (!ok) {
            std::fprintf(stderr,
                         "rocm-q4-prefill-bench: %s/%s timed dispatch failed\n",
                         case_name, which.name);
            return false;
        }
        samples->push_back(static_cast<double>(elapsed));
        return true;
    };

    // Each cycle contributes two samples per arm. Alternating ABBA/BAAB
    // balances first/last position, while both arms see identical weight sets.
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
    const double macs = static_cast<double>(n_tokens) * in_dim * out_dim;
    const double a_gmac_s = macs / (a.median * 1.0e6);
    const double b_gmac_s = macs / (b.median * 1.0e6);

    std::printf(
        "DS4_ROCM_Q4_PREFILL_BENCH case=%s N=%u K=%u M=%u "
        "baseline=%s candidate=%s samples=%u sets=%u "
        "baseline_ms_p50=%.6f candidate_ms_p50=%.6f "
        "baseline_ms_min=%.6f candidate_ms_min=%.6f "
        "baseline_ms_p95=%.6f candidate_ms_p95=%.6f "
        "baseline_gmac_s=%.3f candidate_gmac_s=%.3f "
        "candidate_delta_pct=%.3f paired_delta_pct_p50=%.3f "
        "speedup_pct=%.3f\n",
        case_name, n_tokens, in_dim, out_dim, baseline.name, candidate.name,
        cfg.samples, cfg.sets, a.median, b.median, a.minimum, b.minimum,
        a.p95, b.p95, a_gmac_s, b_gmac_s, median_delta, paired_median,
        speedup);
    std::fflush(stdout);
    return true;
}

bool allocate_io(uint32_t n_tokens, uint32_t in_dim, uint64_t out_elements,
                 tensor_owner *x, tensor_owner *out_a, tensor_owner *out_b) {
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, in_dim);
    if (!x->ptr || !out_a->ptr || !out_b->ptr ||
        !ds4_gpu_tensor_write(x->ptr, 0, activation.data(),
                              activation.size() * sizeof(float))) {
        return false;
    }
    const uint64_t logical_bytes = out_elements * sizeof(float);
    return poison_output(out_a->ptr, logical_bytes, 0x7fc10001u) &&
           poison_output(out_b->ptr, logical_bytes, 0x7fc20002u);
}

bool run_dense(const model_fixture &model, const config &cfg,
               uint32_t n_tokens) {
    uint64_t out_elements = 0;
    if (!checked_mul(n_tokens, kDenseM, &out_elements)) return false;
    const uint64_t logical_bytes = out_elements * sizeof(float);
    const uint64_t allocation_bytes = logical_bytes +
                                      kGuardWords * sizeof(uint32_t);
    tensor_owner x(static_cast<uint64_t>(n_tokens) * kDenseK * sizeof(float));
    tensor_owner legacy(allocation_bytes);
    tensor_owner tiled(allocation_bytes);
    if (!allocate_io(n_tokens, kDenseK, out_elements, &x, &legacy, &tiled)) {
        std::fprintf(stderr, "dense N=%u: tensor setup failed\n", n_tokens);
        return false;
    }
    const arm baseline = {
        "legacy", select_legacy,
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       legacy.ptr, model.data, model.size,
                       model.weights[set].dense_offset, kQ4Type, kDenseK,
                       kDenseM, x.ptr, n_tokens) != 0;
        }};
    const arm candidate = {
        "tile8", []() { select_tile8(false); },
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       tiled.ptr, model.data, model.size,
                       model.weights[set].dense_offset, kQ4Type, kDenseK,
                       kDenseM, x.ptr, n_tokens) != 0;
        }};
    if (!benchmark_arms(
            "dense", n_tokens, kDenseK, kDenseM, cfg, baseline, candidate,
            [&]() {
                return poison_output(legacy.ptr, logical_bytes, 0x7fc10001u) &&
                       poison_output(tiled.ptr, logical_bytes, 0x7fc20002u);
            },
            [&]() {
                return bitwise_equal(legacy.ptr, tiled.ptr, logical_bytes,
                                     "dense legacy vs tile8") &&
                       check_guard(legacy.ptr, logical_bytes,
                                   "dense legacy oracle") &&
                       check_guard(tiled.ptr, logical_bytes,
                                   "dense tile8 oracle");
            })) return false;
    return bitwise_equal(legacy.ptr, tiled.ptr, logical_bytes,
                         "dense legacy vs tile8") &&
           check_guard(legacy.ptr, logical_bytes, "dense legacy") &&
           check_guard(tiled.ptr, logical_bytes, "dense tile8");
}

bool run_pair(const model_fixture &model, const config &cfg,
              uint32_t n_tokens) {
    uint64_t out0_elements = 0, out1_elements = 0;
    if (!checked_mul(n_tokens, kDenseM, &out0_elements) ||
        !checked_mul(n_tokens, kKvM, &out1_elements)) return false;
    const uint64_t out0_bytes = out0_elements * sizeof(float);
    const uint64_t out1_bytes = out1_elements * sizeof(float);
    const uint64_t guard_bytes = kGuardWords * sizeof(uint32_t);
    tensor_owner x(static_cast<uint64_t>(n_tokens) * kDenseK * sizeof(float));
    tensor_owner separate0(out0_bytes + guard_bytes);
    tensor_owner separate1(out1_bytes + guard_bytes);
    tensor_owner pair0(out0_bytes + guard_bytes);
    tensor_owner pair1(out1_bytes + guard_bytes);
    std::vector<float> activation;
    fill_activation(&activation, n_tokens, kDenseK);
    if (!x.ptr || !separate0.ptr || !separate1.ptr || !pair0.ptr || !pair1.ptr ||
        !ds4_gpu_tensor_write(x.ptr, 0, activation.data(),
                              activation.size() * sizeof(float)) ||
        !prepare_guard(separate0.ptr, out0_bytes) ||
        !prepare_guard(separate1.ptr, out1_bytes) ||
        !prepare_guard(pair0.ptr, out0_bytes) ||
        !prepare_guard(pair1.ptr, out1_bytes)) {
        std::fprintf(stderr, "pair N=%u: tensor setup failed\n", n_tokens);
        return false;
    }
    const arm baseline = {
        "two_dense_tile8", []() { select_tile8(false); },
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       separate0.ptr, model.data, model.size,
                       model.weights[set].dense_offset, kQ4Type, kDenseK,
                       kDenseM, x.ptr, n_tokens) != 0 &&
                   ds4_gpu_matmul_quant_tensor(
                       separate1.ptr, model.data, model.size,
                       model.weights[set].kv_offset, kQ4Type, kDenseK, kKvM,
                       x.ptr, n_tokens) != 0;
        }};
    const arm candidate = {
        "pair_tile8", []() { select_tile8(false); },
        [&](uint32_t set) {
            return ds4_gpu_matmul_q4_K_pair_tensor(
                       pair0.ptr, pair1.ptr, model.data, model.size,
                       model.weights[set].dense_offset,
                       model.weights[set].kv_offset, kDenseK, kDenseM, kKvM,
                       x.ptr, n_tokens) != 0;
        }};
    if (!benchmark_arms(
            "pair", n_tokens, kDenseK, kDenseM + kKvM, cfg, baseline,
            candidate,
            [&]() {
                return poison_output(separate0.ptr, out0_bytes, 0x7fc10001u) &&
                       poison_output(separate1.ptr, out1_bytes, 0x7fc20002u) &&
                       poison_output(pair0.ptr, out0_bytes, 0x7fc30003u) &&
                       poison_output(pair1.ptr, out1_bytes, 0x7fc40004u);
            },
            [&]() {
                return bitwise_equal(separate0.ptr, pair0.ptr, out0_bytes,
                                     "pair q_a output") &&
                       bitwise_equal(separate1.ptr, pair1.ptr, out1_bytes,
                                     "pair kv output") &&
                       check_guard(separate0.ptr, out0_bytes,
                                   "pair separate q_a oracle") &&
                       check_guard(separate1.ptr, out1_bytes,
                                   "pair separate kv oracle") &&
                       check_guard(pair0.ptr, out0_bytes,
                                   "pair fused q_a oracle") &&
                       check_guard(pair1.ptr, out1_bytes,
                                   "pair fused kv oracle");
            })) return false;
    return bitwise_equal(separate0.ptr, pair0.ptr, out0_bytes,
                         "pair q_a output") &&
           bitwise_equal(separate1.ptr, pair1.ptr, out1_bytes,
                         "pair kv output") &&
           check_guard(separate0.ptr, out0_bytes, "pair separate q_a") &&
           check_guard(separate1.ptr, out1_bytes, "pair separate kv") &&
           check_guard(pair0.ptr, out0_bytes, "pair fused q_a") &&
           check_guard(pair1.ptr, out1_bytes, "pair fused kv");
}

bool run_qb(const model_fixture &model, const config &cfg,
            uint32_t n_tokens) {
    uint64_t out_elements = 0;
    if (!checked_mul(n_tokens, kQbM, &out_elements)) return false;
    const uint64_t logical_bytes = out_elements * sizeof(float);
    const uint64_t allocation_bytes = logical_bytes +
                                      kGuardWords * sizeof(uint32_t);
    tensor_owner x(static_cast<uint64_t>(n_tokens) * kQbK * sizeof(float));
    tensor_owner tile8(allocation_bytes);
    tensor_owner tile4(allocation_bytes);
    if (!allocate_io(n_tokens, kQbK, out_elements, &x, &tile8, &tile4)) {
        std::fprintf(stderr, "q_b N=%u: tensor setup failed\n", n_tokens);
        return false;
    }
    const arm baseline = {
        "tile8", []() { select_tile8(true); },
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       tile8.ptr, model.data, model.size,
                       model.weights[set].qb_offset, kQ4Type, kQbK, kQbM,
                       x.ptr, n_tokens) != 0;
        }};
    const arm candidate = {
        "tile4", select_k1024_tile4,
        [&](uint32_t set) {
            return ds4_gpu_matmul_quant_tensor(
                       tile4.ptr, model.data, model.size,
                       model.weights[set].qb_offset, kQ4Type, kQbK, kQbM,
                       x.ptr, n_tokens) != 0;
        }};
    if (!benchmark_arms(
            "q_b", n_tokens, kQbK, kQbM, cfg, baseline, candidate,
            [&]() {
                return poison_output(tile8.ptr, logical_bytes, 0x7fc10001u) &&
                       poison_output(tile4.ptr, logical_bytes, 0x7fc20002u);
            },
            [&]() {
                return bitwise_equal(tile8.ptr, tile4.ptr, logical_bytes,
                                     "q_b tile8 vs tile4") &&
                       check_guard(tile8.ptr, logical_bytes,
                                   "q_b tile8 oracle") &&
                       check_guard(tile4.ptr, logical_bytes,
                                   "q_b tile4 oracle");
            })) return false;
    return bitwise_equal(tile8.ptr, tile4.ptr, logical_bytes,
                         "q_b tile8 vs tile4") &&
           check_guard(tile8.ptr, logical_bytes, "q_b tile8") &&
           check_guard(tile4.ptr, logical_bytes, "q_b tile4");
}

void usage(FILE *stream, const char *argv0) {
    std::fprintf(
        stream,
        "usage: %s [options]\n\n"
        "Resident ROCm Q4_K prefill kernel A/B (HIP event timing only).\n\n"
        "  --case all|dense|pair|qb   comparison to run (default: all)\n"
        "  --tokens N[,N...]          token counts, each 9..4096\n"
        "  --full                     use 9,16,17,31,32,33,128,512,4096\n"
        "  --sets N                   rotating resident weight sets (default: %u)\n"
        "  --samples N                samples/arm, multiple of 4 (default: %u)\n"
        "  --warmup N                 untimed dispatches/arm (default: %u)\n"
        "  -h, --help                 show this help\n\n"
        "dense compares legacy with TILE8 at K=4096,M=1024; pair compares\n"
        "two TILE8 projections with the fused K=4096,M=(1024+512) path; qb\n"
        "compares TILE8 with TILE4 at the production K=1024,M=32768 shape.\n",
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
        } else if (!std::strcmp(argv[i], "--case")) {
            const char *value = need_value(&i, argc, argv);
            if (!std::strcmp(value, "all")) cfg.selected = bench_case::all;
            else if (!std::strcmp(value, "dense")) cfg.selected = bench_case::dense;
            else if (!std::strcmp(value, "pair")) cfg.selected = bench_case::pair;
            else if (!std::strcmp(value, "qb")) cfg.selected = bench_case::qb;
            else {
                std::fprintf(stderr, "invalid --case: %s\n", value);
                std::exit(2);
            }
        } else if (!std::strcmp(argv[i], "--tokens")) {
            cfg.tokens = parse_tokens(need_value(&i, argc, argv));
        } else if (!std::strcmp(argv[i], "--full")) {
            cfg.tokens = {9u, 16u, 17u, 31u, 32u, 33u, 128u, 512u, 4096u};
        } else if (!std::strcmp(argv[i], "--sets")) {
            cfg.sets = parse_u32(need_value(&i, argc, argv), "--sets", 1u, 32u);
        } else if (!std::strcmp(argv[i], "--samples")) {
            cfg.samples =
                parse_u32(need_value(&i, argc, argv), "--samples", 4u, 1000u);
        } else if (!std::strcmp(argv[i], "--warmup")) {
            cfg.warmup =
                parse_u32(need_value(&i, argc, argv), "--warmup", 0u, 100u);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage(stderr, argv[0]);
            std::exit(2);
        }
    }
    if ((cfg.samples % 4u) != 0u) {
        std::fprintf(stderr,
                     "--samples must be a multiple of 4 for ABBA/BAAB balance\n");
        std::exit(2);
    }
    return cfg;
}

bool includes(bench_case selected, bench_case wanted) {
    return selected == bench_case::all || selected == wanted;
}

}  // namespace

int main(int argc, char **argv) {
    const config cfg = parse_options(argc, argv);
    env_snapshot enable_guard(kPrefillEnable);
    env_snapshot disable_guard(kPrefillDisable);
    env_snapshot require_guard(kPrefillRequire);
    env_snapshot tile4_guard(kK1024Tile4Disable);
    env_snapshot tile4_ssd_guard(kK1024Tile4SsdEnable);
    env_snapshot tile4_require_guard(kK1024Tile4Require);

    int device_count = 0;
    hipError_t hip_rc = hipGetDeviceCount(&device_count);
    if (hip_rc != hipSuccess || device_count <= 0) {
        std::fprintf(stderr,
                     "rocm-q4-prefill-bench: no visible HIP device (%s)\n",
                     hip_rc == hipSuccess ? "device count is zero"
                                          : hipGetErrorString(hip_rc));
        return 77;
    }
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) {
        std::fprintf(stderr,
                     "rocm-q4-prefill-bench: cannot query device properties\n");
        return 1;
    }
    if (!ds4_gpu_init()) {
        std::fprintf(stderr, "rocm-q4-prefill-bench: ds4_gpu_init failed\n");
        return 1;
    }

    bool ok = true;
    model_fixture model;
    if (!make_model(&model, cfg.sets)) {
        std::fprintf(stderr, "rocm-q4-prefill-bench: model fixture allocation failed\n");
        ok = false;
    }
    if (ok) {
        ds4_gpu_set_ssd_streaming(false);
        const uint64_t max_tensor = q4_weight_bytes(kQbK, kQbM);
        if (!ds4_gpu_set_model_fd(fileno(model.file)) ||
            !ds4_gpu_set_model_map_spans(
                model.data, model.size, model.span_offsets.data(),
                model.span_sizes.data(),
                static_cast<uint32_t>(model.span_offsets.size()), max_tensor) ||
            !ds4_gpu_synchronize()) {
            std::fprintf(stderr,
                         "rocm-q4-prefill-bench: device-resident weight copy failed\n");
            ok = false;
        }
    }

    if (ok) {
        std::printf(
            "DS4_ROCM_Q4_PREFILL_SETUP device=%s arch=%s warp=%d sets=%u "
            "resident_mib=%.2f timing=hip_events ssd_streaming=off\n",
            properties.name, properties.gcnArchName, properties.warpSize,
            cfg.sets, static_cast<double>(model.resident_bytes) / 1048576.0);
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

    (void)ds4_gpu_set_model_fd(-1);
    ds4_gpu_cleanup();
    std::fprintf(stderr, "rocm-q4-prefill-bench: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
