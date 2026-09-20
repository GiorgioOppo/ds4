/* Model-free regression for the public Q4_K projection API used by V4/V4.1.
 *
 * No arguments: 108 decode/micro-batch/prefill cases (six shapes, six row
 * counts, resident/quality/SSD), plus six resident 256-row WMMA-eligible cases.
 * Select one: ds4-kernel-q4-attention-projections shape rows mode
 * Shapes 0..5: V4 QA/QB/KV, V4.1 QA/QB/KV. Modes 0/1/2: resident/quality/SSD.
 *
 * The fixture is exactly representable both in F16 and after Q8_K activation
 * quantization, so the independent double-precision dot applies to both the
 * WMMA and canonical paths. This does not measure general-input WMMA drift.
 * Device input/output guards, all output writes/finite values, host model
 * guards and model immutability are checked. The host model checks do not
 * inspect a separate device copy of the weights.
 */
#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)
enum { GUARD = 64, MODEL_OFFSET = 4096, Q4_K = 12 };
typedef struct { uint16_t d, dmin; uint8_t scales[12], q[128]; } block_q4;
typedef struct { const char *name; unsigned k, m, v41; } shape;
/* DS4_SHAPE_FLASH/FLASH41 and validate_weights() in ds4.c. */
static const shape shapes[] = {
    {"V4 QA", 4096, 1024, 0}, {"V4 QB", 1024, 32768, 0},
    {"V4 KV", 4096, 512, 0}, {"V4.1 QA", 5120, 1280, 1},
    {"V4.1 QB", 1280, 32768, 1}, {"V4.1 KV", 5120, 512, 1},
};
typedef struct {
    unsigned char *model;
    size_t bytes, weight_bytes;
    uint64_t hash;
    FILE *file;
} fixture;
typedef struct { ds4_gpu_tensor *storage, *view; size_t bytes; } guarded;

static uint32_t random_u32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return *state = x;
}

static uint64_t hash_bytes(const void *ptr, size_t n) {
    const unsigned char *p = ptr;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; ++i) hash = (hash ^ p[i]) * UINT64_C(1099511628211);
    return hash;
}

static float half_float(uint16_t h) {
    const int sign = (h & 0x8000) ? -1 : 1;
    const unsigned exponent = (h >> 10) & 31, mantissa = h & 1023;
    return sign * (exponent ? ldexpf((float)(1024 + mantissa), (int)exponent - 25)
                            : ldexpf((float)mantissa, -24));
}

/* Scalar GGUF decode, independent of the device's packed integer dot. */
static float weight_at(const block_q4 *row, unsigned column) {
    const block_q4 *b = row + column / 256;
    const unsigned j = column % 256, group = j / 32;
    unsigned scale, min;
    if (group < 4) {
        scale = b->scales[group] & 63;
        min = b->scales[group + 4] & 63;
    } else {
        scale = (b->scales[group + 4] & 15) | ((b->scales[group - 4] >> 6) << 4);
        min = (b->scales[group + 4] >> 4) | ((b->scales[group] >> 6) << 4);
    }
    const unsigned packed = b->q[(j / 64) * 32 + j % 32];
    const unsigned q = (packed >> ((group & 1) * 4)) & 15;
    return half_float(b->d) * scale * q - half_float(b->dmin) * min;
}

/* Reproduce the mathematical Q8_K format, including signed maximum ties,
 * without reusing the optimized device quantizer or its reduction helper. */
static void quantize_q8_k(const float *input, float *dequant, unsigned k) {
    for (unsigned b = 0; b < k; b += 256) {
        float absolute[256], value[256];
        for (unsigned i = 0; i < 256; ++i) {
            absolute[i] = fabsf(input[b + i]); value[i] = input[b + i];
        }
        for (unsigned stride = 128; stride; stride /= 2)
            for (unsigned i = 0; i < stride; ++i)
                if (absolute[i + stride] > absolute[i]) {
                    absolute[i] = absolute[i + stride]; value[i] = value[i + stride];
                }
        if (absolute[0] == 0) {
            memset(dequant + b, 0, 256 * sizeof(float));
            continue;
        }
        const float iscale = -127.0f / value[0], d = 1.0f / iscale;
        for (unsigned i = 0; i < 256; ++i) {
            long q = lrintf(iscale * input[b + i]);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            dequant[b + i] = d * q;
        }
    }
}

static void fill_input(float *x, unsigned rows, unsigned k) {
    for (unsigned t = 0; t < rows; ++t)
        for (unsigned b = 0; b < k / 256; ++b) {
            float *p = x + (size_t)t * k + b * 256;
            for (unsigned i = 0; i < 256; ++i)
                p[i] = ((int)((i * 73 + t * 37 + b * 19) % 241) - 120) / 32.0f;
            p[0] = ((t + b) & 1) ? 127.0f / 32 : -127.0f / 32;
            p[128] = -p[0]; /* Equal-magnitude extrema of opposite sign. */
            if ((b + t) % 11 == 5) memset(p, 0, 256 * sizeof(float));
        }
}

static int fixture_init(fixture *f, const shape *s) {
    memset(f, 0, sizeof(*f));
    CHECK(sizeof(block_q4) == 144);
    f->weight_bytes = (size_t)s->m * (s->k / 256) * sizeof(block_q4);
    f->bytes = MODEL_OFFSET + f->weight_bytes + GUARD;
    CHECK(!posix_memalign((void **)&f->model, 4096, f->bytes));
    memset(f->model, 0xa5, f->bytes);
    block_q4 *w = (block_q4 *)(f->model + MODEL_OFFSET);
    uint32_t seed = 0x178abc39;
    for (size_t i = 0; i < f->weight_bytes / sizeof(*w); ++i) {
        /* Decoded weights are multiples of 1/1024 in [-2, 2], hence F16-exact. */
        w[i].d = 0x1800; w[i].dmin = 0x1400;
        for (unsigned j = 0; j < 12; ++j) w[i].scales[j] = random_u32(&seed) >> 24;
        for (unsigned j = 0; j < 128; ++j) w[i].q[j] = random_u32(&seed) >> 24;
    }
    f->hash = hash_bytes(f->model, f->bytes);
    return 1;
}

static int fixture_file(fixture *f) {
    if (f->file) return 1;
    f->file = tmpfile();
    CHECK(f->file && fwrite(f->model, 1, f->bytes, f->file) == f->bytes);
    CHECK(!fflush(f->file));
    return 1;
}

static void fixture_free(fixture *f) {
    /* Drain/release registered or cached model ranges before freeing the map. */
    ds4_gpu_set_ssd_streaming(false);
    (void)ds4_gpu_set_model_fd(-1);
    if (f->file) fclose(f->file);
    free(f->model);
}

static int guarded_init(guarded *t, const float *x, size_t elements) {
    memset(t, 0, sizeof(*t));
    t->bytes = elements * sizeof(float);
    t->storage = ds4_gpu_tensor_alloc(t->bytes + 2 * GUARD);
    CHECK(t->storage);
    unsigned char sentinel[GUARD];
    memset(sentinel, 0xa5, sizeof(sentinel));
    CHECK(ds4_gpu_tensor_write(t->storage, 0, sentinel, GUARD));
    CHECK(ds4_gpu_tensor_write(t->storage, GUARD + t->bytes, sentinel, GUARD));
    t->view = ds4_gpu_tensor_view(t->storage, GUARD, t->bytes);
    CHECK(t->view && ds4_gpu_tensor_write(t->view, 0, x, t->bytes));
    return 1;
}

static int guarded_check(const guarded *t) {
    unsigned char before[GUARD], after[GUARD];
    CHECK(ds4_gpu_tensor_read(t->storage, 0, before, GUARD));
    CHECK(ds4_gpu_tensor_read(t->storage, GUARD + t->bytes, after, GUARD));
    for (unsigned i = 0; i < GUARD; ++i) CHECK(before[i] == 0xa5 && after[i] == 0xa5);
    return 1;
}

static void guarded_free(guarded *t) {
    ds4_gpu_tensor_free(t->view);
    ds4_gpu_tensor_free(t->storage);
}

static int verify(const fixture *f, const shape *s, const float *x,
                  const float *out, unsigned rows) {
    const unsigned tokens[] = {0, rows / 2, rows - 1};
    const unsigned columns[] = {0, 1, 15, 16, 31, 32, 63, 64, 127, 128,
                               255, 256, s->m / 2 - 1, s->m / 2, s->m - 1};
    for (size_t i = 0; i < (size_t)rows * s->m; ++i) CHECK(isfinite(out[i]));
    float *qx = malloc(s->k * sizeof(float));
    CHECK(qx);
    double max_error = 0;
    unsigned dots = 0;
    for (unsigned ti = 0; ti < 3; ++ti) {
        const unsigned t = tokens[ti];
        if (ti && t == tokens[ti - 1]) continue;
        quantize_q8_k(x + (size_t)t * s->k, qx, s->k);
        /* A failed exactness precondition invalidates the shared WMMA oracle. */
        for (unsigned j = 0; j < s->k; ++j) CHECK(qx[j] == x[(size_t)t * s->k + j]);
        for (unsigned c = 0; c < sizeof(columns) / sizeof(columns[0]); ++c) {
            const unsigned r = columns[c];
            const block_q4 *w = (const block_q4 *)(f->model + MODEL_OFFSET) +
                (size_t)r * (s->k / 256);
            double sum = 0, l1 = 0;
            for (unsigned j = 0; j < s->k; ++j) {
                const double term = (double)weight_at(w, j) * qx[j];
                sum += term; l1 += fabs(term);
            }
            /* Conservative F32 multiply/reduction bound relative to this
             * dot's absolute term sum, rather than a fixed global tolerance. */
            const double nu = (2.0 * s->k + 1) * (FLT_EPSILON / 2);
            const double radius = (nu / (1 - nu)) * fmax(l1, 0.001);
            const double got = out[(size_t)t * s->m + r], error = fabs(got - sum);
            if (error > radius) {
                fprintf(stderr, "%s mismatch token=%u row=%u got=%a oracle=%a error=%g bound=%g\n",
                        s->name, t, r, got, sum, error, radius);
                free(qx); return 0;
            }
            max_error = fmax(max_error, error); ++dots;
        }
    }
    free(qx);
    fprintf(stderr, "  %u sampled independent Q4_K/Q8_K dots; %zu finite outputs; max error=%g\n",
            dots, (size_t)rows * s->m, max_error);
    return 1;
}

static int run_case(fixture *f, const shape *s, unsigned rows, unsigned mode) {
    fprintf(stderr, "%s Q4_K K=%u M=%u N=%u mode=%u\n", s->name, s->k, s->m, rows, mode);
    const size_t nx = (size_t)rows * s->k, ny = (size_t)rows * s->m;
    float *x = malloc(nx * sizeof(float)), *read_x = malloc(nx * sizeof(float));
    float *out = malloc(ny * sizeof(float));
    CHECK(x && read_x && out);
    fill_input(x, rows, s->k);
    for (size_t i = 0; i < ny; ++i) out[i] = NAN;
    ds4_gpu_set_ssd_streaming(mode == 2);
    ds4_gpu_set_deepseek41_model(s->v41 != 0);
    ds4_gpu_set_quality(mode == 1);
    CHECK(ds4_gpu_set_model_map(f->model, f->bytes));
    if (mode == 2) {
        CHECK(fixture_file(f) && ds4_gpu_set_model_fd(fileno(f->file)));
        ds4_gpu_set_streaming_free_reserve(0);
    } else CHECK(ds4_gpu_set_model_fd(-1));
    guarded xt, yt;
    CHECK(guarded_init(&xt, x, nx) && guarded_init(&yt, out, ny));
    CHECK(ds4_gpu_matmul_quant_tensor(yt.view, f->model, f->bytes, MODEL_OFFSET,
        Q4_K, s->k, s->m, xt.view, rows));
    CHECK(ds4_gpu_synchronize() && guarded_check(&xt) && guarded_check(&yt));
    CHECK(ds4_gpu_tensor_read(yt.view, 0, out, ny * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(xt.view, 0, read_x, nx * sizeof(float)));
    CHECK(!memcmp(x, read_x, nx * sizeof(float)));
    CHECK(hash_bytes(f->model, f->bytes) == f->hash);
    CHECK(verify(f, s, x, out, rows));
    CHECK(!ds4_gpu_matmul_quant_tensor(yt.view, f->model,
        MODEL_OFFSET + f->weight_bytes - 1, MODEL_OFFSET,
        Q4_K, s->k, s->m, xt.view, rows));
    CHECK(!ds4_gpu_matmul_quant_tensor(yt.view, f->model, f->bytes, MODEL_OFFSET,
        Q4_K, s->k, s->m, xt.view, rows + 1));
    CHECK(!ds4_gpu_matmul_quant_tensor(yt.view, f->model, f->bytes, MODEL_OFFSET,
        Q4_K, s->k - 1, s->m, xt.view, rows));
    CHECK(ds4_gpu_synchronize() && guarded_check(&xt) && guarded_check(&yt));
    guarded_free(&yt); guarded_free(&xt);
    free(out); free(read_x); free(x);
    return 1;
}

static int run_all(void) {
    const unsigned counts[] = {1, 2, 8, 9, 32, 33};
    unsigned cases = 0;
    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
        fixture f;
        CHECK(fixture_init(&f, shapes + s));
        for (unsigned mode = 0; mode < 3; ++mode)
            for (unsigned n = 0; n < sizeof(counts) / sizeof(counts[0]); ++n) {
                CHECK(run_case(&f, shapes + s, counts[n], mode)); ++cases;
            }
        CHECK(run_case(&f, shapes + s, 256, 0)); ++cases;
        fixture_free(&f);
    }
    fprintf(stderr, "PASS: %u public-API ROCm Q4 attention projection cases\n", cases);
    return 1;
}

static int parse_unsigned(const char *text, unsigned *value) {
    char *end;
    errno = 0;
    unsigned long n = strtoul(text, &end, 10);
    if (errno || !*text || *end || n > 8192) return 0;
    *value = (unsigned)n;
    return 1;
}

int main(int argc, char **argv) {
    unsigned selected = 0, rows = 0, mode = 0;
    if (argc != 1 && argc != 4) goto usage;
    if (argc == 4 && (!parse_unsigned(argv[1], &selected) ||
        !parse_unsigned(argv[2], &rows) || !parse_unsigned(argv[3], &mode) ||
        selected >= sizeof(shapes) / sizeof(shapes[0]) || !rows || mode > 2)) goto usage;
    /* Remove inherited overrides so the fixture follows default dispatch. */
    const char *clear[] = {
        "DS4_ROCM_ENABLE_Q4_PREFILL_TILE8", "DS4_ROCM_DISABLE_Q4_PREFILL_TILE8",
        "DS4_ROCM_REQUIRE_Q4_PREFILL_TILE8", "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA",
        "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA", "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA",
        "DS4_ROCM_REQUIRE_Q4_PREFILL_K1024_TILE4", "DS4_ROCM_DISABLE_Q4_PREFILL_K1024_TILE4",
    };
    for (unsigned i = 0; i < sizeof(clear) / sizeof(clear[0]); ++i)
        if (unsetenv(clear[i])) return 1;
    if (!ds4_gpu_init()) return 1;
    int ok;
    if (argc == 1) ok = run_all();
    else {
        fixture f;
        ok = fixture_init(&f, shapes + selected);
        if (ok) {
            ok = run_case(&f, shapes + selected, rows, mode);
            fixture_free(&f);
        }
    }
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
usage:
    fprintf(stderr, "usage: %s [shape rows mode]\n"
        "shape: 0..5 (V4 QA/QB/KV, V4.1 QA/QB/KV); rows: 1..8192; mode: 0 resident, 1 quality, 2 SSD\n",
        argv[0]);
    return 2;
}
