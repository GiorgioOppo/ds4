/* Native ROCm V4.1 attention-output regression. No-argument execution runs
 * 48 resident cases plus 18 quality/SSD/tile8-rollback cases. To select one:
 *   ds4-kernel-v41-q4-output A_type B_type world rank rows mode
 * Types are 8 (Q8_0) or 12 (Q4_K); modes: 0 resident, 1 quality, 2 SSD,
 * 3 tile8 rollback. Rows up to 8192 are supported for optional large tests.
 *
 * Production strides, independently decoded weights and CPU Q8_K activation
 * quantization check sampled dots. Every low/output element is inspected for
 * writes/finite values, and every low must retain the BF16 boundary. The A
 * inputs and dequantized weights are exactly representable in half so both
 * resident WMMA and canonical paths share this oracle; this fixture does not
 * measure general-input WMMA drift. B uses the downloaded, validated BF16 low.
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

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); return 0; \
} } while (0)
enum { GUARD = 64, K_A = 4096, M_A = 1024, GROUPS = 8, K_B = 8192, M_B = 5120 };
typedef struct { uint16_t d; int8_t q[32]; } block_q8;
typedef struct { uint16_t d, dmin; uint8_t scales[12], q[128]; } block_q4;
typedef struct { ds4_gpu_tensor *storage, *view; size_t bytes; } guarded_tensor;
typedef struct {
    unsigned a_type, b_type;
    size_t a_bytes, b_bytes, bytes;
    unsigned char *data;
    FILE *file;
} fixture;

static uint32_t random_u32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return *state = x;
}

static size_t row_bytes(unsigned type, unsigned k) {
    return type == 12 ? (size_t)(k / 256) * sizeof(block_q4)
                      : (size_t)(k / 32) * sizeof(block_q8);
}

static float half_float(uint16_t h) {
    const int sign = (h & 0x8000) ? -1 : 1;
    const unsigned exponent = (h >> 10) & 31, mantissa = h & 1023;
    return sign * (exponent ? ldexpf((float)(1024 + mantissa), (int)exponent - 25)
                            : ldexpf((float)mantissa, -24));
}

static float bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    bits &= 0xffff0000;
    memcpy(&x, &bits, sizeof(x));
    return x;
}

static int is_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return !(bits & 0xffff);
}

static void fill_weights(void *data, unsigned type, unsigned rows, unsigned k,
                         uint32_t state) {
    /* Every physical row and block, including the last TP column, differs. */
    if (type == 12) {
        block_q4 *w = data;
        for (size_t i = 0; i < (size_t)rows * (k / 256); ++i) {
            w[i].d = 0x1800;       /* 1/512 */
            w[i].dmin = 0x1400;    /* 1/1024: nonzero min corrections */
            for (unsigned j = 0; j < 12; ++j) w[i].scales[j] = random_u32(&state) >> 24;
            for (unsigned j = 0; j < 128; ++j) w[i].q[j] = random_u32(&state) >> 24;
        }
    } else {
        block_q8 *w = data;
        for (size_t i = 0; i < (size_t)rows * (k / 32); ++i) {
            w[i].d = 0x2000;       /* 1/128 */
            for (unsigned j = 0; j < 32; ++j) w[i].q[j] = (int)(random_u32(&state) % 31) - 15;
        }
    }
}

static int fixture_init(fixture *f, unsigned a_type, unsigned b_type) {
    memset(f, 0, sizeof(*f));
    CHECK(sizeof(block_q4) == 144 && sizeof(block_q8) == 34);
    f->a_type = a_type; f->b_type = b_type;
    f->a_bytes = GROUPS * M_A * row_bytes(a_type, K_A);
    f->b_bytes = M_B * row_bytes(b_type, K_B);
    f->bytes = f->a_bytes + f->b_bytes;
    CHECK(!posix_memalign((void **)&f->data, 4096, f->bytes));
    fill_weights(f->data, a_type, GROUPS * M_A, K_A, 0x13789abc);
    fill_weights(f->data + f->a_bytes, b_type, M_B, K_B, 0x94857def);
    return 1;
}

static int fixture_file(fixture *f) {
    if (f->file) return 1;
    f->file = tmpfile();
    CHECK(f->file);
    CHECK(fwrite(f->data, 1, f->bytes, f->file) == f->bytes);
    CHECK(!fflush(f->file));
    return 1;
}

static void fixture_free(fixture *f) {
    /* Release registered/cache pointers before releasing their backing map. */
    ds4_gpu_set_ssd_streaming(false);
    (void)ds4_gpu_set_model_fd(-1);
    if (f->file) fclose(f->file);
    free(f->data);
}

static int guarded_init(guarded_tensor *t, const float *data, size_t elements) {
    memset(t, 0, sizeof(*t));
    t->bytes = elements * sizeof(float);
    t->storage = ds4_gpu_tensor_alloc(t->bytes + 2 * GUARD);
    CHECK(t->storage);
    unsigned char guard[GUARD];
    memset(guard, 0xa5, sizeof(guard));
    CHECK(ds4_gpu_tensor_write(t->storage, 0, guard, sizeof(guard)));
    CHECK(ds4_gpu_tensor_write(t->storage, GUARD + t->bytes, guard, sizeof(guard)));
    t->view = ds4_gpu_tensor_view(t->storage, GUARD, t->bytes);
    CHECK(t->view && ds4_gpu_tensor_write(t->view, 0, data, t->bytes));
    return 1;
}

static int guarded_check(const guarded_tensor *t) {
    unsigned char before[GUARD], after[GUARD];
    CHECK(ds4_gpu_tensor_read(t->storage, 0, before, sizeof(before)));
    CHECK(ds4_gpu_tensor_read(t->storage, GUARD + t->bytes, after, sizeof(after)));
    for (unsigned i = 0; i < GUARD; ++i) CHECK(before[i] == 0xa5 && after[i] == 0xa5);
    return 1;
}

static void guarded_free(guarded_tensor *t) {
    ds4_gpu_tensor_free(t->view);
    ds4_gpu_tensor_free(t->storage);
}

/* Independent scalar GGUF decoding, including the packed high scale bits. */
static float weight_at(const unsigned char *row, unsigned type, unsigned k) {
    if (type == 8) {
        const block_q8 *b = (const block_q8 *)row + k / 32;
        return half_float(b->d) * b->q[k % 32];
    }
    const block_q4 *b = (const block_q4 *)row + k / 256;
    const unsigned j = k % 256, group = j / 32;
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

/* Q8_K has one signed F32 scale per 256 activations, not CUDA Q8_1's
 * half scale per 32. The strict greater-than tree resolves signed ties. */
static void quantize_q8_k(const float *x, float *dequant, unsigned k) {
    for (unsigned b = 0; b < k; b += 256) {
        float absolute[256], value[256];
        for (unsigned i = 0; i < 256; ++i) {
            absolute[i] = fabsf(x[b + i]); value[i] = x[b + i];
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
            long q = lrintf(iscale * x[b + i]);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            dequant[b + i] = d * q;
        }
    }
}

static void fill_input(float *x, unsigned rows, unsigned groups, unsigned rank) {
    for (unsigned t = 0; t < rows; ++t)
        for (unsigned g = 0; g < groups; ++g)
            for (unsigned b = 0; b < K_A / 256; ++b) {
                float *p = x + ((size_t)t * groups + g) * K_A + b * 256;
                const unsigned global_group = rank * groups + g;
                for (unsigned i = 0; i < 256; ++i)
                    p[i] = ((int)((i * 73 + t * 37 + b * 19 + global_group * 47) % 241) - 120) / 32.0f;
                p[0] = ((t + b + global_group) & 1) ? 127.0f / 32 : -127.0f / 32;
                if ((b + global_group) % 11 == 5) memset(p, 0, 256 * sizeof(float));
            }
}

static void dot_reference(const unsigned char *row, unsigned type, unsigned start,
                          const float *x, unsigned k, double *sum, double *radius) {
    double total = 0, l1 = 0;
    for (unsigned j = 0; j < k; ++j) {
        const double term = (double)weight_at(row, type, start + j) * x[j];
        total += term; l1 += fabs(term);
    }
    /* Conservative F32 product/accumulation bound, propagated through BF16
     * below. Near BF16 ties either adjacent rounding result may be legal. */
    const double nu = (2.0 * k + 1) * (FLT_EPSILON / 2);
    *sum = total; *radius = (nu / (1 - nu)) * fmax(l1, 0.001);
}

static int verify(const fixture *f, const float *x, const float *low, const float *out,
                  unsigned rows, unsigned world, unsigned rank) {
    const unsigned groups = GROUPS / world, low_dim = groups * M_A;
    const unsigned a_samples[] = {0, 1, 31, 32, 511, 512, 1023};
    const unsigned b_samples[] = {0, 1, 31, 32, 1023, 1024, 2559, 2560, 5119};
    const unsigned tokens[] = {0, rows / 2, rows - 1};
    const size_t nl = (size_t)rows * low_dim, ny = (size_t)rows * M_B;
    size_t raw_outputs = 0;
    for (size_t i = 0; i < nl; ++i) CHECK(isfinite(low[i]) && is_bf16(low[i]));
    for (size_t i = 0; i < ny; ++i) {
        CHECK(isfinite(out[i]));
        raw_outputs += !is_bf16(out[i]);
    }
    CHECK(raw_outputs); /* A TP partial must not receive final BF16 rounding. */
    float *qx = malloc(K_B * sizeof(float));
    CHECK(qx);
    unsigned checked_a = 0, checked_b = 0;
    double max_b_error = 0;
    for (unsigned ti = 0; ti < 3; ++ti) {
        const unsigned t = tokens[ti];
        if (ti && t == tokens[ti - 1]) continue;
        for (unsigned g = 0; g < groups; ++g) {
            const float *input = x + ((size_t)t * groups + g) * K_A;
            if (f->a_type == 12) { quantize_q8_k(input, qx, K_A); input = qx; }
            for (unsigned s = 0; s < sizeof(a_samples) / sizeof(a_samples[0]); ++s) {
                const unsigned r = a_samples[s], wr = (rank * groups + g) * M_A + r;
                double expected, radius;
                dot_reference(f->data + wr * row_bytes(f->a_type, K_A), f->a_type,
                              0, input, K_A, &expected, &radius);
                const float got = low[((size_t)t * groups + g) * M_A + r];
                if (got < bf16((float)(expected - radius)) || got > bf16((float)(expected + radius))) {
                    fprintf(stderr, "A mismatch token=%u group=%u row=%u got=%a oracle=%a radius=%g\n",
                            t, g, r, (double)got, expected, radius);
                    free(qx); return 0;
                }
                ++checked_a;
            }
        }
        const float *input = low + (size_t)t * low_dim;
        if (f->b_type == 12) { quantize_q8_k(input, qx, low_dim); input = qx; }
        for (unsigned s = 0; s < sizeof(b_samples) / sizeof(b_samples[0]); ++s) {
            const unsigned r = b_samples[s];
            double expected, radius;
            dot_reference(f->data + f->a_bytes + r * row_bytes(f->b_type, K_B), f->b_type,
                          rank * low_dim, input, low_dim, &expected, &radius);
            const double error = fabs(out[(size_t)t * M_B + r] - expected);
            max_b_error = fmax(max_b_error, error);
            if (error > radius) {
                fprintf(stderr, "B mismatch token=%u row=%u got=%a oracle=%a error=%g radius=%g\n",
                        t, r, (double)out[(size_t)t * M_B + r], expected, error, radius);
                free(qx); return 0;
            }
            ++checked_b;
        }
    }
    fprintf(stderr, "  checked A=%u B=%u sampled dots, low=%zu output=%zu values; max B error=%g\n",
            checked_a, checked_b, nl, ny, max_b_error);
    free(qx);
    return 1;
}

static int run_case(fixture *f, unsigned world, unsigned rank, unsigned rows, unsigned mode) {
    fprintf(stderr, "V4.1 Q4 output A=%u B=%u world=%u rank=%u rows=%u mode=%u\n",
            f->a_type, f->b_type, world, rank, rows, mode);
    const unsigned groups = GROUPS / world;
    const size_t nx = (size_t)rows * groups * K_A, nl = (size_t)rows * groups * M_A;
    const size_t ny = (size_t)rows * M_B;
    float *x = malloc(nx * sizeof(float)), *read_x = malloc(nx * sizeof(float));
    float *low = malloc(nl * sizeof(float)), *out = malloc(ny * sizeof(float));
    CHECK(x && read_x && low && out);
    fill_input(x, rows, groups, rank);
    for (size_t i = 0; i < nl; ++i) low[i] = NAN;
    for (size_t i = 0; i < ny; ++i) out[i] = NAN;
    ds4_gpu_set_ssd_streaming(mode == 2);
    ds4_gpu_set_deepseek41_model(true);
    ds4_gpu_set_quality(mode == 1);
    /* TILE8 rollback is presence-based, including an environment value 0. */
    CHECK(!(mode == 3 ? setenv("DS4_ROCM_DISABLE_Q4_PREFILL_TILE8", "1", 1)
                       : unsetenv("DS4_ROCM_DISABLE_Q4_PREFILL_TILE8")));
    CHECK(ds4_gpu_set_model_map(f->data, f->bytes));
    if (mode == 2) {
        CHECK(fixture_file(f));
        CHECK(ds4_gpu_set_model_fd(fileno(f->file)));
        ds4_gpu_set_streaming_free_reserve(0);
    } else CHECK(ds4_gpu_set_model_fd(-1));
    guarded_tensor xt, lt, yt;
    CHECK(guarded_init(&xt, x, nx) && guarded_init(&lt, low, nl) && guarded_init(&yt, out, ny));
    CHECK(ds4_gpu_dsv41_attention_output_typed_batch(yt.view, lt.view, f->data, f->bytes,
        0, f->a_bytes, f->a_type, f->b_type, xt.view, rows, world, rank));
    CHECK(ds4_gpu_synchronize());
    CHECK(guarded_check(&xt) && guarded_check(&lt) && guarded_check(&yt));
    CHECK(ds4_gpu_tensor_read(lt.view, 0, low, nl * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(yt.view, 0, out, ny * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(xt.view, 0, read_x, nx * sizeof(float)));
    CHECK(!memcmp(x, read_x, nx * sizeof(float)));
    CHECK(verify(f, x, low, out, rows, world, rank));
    CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(yt.view, lt.view, f->data, f->bytes - 1,
        0, f->a_bytes, f->a_type, f->b_type, xt.view, rows, world, rank));
    CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(yt.view, lt.view, f->data, f->bytes,
        0, f->a_bytes, f->a_type, f->b_type, xt.view, rows + 1, world, rank));
    CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(yt.view, lt.view, f->data, f->bytes,
        0, f->a_bytes, f->a_type, f->b_type, xt.view, rows, world, world));
    CHECK(ds4_gpu_synchronize());
    CHECK(guarded_check(&xt) && guarded_check(&lt) && guarded_check(&yt));
    guarded_free(&yt); guarded_free(&lt); guarded_free(&xt);
    free(out); free(low); free(read_x); free(x);
    return 1;
}

static int parse_unsigned(const char *text, unsigned *value) {
    char *end;
    errno = 0;
    unsigned long parsed = strtoul(text, &end, 10);
    if (errno || !*text || *end || parsed > 8192) return 0;
    *value = (unsigned)parsed;
    return 1;
}

static int run_all(void) {
    const unsigned types[] = {8, 12}, counts[] = {1, 9, 33, 256};
    unsigned cases = 0;
    for (unsigned a = 0; a < 2; ++a) for (unsigned b = 0; b < 2; ++b) {
        fixture f;
        CHECK(fixture_init(&f, types[a], types[b]));
        for (unsigned shard = 0; shard < 3; ++shard)
            for (unsigned n = 0; n < 4; ++n) {
                CHECK(run_case(&f, shard ? 2 : 1, shard == 2, counts[n], 0));
                ++cases;
            }
        if (a || b) for (unsigned mode = 1; mode <= 3; ++mode)
            for (unsigned n = 0; n < 2; ++n) {
                CHECK(run_case(&f, 2, 1, n ? 256 : 9, mode));
                ++cases;
            }
        fixture_free(&f);
    }
    fprintf(stderr, "PASS: %u native ROCm attention cases, sampled CPU Q4_K/Q8_K oracle and guards\n", cases);
    return 1;
}

int main(int argc, char **argv) {
    unsigned args[6];
    if (argc != 1 && argc != 7) goto usage;
    if (argc == 7) {
        for (unsigned i = 0; i < 6; ++i) if (!parse_unsigned(argv[i + 1], args + i)) goto usage;
        if ((args[0] != 8 && args[0] != 12) || (args[1] != 8 && args[1] != 12) ||
            (args[2] != 1 && args[2] != 2) || args[3] >= args[2] || !args[4] || args[5] > 3) goto usage;
    }
    /* The fixture controls dispatch knobs; inherited force/rollback switches
     * must not silently turn its resident and fallback cases into one path. */
    const char *clear[] = {"DS4_ROCM_ENABLE_Q4_PREFILL_TILE8", "DS4_ROCM_REQUIRE_Q4_PREFILL_TILE8",
        "DS4_ROCM_ENABLE_Q4_PREFILL_WMMA", "DS4_ROCM_DISABLE_Q4_PREFILL_WMMA",
        "DS4_ROCM_REQUIRE_Q4_PREFILL_WMMA"};
    for (unsigned i = 0; i < sizeof(clear) / sizeof(clear[0]); ++i)
        if (unsetenv(clear[i])) return 1;
    if (!ds4_gpu_init()) return 1;
    int ok;
    if (argc == 1) ok = run_all();
    else {
        fixture f;
        ok = fixture_init(&f, args[0], args[1]);
        if (ok) {
            ok = run_case(&f, args[2], args[3], args[4], args[5]);
            fixture_free(&f);
        }
    }
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
usage:
    fprintf(stderr, "usage: %s [A_type B_type world rank rows mode]\n"
            "types: 8 or 12; world: 1 or 2; rows: 1..8192; modes: 0 resident, 1 quality, 2 SSD, 3 tile8-off\n", argv[0]);
    return 2;
}
