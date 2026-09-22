/* Bounded Q4_K projection/BF16 oracle and resident Metal A/B benchmark.
 * No GGUF is required. Run with --bench for ABBA/BAAB dispatch timings. */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

enum { MAX_IN = 5120, MAX_OUT = 32768, MAX_ROWS = 8,
       PAD = 16, WEIGHT_OFFSET = 128, Q4_K = 12 };
enum { INPUT, OUTPUT, REFERENCE, NT };
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
typedef struct { uint32_t in, out; const char *name; } shape;
static const shape shapes[] = {
    {5120, 1280, "Q_A"}, {5120, 512, "KV"}, {1280, 32768, "Q_B"}
};
static const char *rollback = "DS4_METAL_DISABLE_V41_Q4_BF16";
static const uint32_t edges[32] = {
    0x00000000u, 0x80000000u,
    0x3f807fffu, 0x3f808000u, 0x3f808001u,
    0x3f817fffu, 0x3f818000u, 0x3f818001u,
    0xbf807fffu, 0xbf808000u, 0xbf808001u,
    0xbf817fffu, 0xbf818000u, 0xbf818001u,
    0x3c007fffu, 0x3c008000u, 0x3c008001u,
    0xbc007fffu, 0xbc008000u, 0xbc008001u,
    0x3c017fffu, 0x3c018000u, 0x3c018001u,
    0xbc017fffu, 0xbc018000u, 0xbc018001u,
    0x3f7fffffu, 0x3fffffffu, 0xbf7fffffu, 0xbfffffffu,
    0x00800000u, 0x80800000u
};
typedef struct {
    void *model;
    uint64_t model_bytes, bytes[NT];
    ds4_gpu_tensor *storage[NT], *t[NT];
    unsigned char *input_snapshot;
    char *saved_rollback;
    int had_rollback;
} fixture;

static uint32_t random_bits(void) {
    static uint32_t state = 7919;
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

static uint32_t bf16_bits(uint32_t bits) {
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return bits & 0xffff0000u;
}

static uint64_t weight_bytes(shape s) {
    return (uint64_t)s.in * s.out / 256u * sizeof(q4_block);
}

static uint64_t hash_bytes(const void *data, uint64_t bytes) {
    const unsigned char *p = data;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint64_t i = 0; i < bytes; i++) {
        hash ^= p[i]; hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int initialize(fixture *f) {
    CHECK(sizeof(q4_block) == 144);
    const char *prior = getenv(rollback);
    f->had_rollback = prior != NULL;
    f->saved_rollback = prior ? strdup(prior) : NULL;
    CHECK(!prior || f->saved_rollback);
    f->model_bytes = WEIGHT_OFFSET + weight_bytes(shapes[2]);
    CHECK(posix_memalign(&f->model, getpagesize(), f->model_bytes) == 0);
    memset(f->model, 0xa5, f->model_bytes);
    for (unsigned i = 0; i < NT; i++) {
        f->bytes[i] = (uint64_t)MAX_ROWS * (i == INPUT ? MAX_IN : MAX_OUT) * 4u;
        f->storage[i] = ds4_gpu_tensor_alloc(f->bytes[i] + 2u * PAD);
        CHECK(f->storage[i] && ds4_gpu_tensor_contents(f->storage[i]));
        memset(ds4_gpu_tensor_contents(f->storage[i]), 0xa5, f->bytes[i] + 2u * PAD);
        f->t[i] = ds4_gpu_tensor_view(f->storage[i], PAD, f->bytes[i]);
        CHECK(f->t[i] && ds4_gpu_tensor_contents(f->t[i]));
    }
    f->input_snapshot = malloc(f->bytes[INPUT] + 2u * PAD);
    CHECK(f->input_snapshot);
    CHECK(ds4_gpu_set_model_map(f->model, f->model_bytes));
    return 1;
}

static void destroy(fixture *f) {
    if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
    for (unsigned i = 0; i < NT; i++) {
        ds4_gpu_tensor_free(f->t[i]);
        ds4_gpu_tensor_free(f->storage[i]);
    }
    if (f->had_rollback && f->saved_rollback) (void)setenv(rollback, f->saved_rollback, 1);
    else if (!f->had_rollback) (void)unsetenv(rollback);
    free(f->saved_rollback);
    free(f->input_snapshot);
    ds4_gpu_cleanup();
    free(f->model);
}

static void fill(fixture *f, shape s, unsigned sparse) {
    q4_block *w = (q4_block *)((char *)f->model + WEIGHT_OFFSET);
    const uint64_t blocks = weight_bytes(s) / sizeof(*w);
    memset(w, 0, weight_bytes(s));
    for (uint64_t b = 0; b < blocks; b++) {
        w[b].d = sparse ? 0x3c00u : (uint16_t)(0x0800u + random_bits() % 0x1800u);
        w[b].dmin = sparse ? 0 : (uint16_t)(0x0800u + random_bits() % 0x1800u);
        if (!sparse) {
            for (unsigned i = 0; i < sizeof(w[b].scales); i++) w[b].scales[i] = random_bits();
            for (unsigned i = 0; i < sizeof(w[b].qs); i++) w[b].qs[i] = random_bits();
        }
    }
    if (sparse) for (uint32_t r = 0; r < s.out; r++) {
        /* The first subblock has scale 1 and a single unit coefficient.
         * Its dot product lands on an independently known BF16 boundary. */
        q4_block *b = w + (uint64_t)r * (s.in / 256u);
        b->scales[0] = 1;
        b->qs[r % 32u] = 1;
    }
    memset(ds4_gpu_tensor_contents(f->storage[INPUT]), 0xa5, f->bytes[INPUT] + 2u * PAD);
    float *x = ds4_gpu_tensor_contents(f->t[INPUT]);
    for (uint64_t i = 0; i < (uint64_t)MAX_ROWS * s.in; i++) {
        if (sparse || i % 13u == 0) {
            const uint32_t bits = edges[i % 32u];
            memcpy(x + i, &bits, sizeof(bits));
        } else {
            x[i] = ((int32_t)(random_bits() % 65537u) - 32768) / 32768.0f;
            if (i % 31u == 0) x[i] *= 0x1p-20f;
        }
    }
    memcpy(f->input_snapshot, ds4_gpu_tensor_contents(f->storage[INPUT]),
           f->bytes[INPUT] + 2u * PAD);
}

static int project(fixture *f, ds4_gpu_tensor *out, shape s, uint32_t rows, int fused) {
    /* The shared model-view cache identifies a mapping by both its address
     * and full registered length, even when a projection uses a prefix. */
    const uint64_t bytes = f->model_bytes;
    if (fused) return ds4_gpu_dsv41_q4_bf16_rows(out, f->model, bytes,
        WEIGHT_OFFSET, s.in, s.out, f->t[INPUT], rows);
    return ds4_gpu_matmul_quant_tensor(out, f->model, bytes, WEIGHT_OFFSET,
               Q4_K, s.in, s.out, f->t[INPUT], rows) &&
           ds4_gpu_dsv41_quantize(out, s.out, rows, DS4_V41_BF16);
}

static int equal_output(fixture *f, shape s, uint32_t rows, unsigned sparse) {
    const uint32_t *ref = ds4_gpu_tensor_contents(f->t[REFERENCE]);
    const uint32_t *got = ds4_gpu_tensor_contents(f->t[OUTPUT]);
    const uint64_t count = (uint64_t)s.out * rows;
    for (uint64_t i = 0; i < count; i++) {
        if (got[i] != ref[i]) {
            fprintf(stderr, "%s rows=%u sparse=%u index=%llu: separate=%08x API=%08x\n",
                    s.name, rows, sparse, (unsigned long long)i, ref[i], got[i]);
            return 0;
        }
        CHECK((got[i] & 0xffffu) == 0);
        CHECK((got[i] & 0x7f800000u) != 0x7f800000u);
        if (sparse) {
            const uint32_t bits = edges[(i % s.out) % 32u];
            /* Reduction determines the sign of exact zero. Very small
             * values may be flushed by the device; both remain bitwise
             * covered by the explicit GPU conversion oracle above. */
            if ((bits & 0x7fffffffu) > 0x00800000u)
                CHECK(got[i] == bf16_bits(bits));
        }
    }
    for (unsigned t = OUTPUT; t <= REFERENCE; t++) {
        const unsigned char *p = ds4_gpu_tensor_contents(f->storage[t]);
        for (unsigned i = 0; i < PAD; i++) CHECK(p[i] == 0xa5);
        for (uint64_t i = PAD + count * 4u; i < f->bytes[t] + 2u * PAD; i++)
            CHECK(p[i] == 0xa5);
    }
    CHECK(!memcmp(f->input_snapshot, ds4_gpu_tensor_contents(f->storage[INPUT]),
                  f->bytes[INPUT] + 2u * PAD));
    return 1;
}

static int parity(fixture *f, shape s, uint32_t rows, unsigned sparse) {
    for (unsigned t = OUTPUT; t <= REFERENCE; t++)
        memset(ds4_gpu_tensor_contents(f->storage[t]), 0xa5, f->bytes[t] + 2u * PAD);
    CHECK(ds4_gpu_begin_commands());
    CHECK(project(f, f->t[REFERENCE], s, rows, 0));
    CHECK(project(f, f->t[OUTPUT], s, rows, 1));
    CHECK(ds4_gpu_end_commands());
    return equal_output(f, s, rows, sparse);
}

static int project_output_a(fixture *f, ds4_gpu_tensor *out, uint32_t group0,
                            uint32_t groups, const ds4_gpu_tensor *heads, int fused) {
    if (fused) return ds4_gpu_dsv41_q4_output_low_bf16(out, f->model,
        f->model_bytes, WEIGHT_OFFSET, group0, groups, heads);
    return ds4_gpu_attention_output_low_q4_K_slice_tensor(out, f->model,
               f->model_bytes, WEIGHT_OFFSET, 4096, 1024, group0, groups, heads, 0) &&
           ds4_gpu_dsv41_quantize(out, groups * 1024u, 1, DS4_V41_BF16);
}

static int parity_output_a(fixture *f, uint32_t group0, uint32_t groups, unsigned sparse) {
    const shape s = {4096, groups * 1024u, "O_A"};
    for (unsigned t = OUTPUT; t <= REFERENCE; t++)
        memset(ds4_gpu_tensor_contents(f->storage[t]), 0xa5, f->bytes[t] + 2u * PAD);
    /* A TP rank receives a local head slice but selects its weights using
     * the global group index. Give each view its exact active byte length. */
    ds4_gpu_tensor *heads = ds4_gpu_tensor_view(f->t[INPUT],
        (uint64_t)group0 * 4096u * 4u, (uint64_t)groups * 4096u * 4u);
    CHECK(heads);
    const int ok = ds4_gpu_begin_commands() &&
        project_output_a(f, f->t[REFERENCE], group0, groups, heads, 0) &&
        project_output_a(f, f->t[OUTPUT], group0, groups, heads, 1) &&
        ds4_gpu_end_commands() && equal_output(f, s, 1, sparse);
    if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
    ds4_gpu_tensor_free(heads);
    return ok;
}

static int rejected(fixture *f) {
    const uint64_t bytes = f->model_bytes;
    ds4_gpu_tensor *x = f->t[INPUT], *out = f->t[OUTPUT];
    memset(ds4_gpu_tensor_contents(f->storage[OUTPUT]), 0xa5, f->bytes[OUTPUT] + 2u * PAD);
#define REJECT(o, map, size, off, k, m, in, n) \
    CHECK(!ds4_gpu_dsv41_q4_bf16_rows(o, map, size, off, k, m, in, n))
    REJECT(NULL, f->model, bytes, WEIGHT_OFFSET, 5120, 512, x, 1);
    REJECT(out, NULL, bytes, WEIGHT_OFFSET, 5120, 512, x, 1);
    REJECT(out, f->model, bytes, WEIGHT_OFFSET, 5120, 512, NULL, 1);
    REJECT(out, f->model, bytes, WEIGHT_OFFSET, 0, 512, x, 1);
    REJECT(out, f->model, bytes, WEIGHT_OFFSET, 5120, 0, x, 1);
    REJECT(out, f->model, bytes, WEIGHT_OFFSET, 5119, 512, x, 1);
    REJECT(out, f->model, bytes, WEIGHT_OFFSET, 5120, 511, x, 1);
    REJECT(out, f->model, bytes, WEIGHT_OFFSET, 5120, 512, x, 0);
    REJECT(out, f->model, bytes, WEIGHT_OFFSET, 5120, 512, x, MAX_ROWS + 1);
    REJECT(out, f->model, bytes, WEIGHT_OFFSET, UINT64_MAX - 255u, 512, x, 1);
    REJECT(out, f->model, bytes, WEIGHT_OFFSET, 5120, UINT64_MAX - 3u, x, 1);
    /* This deliberately inconsistent length must fail bounds validation
     * before lookup of the registered model mapping. */
    REJECT(out, f->model, WEIGHT_OFFSET + weight_bytes(shapes[1]) - 1u,
           WEIGHT_OFFSET, 5120, 512, x, 1);
    REJECT(out, f->model, bytes, bytes + 1u, 5120, 512, x, 1);
    REJECT(out, f->model, bytes, UINT64_MAX - 127u, 5120, 512, x, 1);
    REJECT(out, f->model, f->model_bytes, WEIGHT_OFFSET + 1u, 5120, 512, x, 1);
    REJECT(out, ds4_gpu_tensor_contents(f->storage[OUTPUT]),
           f->bytes[OUTPUT] + 2u * PAD, PAD, 256, 4, x, 1);
    /* Views carry enough backing storage: validate their declared lengths,
     * and reject both exact aliasing and partial overlap before encoding. */
    ds4_gpu_tensor *bad[7] = {
        ds4_gpu_tensor_view(x, 0, 5120u * 4u - 1u),
        ds4_gpu_tensor_view(out, 0, 512u * 4u - 1u),
        ds4_gpu_tensor_view(x, 1, 5120u * 4u),
        ds4_gpu_tensor_view(out, 1, 512u * 4u),
        ds4_gpu_tensor_view(x, 4, 512u * 4u),
        ds4_gpu_tensor_view(x, 0, 5120u * 4u * MAX_ROWS - 1u),
        ds4_gpu_tensor_view(out, 0, 512u * 4u * MAX_ROWS - 1u)
    };
    for (unsigned i = 0; i < sizeof(bad) / sizeof(*bad); i++) CHECK(bad[i]);
    REJECT(out, f->model, bytes, WEIGHT_OFFSET, 5120, 512, bad[0], 1);
    REJECT(bad[1], f->model, bytes, WEIGHT_OFFSET, 5120, 512, x, 1);
    REJECT(out, f->model, bytes, WEIGHT_OFFSET, 5120, 512, bad[2], 1);
    REJECT(bad[3], f->model, bytes, WEIGHT_OFFSET, 5120, 512, x, 1);
    REJECT(x, f->model, bytes, WEIGHT_OFFSET, 5120, 512, x, 1);
    REJECT(bad[4], f->model, bytes, WEIGHT_OFFSET, 5120, 512, x, 1);
    REJECT(out, f->model, bytes, WEIGHT_OFFSET, 5120, 512, bad[5], MAX_ROWS);
    REJECT(bad[6], f->model, bytes, WEIGHT_OFFSET, 5120, 512, x, MAX_ROWS);
    for (unsigned i = 0; i < sizeof(bad) / sizeof(*bad); i++) ds4_gpu_tensor_free(bad[i]);
#undef REJECT
    CHECK(!ds4_gpu_commands_active());
    CHECK(!memcmp(f->input_snapshot, ds4_gpu_tensor_contents(f->storage[INPUT]),
                  f->bytes[INPUT] + 2u * PAD));
    const unsigned char *p = ds4_gpu_tensor_contents(f->storage[OUTPUT]);
    for (uint64_t i = 0; i < f->bytes[OUTPUT] + 2u * PAD; i++) CHECK(p[i] == 0xa5);
    return 1;
}

static double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int order_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int benchmark(fixture *f, shape s, int output_a) {
    enum { SAMPLES = 11, WARMUP = 2, ITERATIONS = 128 };
    double samples[2][SAMPLES];
    CHECK(output_a ? parity_output_a(f, 0, 8, 0) : parity(f, s, 1, 0));
    for (unsigned pass = 0; pass < SAMPLES + WARMUP; pass++) {
        double elapsed[2] = {0, 0};
        for (unsigned turn = 0; turn < 4; turn++) {
            /* Alternating ABBA/BAAB balances drift. Both arms use the same
             * physical output; inputs and weights stay resident and fixed. */
            const unsigned fused = (turn == 1 || turn == 2) ^ (pass & 1u);
            const double start = seconds();
            CHECK(ds4_gpu_begin_commands());
            for (unsigned i = 0; i < ITERATIONS; i++)
                CHECK(output_a ? project_output_a(f, f->t[OUTPUT], 0, 8, f->t[INPUT], fused) :
                                 project(f, f->t[OUTPUT], s, 1, fused));
            CHECK(ds4_gpu_end_commands());
            elapsed[fused] += seconds() - start;
        }
        CHECK(equal_output(f, s, 1, 0));
        if (pass >= WARMUP) for (unsigned mode = 0; mode < 2; mode++)
            samples[mode][pass - WARMUP] = elapsed[mode] * 1e6 / (2u * ITERATIONS);
    }
    for (unsigned mode = 0; mode < 2; mode++)
        qsort(samples[mode], SAMPLES, sizeof(double), order_double);
    printf("V4.1 Q4 BF16 %s %u->%u rows=1 resident ABBA/BAAB median us: "
           "separate=%.3f API=%.3f speedup=%.4fx ranges=[%.3f,%.3f]/[%.3f,%.3f] "
           "samples=%u batch=%u\n", s.name, s.in, s.out,
           samples[0][SAMPLES / 2], samples[1][SAMPLES / 2],
           samples[0][SAMPLES / 2] / samples[1][SAMPLES / 2],
           samples[0][0], samples[0][SAMPLES - 1], samples[1][0], samples[1][SAMPLES - 1],
           SAMPLES, ITERATIONS);
    return 1;
}

static int check_all(fixture *f, int bench) {
    for (unsigned si = 0; si < sizeof(shapes) / sizeof(*shapes); si++) {
        const shape s = shapes[si];
        for (unsigned sparse = 0; sparse < 2; sparse++) {
            fill(f, s, sparse);
            const uint64_t before = hash_bytes(f->model, f->model_bytes);
            for (unsigned disabled = 0; disabled < 2; disabled++) {
                CHECK(disabled ? setenv(rollback, "1", 1) == 0 : unsetenv(rollback) == 0);
                for (uint32_t rows = 1; rows <= MAX_ROWS; rows++)
                    CHECK(parity(f, s, rows, sparse));
                CHECK(rejected(f));
            }
            CHECK(hash_bytes(f->model, f->model_bytes) == before);
        }
        fprintf(stderr, "V4.1 Q4 BF16 %s %u->%u: bitwise rows=1..8, rollback, "
                "BF16 ties, signed zero, views, guards and rejection PASS\n", s.name, s.in, s.out);
        if (bench) {
            /* Honor the caller's rollback setting for the benchmark. The
             * correctness loops above exercise both settings explicitly. */
            CHECK(f->had_rollback ? setenv(rollback, f->saved_rollback, 1) == 0 :
                                   unsetenv(rollback) == 0);
            fill(f, s, 0);
            const uint64_t before = hash_bytes(f->model, f->model_bytes);
            CHECK(benchmark(f, s, 0));
            CHECK(hash_bytes(f->model, f->model_bytes) == before);
        }
    }
    const shape oa = {4096, 8u * 1024u, "O_A (8 groups)"};
    for (unsigned sparse = 0; sparse < 2; sparse++) {
        fill(f, oa, sparse);
        const uint64_t before = hash_bytes(f->model, f->model_bytes);
        for (unsigned disabled = 0; disabled < 2; disabled++) {
            CHECK(disabled ? setenv(rollback, "1", 1) == 0 : unsetenv(rollback) == 0);
            CHECK(parity_output_a(f, 0, 8, sparse));
            uint32_t full[8u * 1024u];
            memcpy(full, ds4_gpu_tensor_contents(f->t[REFERENCE]), sizeof(full));
            for (uint32_t group0 = 0; group0 < 8; group0 += 4) {
                CHECK(parity_output_a(f, group0, 4, sparse));
                CHECK(!memcmp(full + group0 * 1024u,
                    ds4_gpu_tensor_contents(f->t[OUTPUT]), 4u * 1024u * sizeof(uint32_t)));
            }
        }
        CHECK(hash_bytes(f->model, f->model_bytes) == before);
    }
    fprintf(stderr, "V4.1 Q4 BF16 O_A 4096->1024 x8: bitwise full and TP ranks0/1, "
            "rollback, BF16 ties, signed zero, views and guards PASS\n");
    if (bench) {
        CHECK(f->had_rollback ? setenv(rollback, f->saved_rollback, 1) == 0 :
                               unsetenv(rollback) == 0);
        fill(f, oa, 0);
        const uint64_t before = hash_bytes(f->model, f->model_bytes);
        CHECK(benchmark(f, oa, 1));
        CHECK(hash_bytes(f->model, f->model_bytes) == before);
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--bench"))) {
        fprintf(stderr, "usage: %s [--bench]\n", argv[0]);
        return 1;
    }
    if (!ds4_gpu_init()) return 1;
    fixture f = {0};
    const int ok = initialize(&f) && check_all(&f, argc == 2);
    destroy(&f);
    return ok ? 0 : 1;
}
