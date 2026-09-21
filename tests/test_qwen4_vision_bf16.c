/* BF16 matrices in the Qwen vision encoder used by Bonsai.
 * The existing F32 path, given exactly widened BF16 weights, is a bitwise
 * oracle for both the tiled matmul and the complete 27-layer encoder.
 * No model download is needed. Build: make tests/test_qwen4_vision_bf16 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static void require_ok(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "%s failed\n", what); exit(1); }
}

static void *checked_alloc(size_t bytes) {
    void *p = malloc(bytes);
    require_ok(p != NULL, "host allocation");
    return p;
}

static uint32_t rng = 0x6731abcd;
static float random_signed(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (float)(rng & 0xffffu) / 32768.0f - 1.0f;
}

typedef struct { uint8_t *base; uint64_t size, used; } arena;
typedef struct { uint64_t bf16, f32; } matrix_pair;

static uint64_t arena_alloc(arena *a, uint64_t bytes) {
    const uint64_t off = (a->used + 63u) & ~63ull;
    require_ok(off <= a->size && bytes <= a->size - off, "model arena capacity");
    a->used = off + bytes;
    return off;
}

static matrix_pair make_matrix(arena *a, uint32_t rows, uint32_t cols, float scale) {
    const uint64_t n = (uint64_t)rows * cols;
    /* Exercise nonzero, scalar-aligned model offsets too. */
    matrix_pair p = { arena_alloc(a, n * 2u + 2u) + 2u,
                      arena_alloc(a, n * 4u + 4u) + 4u };
    uint16_t *b = (uint16_t *)(a->base + p.bf16);
    float *f = (float *)(a->base + p.f32);
    for (uint64_t i = 0; i < n; i++) {
        float v = random_signed() * scale;
        uint32_t bits;
        memcpy(&bits, &v, sizeof(bits));
        b[i] = (uint16_t)(bits >> 16);
        bits = (uint32_t)b[i] << 16;
        memcpy(f + i, &bits, sizeof(bits));
    }
    return p;
}

static uint64_t make_vector(arena *a, uint32_t n, float base, float scale) {
    uint64_t off = arena_alloc(a, (uint64_t)n * sizeof(float));
    float *v = (float *)(a->base + off);
    for (uint32_t i = 0; i < n; i++) v[i] = base + scale * random_signed();
    return off;
}

static void check_exact(const char *what, const float *got, const float *ref, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(got[i]) || !isfinite(ref[i]) || memcmp(got + i, ref + i, sizeof(float))) {
            fprintf(stderr, "%s: element %llu: BF16 %.9g, F32 %.9g\n",
                    what, (unsigned long long)i, got[i], ref[i]);
            exit(1);
        }
    }
}

enum { GUARD = 16 };
static const float canary = 9137.25f;

static void check_guards(const char *what, const float *v, uint64_t n) {
    for (unsigned i = 0; i < GUARD; i++)
        require_ok(v[i] == canary && v[GUARD + n + i] == canary, what);
}

static void test_matmul(arena *a, uint32_t K, uint32_t R, uint32_t T, int exponent) {
    matrix_pair w = make_matrix(a, R, K, ldexpf(0.125f, exponent));
    const uint64_t nx = (uint64_t)T * K, ny = (uint64_t)T * R;
    float *x = checked_alloc((nx + 2u * GUARD) * sizeof(float));
    float *init = checked_alloc((ny + 2u * GUARD) * sizeof(float));
    float *got = checked_alloc((ny + 2u * GUARD) * sizeof(float));
    float *ref = checked_alloc((ny + 2u * GUARD) * sizeof(float));
    double *cpu = checked_alloc(ny * sizeof(double));
    for (uint64_t i = 0; i < nx + 2u * GUARD; i++) x[i] = canary;
    for (uint64_t i = 0; i < nx; i++) x[GUARD + i] = ldexpf(random_signed(), -exponent);
    for (uint64_t i = 0; i < ny + 2u * GUARD; i++) init[i] = canary;
    for (uint64_t i = 0; i < ny; i++) init[GUARD + i] = NAN;
    const float *wf = (const float *)(a->base + w.f32);
    double peak = 0.0;
    for (uint32_t t = 0; t < T; t++) for (uint32_t r = 0; r < R; r++) {
        double sum = 0.0;
        for (uint32_t k = 0; k < K; k++)
            sum += (double)wf[(uint64_t)r * K + k] * x[GUARD + (uint64_t)t * K + k];
        cpu[(uint64_t)t * R + r] = sum;
        peak = fmax(peak, fabs(sum));
    }
    ds4_gpu_tensor *xb = ds4_gpu_tensor_alloc((nx + 2u * GUARD) * sizeof(float));
    ds4_gpu_tensor *yb = ds4_gpu_tensor_alloc((ny + 2u * GUARD) * sizeof(float));
    ds4_gpu_tensor *rb = ds4_gpu_tensor_alloc((ny + 2u * GUARD) * sizeof(float));
    require_ok(xb && yb && rb, "matmul buffer allocation");
    ds4_gpu_tensor *xt = ds4_gpu_tensor_view(xb, GUARD * sizeof(float), nx * sizeof(float));
    ds4_gpu_tensor *yt = ds4_gpu_tensor_view(yb, GUARD * sizeof(float), ny * sizeof(float));
    ds4_gpu_tensor *rt = ds4_gpu_tensor_view(rb, GUARD * sizeof(float), ny * sizeof(float));
    require_ok(xt && yt && rt, "matmul tensor views");
    require_ok(ds4_gpu_tensor_write(xb, 0, x, (nx + 2u * GUARD) * sizeof(float)), "matmul input upload");
    for (unsigned split = 0; split < 2; split++) {
        require_ok(ds4_gpu_tensor_write(yb, 0, init, (ny + 2u * GUARD) * sizeof(float)) &&
                   ds4_gpu_tensor_write(rb, 0, init, (ny + 2u * GUARD) * sizeof(float)), "matmul output canaries");
        require_ok(ds4_gpu_begin_commands(), "matmul begin commands");
        int ok;
        if (split) {
            ok = ds4_gpu_qwen4_dense_mm_tensor(yt, xt, a->base, a->size, w.bf16, 30, T, K, R) &&
                 ds4_gpu_qwen4_dense_mm_tensor(rt, xt, a->base, a->size, w.f32, 0, T, K, R);
        } else {
            ok = ds4_gpu_qwen4_dense_mm_prefill_tensor(yt, xt, a->base, a->size, w.bf16, 30, T, K, R) &&
                 ds4_gpu_qwen4_dense_mm_prefill_tensor(rt, xt, a->base, a->size, w.f32, 0, T, K, R);
        }
        require_ok(ds4_gpu_end_commands() && ok, "BF16/F32 matmul dispatch");
        require_ok(ds4_gpu_tensor_read(yb, 0, got, (ny + 2u * GUARD) * sizeof(float)) &&
                   ds4_gpu_tensor_read(rb, 0, ref, (ny + 2u * GUARD) * sizeof(float)), "matmul output read");
        check_guards("BF16 output guards", got, ny);
        check_guards("F32 output guards", ref, ny);
        check_exact("matmul", got + GUARD, ref + GUARD, ny);
        double worst = 0.0;
        for (uint64_t i = 0; i < ny; i++) worst = fmax(worst, fabs((double)got[GUARD + i] - cpu[i]));
        require_ok(worst <= 3e-5 * peak + 1e-6, "matmul double-precision oracle");
    }
    require_ok(ds4_gpu_tensor_read(xb, 0, x, (nx + 2u * GUARD) * sizeof(float)), "matmul input guards read");
    check_guards("matmul input guards", x, nx);
    ds4_gpu_tensor_free(rt); ds4_gpu_tensor_free(yt); ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(rb); ds4_gpu_tensor_free(yb); ds4_gpu_tensor_free(xb);
    free(cpu); free(ref); free(got); free(init); free(x);
    printf("BF16 dense K=%u rows=%u tokens=%u exponent=%d: exact F32, split/unsplit\n", K, R, T, exponent);
}

static void test_vision(arena *a) {
    /* Reduce the head count, retaining the projector's D=72, FF=4304,
     * patch=16, merge=2, output=5120 and all 27 transformer layers. Reusing
     * each synthetic block's weights keeps the fixture below 64 MiB. */
    const uint32_t E = 288, FF = 4304, ME = E * 4u, O = 5120;
    ds4_qwen4_vision_weights b = {0}, f = {0};
    b.n_embd = E; b.n_ff = FF; b.n_head = 4; b.n_patch = 16;
    b.n_merge = 2; b.n_pos_side = 48; b.n_out = O; b.eps = 1e-6f;
    b.patch_type = 0; b.mm0_type = b.mm2_type = 30;
    matrix_pair patch0 = make_matrix(a, E, 768, 0.015625f);
    matrix_pair patch1 = make_matrix(a, E, 768, 0.015625f);
    b.patch_w0 = patch0.f32; b.patch_w1 = patch1.f32;
    b.patch_b = make_vector(a, E, 0.0f, 0.01f);
    b.post_ln_w = make_vector(a, E, 1.0f, 0.05f);
    b.post_ln_b = make_vector(a, E, 0.0f, 0.01f);
    b.mm0_b = make_vector(a, ME, 0.0f, 0.01f);
    b.mm2_b = make_vector(a, O, 0.0f, 0.01f);
    ds4_qwen4_vision_layer_weights l = {0};
    l.qkv_type = l.out_type = l.up_type = l.down_type = 30;
    l.ln1_w = make_vector(a, E, 1.0f, 0.05f);
    l.ln1_b = make_vector(a, E, 0.0f, 0.01f);
    l.ln2_w = make_vector(a, E, 1.0f, 0.05f);
    l.ln2_b = make_vector(a, E, 0.0f, 0.01f);
    l.qkv_b = make_vector(a, 3u * E, 0.0f, 0.01f);
    l.out_b = make_vector(a, E, 0.0f, 0.01f);
    l.up_b = make_vector(a, FF, 0.0f, 0.01f);
    l.down_b = make_vector(a, E, 0.0f, 0.01f);
    matrix_pair qkv = make_matrix(a, 3u * E, E, 0.125f / sqrtf((float)E));
    matrix_pair out = make_matrix(a, E, E, 0.125f / sqrtf((float)E));
    matrix_pair up = make_matrix(a, FF, E, 0.125f / sqrtf((float)E));
    matrix_pair down = make_matrix(a, E, FF, 0.125f / sqrtf((float)FF));
    matrix_pair mm0 = make_matrix(a, ME, ME, 0.25f / sqrtf((float)ME));
    matrix_pair mm2 = make_matrix(a, O, ME, 0.25f / sqrtf((float)ME));
    l.qkv_w = qkv.bf16; l.out_w = out.bf16; l.up_w = up.bf16; l.down_w = down.bf16;
    for (unsigned i = 0; i < DS4_QWEN4_VISION_LAYERS; i++) b.layer[i] = l;
    b.mm0_w = mm0.bf16; b.mm2_w = mm2.bf16;
    f = b; f.mm0_w = mm0.f32; f.mm2_w = mm2.f32; f.mm0_type = f.mm2_type = 0;
    for (unsigned i = 0; i < DS4_QWEN4_VISION_LAYERS; i++) {
        f.layer[i].qkv_w = qkv.f32; f.layer[i].out_w = out.f32;
        f.layer[i].up_w = up.f32; f.layer[i].down_w = down.f32;
        f.layer[i].qkv_type = f.layer[i].out_type = f.layer[i].up_type = f.layer[i].down_type = 0;
    }
    const uint32_t counts[] = {4, 12};
    for (unsigned c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
        const uint32_t N = counts[c], grid_w = N / 2u;
        const uint64_t nout = (uint64_t)(N / 4u) * O;
        float *patches = checked_alloc((uint64_t)N * 768 * sizeof(float));
        float *pos = checked_alloc((uint64_t)N * E * sizeof(float));
        float *got = checked_alloc((nout + 2u * GUARD) * sizeof(float));
        float *ref = checked_alloc((nout + 2u * GUARD) * sizeof(float));
        for (uint64_t i = 0; i < (uint64_t)N * 768; i++) patches[i] = random_signed();
        for (uint64_t i = 0; i < (uint64_t)N * E; i++) pos[i] = random_signed() * 0.01f;
        for (uint64_t i = 0; i < nout + 2u * GUARD; i++) got[i] = ref[i] = canary;
        for (uint64_t i = 0; i < nout; i++) got[GUARD + i] = ref[GUARD + i] = NAN;
        require_ok(ds4_gpu_qwen4_vision_encode(got + GUARD, patches, pos, N, grid_w, a->base, a->size, &b),
                   "BF16 vision encoder");
        require_ok(!ds4_gpu_commands_active(), "BF16 vision commands completed");
        require_ok(ds4_gpu_qwen4_vision_encode(ref + GUARD, patches, pos, N, grid_w, a->base, a->size, &f),
                   "F32 vision encoder oracle");
        require_ok(!ds4_gpu_commands_active(), "F32 vision commands completed");
        check_guards("BF16 vision output guards", got, nout);
        check_guards("F32 vision output guards", ref, nout);
        check_exact("vision encoder", got + GUARD, ref + GUARD, nout);
        printf("BF16 vision 27 layers, D72 FF4304 O5120, patches=%u: exact F32\n", N);
        free(ref); free(got); free(pos); free(patches);
    }
}

int main(void) {
    arena a = {0};
    a.size = 512ull << 20;
    a.base = mmap(NULL, a.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    require_ok(a.base != MAP_FAILED, "model mmap");
    setenv("DS4_QWEN4_NO_DENSE_MM_KSPLIT", "0", 1);
    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(a.base, a.size), "model registration");
    test_matmul(&a, 8, 1, 1, 0);
    test_matmul(&a, 24, 33, 3, 0);
    test_matmul(&a, 40, 35, 33, 0);
    test_matmul(&a, 136, 17, 31, 0);
    test_matmul(&a, 40, 3, 2, 80);
    test_matmul(&a, 40, 3, 2, -80);
    test_matmul(&a, 4304, 67, 33, 0);
    test_matmul(&a, 1152, 4304, 1, 0);
    test_matmul(&a, 1152, 3456, 3, 0);
    test_matmul(&a, 4608, 5120, 1, 0);
    test_vision(&a);
    ds4_gpu_cleanup();
    require_ok(munmap(a.base, a.size) == 0, "model munmap");
    puts("Qwen vision BF16 tests passed");
    return 0;
}
