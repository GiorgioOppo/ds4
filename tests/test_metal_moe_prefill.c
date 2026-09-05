#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { INPUT = 256, MID = 512, OUTPUT = 260, EXPERTS = 256, SELECTED = 6 };
typedef struct { uint16_t d; uint8_t qs[64]; } iq2_block;
typedef struct { uint8_t scales[16], qs[64]; uint16_t d, dmin; } q2_block;

static uint32_t rng = 1;
static uint32_t random_u32(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static uint64_t aligned(uint64_t n, uint64_t page) {
    return (n + page - 1) / page * page;
}

static int check_case(const void *model, uint64_t model_size,
                      uint64_t up_off, uint64_t down_off, uint32_t tokens) {
    const uint64_t pairs = (uint64_t)tokens * SELECTED;
    const uint64_t counts[] = {pairs * MID, pairs * MID, pairs * MID,
                              pairs * OUTPUT, (uint64_t)tokens * OUTPUT};
    ds4_gpu_tensor *result[5] = {0};
    void *reference[5] = {0};
    float *actual = malloc(counts[0] * sizeof(float));
    float *x = malloc((uint64_t)tokens * INPUT * sizeof(float));
    int32_t *ids = malloc(pairs * sizeof(int32_t));
    float *weights = malloc(pairs * sizeof(float));
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)tokens * INPUT * sizeof(float));
    ds4_gpu_tensor *it = ds4_gpu_tensor_alloc(pairs * sizeof(int32_t));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(pairs * sizeof(float));
    int ok = actual && x && ids && weights && xt && it && wt;
    for (int i = 0; i < 5 && ok; i++) {
        result[i] = ds4_gpu_tensor_alloc(counts[i] * sizeof(float));
        reference[i] = malloc(counts[i] * sizeof(float));
        ok = result[i] && reference[i];
    }
    if (!ok) goto cleanup;
    for (uint64_t i = 0; i < (uint64_t)tokens * INPUT; i++)
        x[i] = ((int)(random_u32() % 101) - 50) / 256.0f;
    /* One hot expert, uneven small groups and many unused experts. */
    for (uint32_t t = 0; t < tokens; t++) {
        ids[(uint64_t)t * SELECTED] = 0;
        weights[(uint64_t)t * SELECTED] = 1.0f / SELECTED;
        for (uint32_t s = 1; s < SELECTED; s++) {
            ids[(uint64_t)t * SELECTED + s] = 1 + (t * 7 + s * 19) % 111;
            weights[(uint64_t)t * SELECTED + s] = 1.0f / SELECTED;
        }
    }
    ok = ds4_gpu_tensor_write(xt, 0, x, (uint64_t)tokens * INPUT * sizeof(float)) &&
         ds4_gpu_tensor_write(it, 0, ids, pairs * sizeof(int32_t)) &&
         ds4_gpu_tensor_write(wt, 0, weights, pairs * sizeof(float));
    for (int run = 0; run < 3 && ok; run++) {
        if (run == 0) setenv("DS4_METAL_DISABLE_ROUTED_MPP_PACKED", "1", 1);
        else unsetenv("DS4_METAL_DISABLE_ROUTED_MPP_PACKED");
        for (int i = 0; i < 5 && ok; i++)
            ok = ds4_gpu_tensor_fill_f32(result[i], NAN, counts[i]);
        bool half_mid = false;
        ok = ok && ds4_gpu_routed_moe_batch_tensor(
            result[4], result[0], result[1], result[2], result[3],
            model, model_size, 0, up_off, down_off, 16, 10,
            MID * sizeof(iq2_block), sizeof(iq2_block),
            OUTPUT * 2 * sizeof(q2_block), 2 * sizeof(q2_block),
            INPUT, MID, OUTPUT, it, wt, EXPERTS, SELECTED, 7.0f,
            xt, 0, tokens, &half_mid, true);
        ok = ok && half_mid;
        for (int i = 0; i < 5 && ok; i++) {
            const uint64_t bytes = counts[i] * (i == 2 ? sizeof(uint16_t) : sizeof(float));
            ok = ds4_gpu_tensor_read(result[i], 0, actual, bytes);
            for (uint64_t j = 0; j < counts[i] && ok; j++) {
                const float v = i == 2 ? (float)((_Float16 *)actual)[j] : actual[j];
                if (!isfinite(v)) ok = 0;
            }
            if (run == 0) memcpy(reference[i], actual, bytes);
            else if (memcmp(reference[i], actual, bytes)) {
                fprintf(stderr, "MoE prefill mismatch tokens=%u run=%d tensor=%d\n", tokens, run, i);
                ok = 0;
            }
        }
    }
cleanup:
    unsetenv("DS4_METAL_DISABLE_ROUTED_MPP_PACKED");
    for (int i = 0; i < 5; i++) {
        ds4_gpu_tensor_free(result[i]);
        free(reference[i]);
    }
    ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(it);
    ds4_gpu_tensor_free(wt);
    free(actual);
    free(x);
    free(ids);
    free(weights);
    fprintf(stderr, "MoE prefill tokens=%u: %s\n", tokens, ok ? "PASS" : "FAIL");
    return ok;
}

int main(void) {
    const uint64_t page = getpagesize();
    const uint64_t up_off = aligned((uint64_t)EXPERTS * MID * sizeof(iq2_block), page);
    const uint64_t down_off = up_off * 2;
    const uint64_t model_size = aligned(down_off + (uint64_t)EXPERTS * OUTPUT * 2 * sizeof(q2_block), page);
    void *model = NULL;
    if (posix_memalign(&model, page, model_size)) return 1;
    memset(model, 0, model_size);
    for (uint64_t off = 0; off < down_off; off += up_off) {
        iq2_block *w = (iq2_block *)((char *)model + off);
        for (uint32_t b = 0; b < EXPERTS * MID; b++) {
            w[b].d = 0x1400; /* 2^-10, finite nontrivial activations. */
            for (uint32_t j = 0; j < sizeof(w[b].qs); j++) w[b].qs[j] = random_u32();
        }
    }
    q2_block *w = (q2_block *)((char *)model + down_off);
    for (uint32_t b = 0; b < EXPERTS * OUTPUT * 2; b++) {
        w[b].d = w[b].dmin = 0x2000;
        for (uint32_t j = 0; j < sizeof(w[b].scales); j++) w[b].scales[j] = random_u32();
        for (uint32_t j = 0; j < sizeof(w[b].qs); j++) w[b].qs[j] = random_u32();
    }
    int ok = ds4_gpu_init() && ds4_gpu_set_model_map(model, model_size);
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    const uint32_t sizes[] = {511, 512, 513, 1024, 2048, 4096, 512};
    for (unsigned i = 0; i < sizeof(sizes) / sizeof(*sizes) && ok; i++)
        ok = check_case(model, model_size, up_off, down_off, sizes[i]);
    ds4_gpu_cleanup();
    free(model);
    return ok ? 0 : 1;
}
