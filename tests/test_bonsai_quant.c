/* Standalone CPU codec/transform test:
 * cc -std=c11 -O2 -Wall -Wextra -Werror -I. tests/test_bonsai_quant.c -lm -o /tmp/test_bonsai_quant
 * /tmp/test_bonsai_quant
 *
 * DS4_BONSAI_REFERENCE additionally accepts a separately linked adapter to
 * the unmodified reference dequantizers/quantizer (declarations below).
 */
#include "bonsai_quant.h"
#include "bonsai_quant.h" /* Include guard. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef DS4_BONSAI_REFERENCE
void ds4_bonsai_reference_dequantize(uint32_t type, const void *src, float *dst, size_t n);
void ds4_bonsai_reference_quantize_ptq1(const float *src, void *dst, size_t n);
#endif

static void need(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "Bonsai quant: %s failed\n", what); exit(1); }
}

static void same(const float *a, const float *b, size_t n, const char *what) {
    for (size_t i = 0; i < n; i++) {
        if (memcmp(a + i, b + i, sizeof(float)) != 0) {
            fprintf(stderr, "Bonsai quant: %s mismatch at %zu: %.9g != %.9g\n", what, i, a[i], b[i]);
            exit(1);
        }
    }
}

static uint32_t random_state = 0x392901u;
static uint32_t random_u32(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}

static void test_half(void) {
    for (uint32_t h = 0; h <= UINT16_MAX; h++) {
        const float actual = ds4_bonsai_fp16_to_f32((uint16_t)h);
        const unsigned exponent = (h >> 10) & 31u, fraction = h & 1023u;
        if (exponent == 31u) {
            need(fraction ? isnan(actual) : isinf(actual), "half infinity/NaN class");
        } else {
            const float magnitude = exponent ? ldexpf(1.f + fraction / 1024.f, (int)exponent - 15)
                                             : ldexpf((float)fraction, -24);
            const float expected = h & 0x8000u ? -magnitude : magnitude;
            same(&actual, &expected, 1, "all finite half encodings");
        }
    }
}

static void test_pq2(void) {
    /* LE fp16(1.5), then each of the four encodings in low-to-high order.
     * The final byte is deliberately different to check the group boundary. */
    uint8_t raw[1 + 2 * DS4_BONSAI_PQ2_0_BYTES] = {0};
    uint8_t *block = raw + 1; /* Unaligned input is legal. */
    block[0] = 0x00; block[1] = 0x3e;
    memset(block + 2, 0xe4, 32);
    block[33] = 0x1b;
    block[34] = 0x00; block[35] = 0x38; /* Next group has scale 0.5. */
    memset(block + 36, 0xaa, 32);
    float output[258], expected[256];
    output[0] = output[257] = 123.25f;
    const float first[4] = {-1.5f, 0.f, 1.5f, 3.f};
    const float last[4] = {3.f, 1.5f, 0.f, -1.5f};
    for (unsigned i = 0; i < 256; i++)
        expected[i] = i >= 128 ? 0.5f : i >= 124 ? last[i - 124] : first[i % 4];
    need(ds4_bonsai_dequantize_row(142, block, output + 1, 256), "PQ2 known block decode");
    same(output + 1, expected, 256, "PQ2 serialized known vector");
    need(output[0] == 123.25f && output[257] == 123.25f, "PQ2 output guards");

    /* Every possible packed byte, with an exactly representable negative
     * scale as well; zero scale must retain normal IEEE multiply semantics. */
    const uint16_t scales[] = {0, 0x8000, 1, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff};
    for (unsigned s = 0; s < sizeof(scales) / sizeof(scales[0]); s++) {
        block[0] = (uint8_t)scales[s]; block[1] = (uint8_t)(scales[s] >> 8);
        const float scale = ds4_bonsai_fp16_to_f32(scales[s]);
        for (unsigned byte = 0; byte < 256; byte++) {
            memset(block + 2, (int)byte, 32);
            need(ds4_bonsai_dequantize_pq2_0(block, output, 128), "PQ2 exhaustive byte decode");
            unsigned digits = byte;
            for (unsigned j = 0; j < 4; j++, digits /= 4)
                expected[j] = (float)((int)(digits % 4) - 1) * scale;
            for (unsigned j = 0; j < 128; j++) same(output + j, expected + j % 4, 1, "PQ2 byte/code order");
        }
    }
}

static void test_ptq1(void) {
    /* Produced by quantize_row_ptq1_0_ref in the Bonsai reference from the
     * deterministic ternary sequence below. This includes all stage edges,
     * the two qh lanes and the fp16 scale at offset 26, not offset zero. */
    static const uint8_t known[28] = {
        0x96, 0xc1, 0x43, 0xc8, 0x57, 0x06, 0x10, 0x32,
        0x96, 0xc2, 0x47, 0xd5, 0x81, 0x82, 0x87, 0x96,
        0x20, 0x21, 0x2f, 0xa2, 0xb4, 0x51, 0xd2, 0x60,
        0x5f, 0xe4, 0x00, 0x3e
    };
    uint8_t raw[1 + 2 * DS4_BONSAI_PTQ1_0_BYTES];
    uint8_t *block = raw + 1;
    memcpy(block, known, sizeof(known));
    memcpy(block + 28, known, sizeof(known));
    block[54] = 0; block[55] = 0x38;
    float output[258], expected[256];
    for (unsigned j = 0; j < 128; j++) {
        const int trit = (int)(((j * j + 3 * j + 7) / 5) % 3) - 1;
        expected[j] = 1.5f * trit; expected[128 + j] = 0.5f * trit;
    }
#ifdef DS4_BONSAI_REFERENCE
    uint8_t packed[28];
    ds4_bonsai_reference_quantize_ptq1(expected, packed, 128);
    need(memcmp(known, packed, sizeof(known)) == 0, "PTQ1 known bytes match reference quantizer");
#endif
    output[0] = output[257] = 123.25f;
    need(ds4_bonsai_dequantize_row(143, block, output + 1, 256), "PTQ1 known block decode");
    same(output + 1, expected, 256, "PTQ1 serialized known vector");
    need(output[0] == 123.25f && output[257] == 123.25f, "PTQ1 output guards");

    /* All 243 canonical five-trit words, checked against base-3 digits
     * before byte rescaling. qh intentionally ignores the fifth trit. */
    const unsigned divisor[5] = {81, 27, 9, 3, 1};
    for (unsigned code = 0; code < 243; code++) {
        const unsigned packed = (code * 256 + 242) / 243;
        memset(block, (int)packed, 26); block[26] = 0; block[27] = 0x3c;
        need(ds4_bonsai_dequantize_ptq1_0(block, output, 128), "PTQ1 exhaustive ternary decode");
        unsigned next = 0;
        const unsigned lanes[3] = {16, 8, 2};
        for (unsigned stage = 0; stage < 3; stage++) {
            for (unsigned trit = 0; trit < (stage == 2 ? 4u : 5u); trit++) {
                const float value = (float)((int)(code / divisor[trit] % 3) - 1);
                for (unsigned lane = 0; lane < lanes[stage]; lane++, next++)
                    same(output + next, &value, 1, "PTQ1 base-3 digit and lane order");
            }
        }
        need(next == 128, "PTQ1 exhaustive element count");
    }
}

static int parity(unsigned n) {
    n ^= n >> 16; n ^= n >> 8; n ^= n >> 4; n ^= n >> 2; n ^= n >> 1;
    return (int)(n & 1);
}

/* Direct normalized Sylvester matrix, independent of butterfly order. */
static void hadamard_oracle(float *dst, const float *src, const int32_t *signs,
                            size_t width, int inverse) {
    for (size_t b = 0; b < width; b += 1024) for (unsigned r = 0; r < 1024; r++) {
        double sum = 0;
        for (unsigned c = 0; c < 1024; c++) {
            const int sign = (parity(r & c) ? -1 : 1) * (!inverse && signs ? signs[b + c] : 1);
            sum += (double)src[b + c] * sign;
        }
        dst[b + r] = (float)(sum / 32.0) * (inverse && signs ? signs[b + r] : 1);
    }
}

static void test_hadamard(size_t width) {
    float *src = malloc(width * sizeof(float));
    float *dst = malloc((width + 2) * sizeof(float));
    float *oracle = malloc(width * sizeof(float));
    float *restored = malloc(width * sizeof(float));
    int32_t *signs = malloc(width * sizeof(int32_t));
    need(src && dst && oracle && restored && signs, "Hadamard test allocation");
    for (size_t i = 0; i < width; i++) {
        src[i] = ((int)(random_u32() % 63) - 31) / 64.f;
        signs[i] = random_u32() & 1u ? 1 : -1;
    }
    for (unsigned inverse = 0; inverse < 2; inverse++) {
        for (unsigned explicit_signs = 0; explicit_signs < 2; explicit_signs++) {
            const int32_t *s = explicit_signs ? signs : NULL;
            hadamard_oracle(oracle, src, s, width, (int)inverse);
            dst[0] = dst[width + 1] = 123.25f;
            need(ds4_bonsai_hadamard_transform(dst + 1, src, s, width, (int)inverse), "Hadamard transform status");
            for (size_t i = 0; i < width; i++)
                need(dst[i + 1] == oracle[i], "Hadamard matches explicit Sylvester matrix");
            need(dst[0] == 123.25f && dst[width + 1] == 123.25f, "Hadamard output guards");
            memcpy(dst, src, width * sizeof(float));
            need(ds4_bonsai_hadamard_transform(dst + 1, dst, s, width, (int)inverse), "Hadamard overlapping buffers");
            for (size_t i = 0; i < width; i++) need(dst[i + 1] == oracle[i], "Hadamard overlap preserves source");
        }
    }
    need(ds4_bonsai_hadamard_forward(restored, src, signs, width) &&
         ds4_bonsai_hadamard_inverse(restored, restored, signs, width), "Hadamard in-place inverse");
    for (size_t i = 0; i < width; i++) need(restored[i] == src[i], "Hadamard signed round trip");
    free(src); free(dst); free(oracle); free(restored); free(signs);
}

static void test_rejection(void) {
    uint8_t packed[34] = {0};
    float src[1024] = {0}, dst[1024], frozen[1024];
    int32_t signs[1024];
    for (unsigned i = 0; i < 1024; i++) { dst[i] = frozen[i] = 123.25f; signs[i] = 1; }
    need(!ds4_bonsai_dequantize_row(42, packed, dst, 128) &&
         !ds4_bonsai_dequantize_row(34, packed, dst, 128), "reject upstream type IDs");
    for (uint32_t type = 142; type <= 143; type++) {
        need(!ds4_bonsai_dequantize_row(type, packed, dst, 127) &&
             !ds4_bonsai_dequantize_row(type, NULL, dst, 128) &&
             !ds4_bonsai_dequantize_row(type, packed, NULL, 128), "reject invalid quant row");
        need(ds4_bonsai_dequantize_row(type, NULL, NULL, 0), "empty quant row");
    }
    need(!ds4_bonsai_hadamard_forward(dst, src, signs, 1023) &&
         !ds4_bonsai_hadamard_forward(dst, src, signs, 0) &&
         !ds4_bonsai_hadamard_forward(dst, NULL, signs, 1024) &&
         !ds4_bonsai_hadamard_transform(dst, src, signs, 1024, 2), "reject invalid Hadamard arguments");
    signs[1023] = 0;
    need(!ds4_bonsai_hadamard_inverse(dst, src, signs, 1024), "reject invalid sign before writing");
    same(dst, frozen, 1024, "rejected operations leave destination unchanged");
}

#ifdef DS4_BONSAI_REFERENCE
static void test_reference(void) {
    enum { BLOCKS = 32, N = BLOCKS * 128 };
    uint8_t storage[1 + BLOCKS * 34], *bytes = storage + 1;
    float expected[N], actual[N];
    for (uint32_t type = 142; type <= 143; type++) {
        const unsigned stride = type == 142 ? 34 : 28, scale_offset = type == 142 ? 0 : 26;
        for (unsigned trial = 0; trial < 64; trial++) {
            for (unsigned j = 0; j < BLOCKS * stride; j++) bytes[j] = (uint8_t)random_u32();
            for (unsigned b = 0; b < BLOCKS; b++) {
                uint16_t half;
                do { half = (uint16_t)random_u32(); } while ((half & 0x7c00u) == 0x7c00u);
                bytes[b * stride + scale_offset] = (uint8_t)half;
                bytes[b * stride + scale_offset + 1] = (uint8_t)(half >> 8);
            }
            /* The reference structs require alignment, unlike our API. */
            uint16_t aligned[BLOCKS * 34 / 2];
            memcpy(aligned, bytes, BLOCKS * stride);
            ds4_bonsai_reference_dequantize(type, aligned, expected, N);
            need(ds4_bonsai_dequantize_row(type, bytes, actual, N), "reference comparison status");
            same(actual, expected, N, "reference dequantizer bitwise parity");
        }
    }
    puts("PASS Bonsai: 524288 values bitwise identical to reference dequantizers");
}
#endif

int main(void) {
    test_half(); test_pq2(); test_ptq1(); test_rejection();
    const size_t widths[] = {1024, 5120, 6144, 17408};
    for (unsigned i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) test_hadamard(widths[i]);
#ifdef DS4_BONSAI_REFERENCE
    test_reference();
#endif
    puts("PASS Bonsai: PQ2_0/PTQ1_0 packing, fp16, signed forward/inverse Hadamard and argument checks");
    return 0;
}
