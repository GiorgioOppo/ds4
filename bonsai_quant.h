#ifndef DS4_BONSAI_QUANT_H
#define DS4_BONSAI_QUANT_H

/* Bonsai/Prism GGUF codecs. These are the private group-128 types, not
 * upstream Q2_0 (group 64) or TQ1_0 (group 256).
 * Layout and element order follow ggml-common.h and ggml-quants.c in
 * PrismML-Eng/llama.cpp at 9a9394a895b96003ca842a6041cb28ac49a108f7.
 * No compiler-specific half type or aligned block loads are required;
 * serialized fp16 scales are little-endian.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    DS4_BONSAI_PQ2_0 = 142,
    DS4_BONSAI_PTQ1_0 = 143,
    DS4_BONSAI_GROUP_SIZE = 128,
    DS4_BONSAI_PQ2_0_BYTES = 34,
    DS4_BONSAI_PTQ1_0_BYTES = 28,
    DS4_BONSAI_HADAMARD_SIZE = 1024
};

static inline float ds4_bonsai_fp16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 31u;
    uint32_t fraction = h & 1023u;
    if (exponent == 0) {
        if (fraction != 0) {
            int e = -14;
            while ((fraction & 1024u) == 0) { fraction <<= 1; e--; }
            bits |= (uint32_t)(e + 127) << 23;
            bits |= (fraction & 1023u) << 13;
        }
    } else if (exponent == 31u) {
        bits |= 0x7f800000u | (fraction << 13);
    } else {
        bits |= ((exponent + 112u) << 23) | (fraction << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline float ds4_bonsai_load_f16_le(const uint8_t *p) {
    return ds4_bonsai_fp16_to_f32((uint16_t)((uint16_t)p[0] | (uint16_t)p[1] << 8));
}

/* Source and destination must not overlap. n is a whole number of groups;
 * n == 0 is a successful no-op, including with NULL pointers. Invalid shape
 * or pointers return 0 before writing destination. */
static inline int ds4_bonsai_dequantize_pq2_0(const void *src, float *dst, size_t n) {
    if (n % DS4_BONSAI_GROUP_SIZE || n > SIZE_MAX / sizeof(float) || (n && (!src || !dst))) return 0;
    const uint8_t *block = (const uint8_t *)src;
    for (size_t b = 0; b < n / DS4_BONSAI_GROUP_SIZE; b++, block += DS4_BONSAI_PQ2_0_BYTES) {
        const float scale = ds4_bonsai_load_f16_le(block);
        for (unsigned j = 0; j < DS4_BONSAI_GROUP_SIZE; j++) {
            const int code = (block[2 + j / 4] >> (2 * (j % 4))) & 3;
            dst[b * DS4_BONSAI_GROUP_SIZE + j] = (float)(code - 1) * scale;
        }
    }
    return 1;
}

static inline int ds4_bonsai_dequantize_ptq1_0(const void *src, float *dst, size_t n) {
    if (n % DS4_BONSAI_GROUP_SIZE || n > SIZE_MAX / sizeof(float) || (n && (!src || !dst))) return 0;
    const uint8_t *block = (const uint8_t *)src;
    const uint8_t power3[5] = {1, 3, 9, 27, 81};
    for (size_t b = 0; b < n / DS4_BONSAI_GROUP_SIZE; b++, block += DS4_BONSAI_PTQ1_0_BYTES) {
        const float scale = ds4_bonsai_load_f16_le(block + 26);
        for (unsigned j = 0; j < DS4_BONSAI_GROUP_SIZE; j++) {
            unsigned byte, trit;
            /* The 24 qs bytes use a 16-lane group (80 elements), then an
             * 8-lane group (40). qh[2] holds the last 8 in 2-lane order. */
            if (j < 80) { byte = j % 16; trit = j / 16; }
            else if (j < 120) { byte = 16 + (j - 80) % 8; trit = (j - 80) / 8; }
            else { byte = 24 + (j - 120) % 2; trit = (j - 120) / 2; }
            /* The modulo-256 truncation precedes the multiply by 3. The
             * stored byte is ceil(base3_code * 256 / 243), not base3_code. */
            const uint8_t q = (uint8_t)(block[byte] * power3[trit]);
            const int code = ((unsigned)q * 3u) >> 8;
            dst[b * DS4_BONSAI_GROUP_SIZE + j] = (float)(code - 1) * scale;
        }
    }
    return 1;
}

static inline int ds4_bonsai_dequantize_row(uint32_t type, const void *src, float *dst, size_t n) {
    if (type == DS4_BONSAI_PQ2_0) return ds4_bonsai_dequantize_pq2_0(src, dst, n);
    if (type == DS4_BONSAI_PTQ1_0) return ds4_bonsai_dequantize_ptq1_0(src, dst, n);
    return 0;
}

/* One activation/embedding row, with independent 1024-wide Sylvester blocks.
 * Forward is H * (signs * src); inverse is signs * (H * src). The normalized
 * H is symmetric and H*H = I. A sign vector spans the full row, not one block;
 * NULL means identity signs. Explicit signs must be +/-1 and not overlap dst.
 * In-place and overlapping src/dst are supported. Invalid arguments leave dst
 * untouched. This does not perform the separate GDN grouped-head permutation.
 *
 * This butterfly preserves the specified transform, but does not promise
 * bitwise equality with a GPU dense-matmul reduction on arbitrary floats.
 */
static inline int ds4_bonsai_hadamard_transform(float *dst, const float *src,
                                              const int32_t *signs, size_t width, int inverse) {
    if (!dst || !src || !width || width % DS4_BONSAI_HADAMARD_SIZE ||
        width > SIZE_MAX / sizeof(float) || (inverse != 0 && inverse != 1)) return 0;
    if (signs) for (size_t i = 0; i < width; i++) {
        if (signs[i] != 1 && signs[i] != -1) return 0;
    }
    memmove(dst, src, width * sizeof(float));
    if (!inverse && signs) for (size_t i = 0; i < width; i++) dst[i] *= (float)signs[i];
    for (size_t b = 0; b < width; b += DS4_BONSAI_HADAMARD_SIZE) {
        for (size_t step = 1; step < DS4_BONSAI_HADAMARD_SIZE; step *= 2) {
            for (size_t i = 0; i < DS4_BONSAI_HADAMARD_SIZE; i += 2 * step) {
                for (size_t j = 0; j < step; j++) {
                    const float a = dst[b + i + j], c = dst[b + i + j + step];
                    dst[b + i + j] = a + c;
                    dst[b + i + j + step] = a - c;
                }
            }
        }
    }
    for (size_t i = 0; i < width; i++) {
        dst[i] *= 1.0f / 32.0f;
        if (inverse && signs) dst[i] *= (float)signs[i];
    }
    return 1;
}

static inline int ds4_bonsai_hadamard_forward(float *dst, const float *src,
                                            const int32_t *signs, size_t width) {
    return ds4_bonsai_hadamard_transform(dst, src, signs, width, 0);
}

static inline int ds4_bonsai_hadamard_inverse(float *dst, const float *src,
                                            const int32_t *signs, size_t width) {
    return ds4_bonsai_hadamard_transform(dst, src, signs, width, 1);
}

#endif /* DS4_BONSAI_QUANT_H */
