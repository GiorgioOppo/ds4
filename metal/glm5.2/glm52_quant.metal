// GLM 5.2 — helper di dequant/dot per i K-quant (per-thread e simdgroup) — PRIMA di glm52_moe nell'\''ordine
// Parte della libreria concatenata: vedi MetalRuntime.kernelFiles per l'ordine.

#include <metal_stdlib>
using namespace metal;

static inline float glm52_half_at(device const uchar *p) {
    return (float)(*(device const half *)p);
}

static inline uchar2 glm52_scale_min_k4(uint j, device const uchar *q) {
    if (j < 4u) {
        return uchar2(q[j] & 63u, q[j + 4u] & 63u);
    }
    return uchar2((q[j + 4u] & 0x0Fu) | ((q[j - 4u] >> 6u) << 4u),
                  (q[j + 4u] >> 4u) | ((q[j] >> 6u) << 4u));
}

static inline float glm52_dot_q2_K_row(device const uchar *row,
                                       device const float *x,
                                       uint width) {
    float acc = 0.0f;
    for (uint sb = 0u; sb < width / 256u; sb++) {
        device const uchar *base = row + sb * 84u;
        device const uchar *qs = base + 16u;
        const float d = glm52_half_at(base + 80u);
        const float dmin = glm52_half_at(base + 82u);
        for (uint half128 = 0u; half128 < 256u; half128 += 128u) {
            for (uint plane = 0u; plane < 4u; plane++) {
                for (uint l = 0u; l < 32u; l++) {
                    const uchar sc = base[half128 / 16u + plane * 2u + l / 16u];
                    const uint q = (qs[half128 / 4u + l] >> (plane * 2u)) & 3u;
                    const float w = d * (float)(sc & 0x0Fu) * (float)q -
                        dmin * (float)(sc >> 4u);
                    acc += w * x[sb * 256u + half128 + plane * 32u + l];
                }
            }
        }
    }
    return acc;
}

static inline float glm52_dot_q4_K_row(device const uchar *row,
                                       device const float *x,
                                       uint width) {
    float acc = 0.0f;
    for (uint sb = 0u; sb < width / 256u; sb++) {
        device const uchar *base = row + sb * 144u;
        const float d = glm52_half_at(base);
        const float dmin = glm52_half_at(base + 2u);
        device const uchar *scales = base + 4u;
        device const uchar *qs = base + 16u;
        for (uint j = 0u; j < 8u; j++) {
            const uchar2 sm = glm52_scale_min_k4(j, scales);
            const float dj = d * (float)sm.x;
            const float mj = dmin * (float)sm.y;
            const uint chunk = (j / 2u) * 32u;
            const uint shift = (j & 1u) * 4u;
            for (uint l = 0u; l < 32u; l++) {
                const uint q = (qs[chunk + l] >> shift) & 0x0Fu;
                acc += (dj * (float)q - mj) * x[sb * 256u + j * 32u + l];
            }
        }
    }
    return acc;
}

static inline float glm52_dot_q5_K_row(device const uchar *row,
                                       device const float *x,
                                       uint width) {
    float acc = 0.0f;
    for (uint sb = 0u; sb < width / 256u; sb++) {
        device const uchar *base = row + sb * 176u;
        const float d = glm52_half_at(base);
        const float dmin = glm52_half_at(base + 2u);
        device const uchar *scales = base + 4u;
        device const uchar *qh = base + 16u;
        device const uchar *qs = base + 48u;
        for (uint j = 0u; j < 8u; j++) {
            const uchar2 sm = glm52_scale_min_k4(j, scales);
            const float dj = d * (float)sm.x;
            const float mj = dmin * (float)sm.y;
            const uint chunk = (j / 2u) * 32u;
            const uint shift = (j & 1u) * 4u;
            for (uint l = 0u; l < 32u; l++) {
                const uint q = ((qs[chunk + l] >> shift) & 0x0Fu) +
                    ((qh[l] >> j) & 1u) * 16u;
                acc += (dj * (float)q - mj) * x[sb * 256u + j * 32u + l];
            }
        }
    }
    return acc;
}

// Q3_K (110 B / 256 elementi): hmask[32] + qs[64] (piani a 2 bit) +
// scales[12] (16 scale a 6 bit con segno, impacchettamento llama.cpp) + d.
// Valore dell'elemento: (2 bit bassi | bit alto da hmask) − 4 quando il bit
// alto è a zero. Consumatore: gli esperti instradati del file Laguna misto
// RoutedQ2_K-Last27Q3_K (GLM non pubblica pesi Q3_K).
static inline int glm52_q3_K_scale(device const uchar *sc, uint k) {
    const uint low = k < 8u ? (uint)(sc[k] & 0x0Fu)
                            : (uint)(sc[k - 8u] >> 4u);
    const uint high = ((uint)sc[8u + (k % 4u)] >> (2u * (k / 4u))) & 3u;
    return (int)(low | (high << 4u)) - 32;
}

static inline float glm52_dot_q3_K_row(device const uchar *row,
                                       device const float *x,
                                       uint width) {
    float acc = 0.0f;
    for (uint sb = 0u; sb < width / 256u; sb++) {
        device const uchar *base = row + sb * 110u;
        device const uchar *hm = base;
        device const uchar *qs = base + 32u;
        device const uchar *sc = base + 96u;
        const float d = glm52_half_at(base + 108u);
        for (uint j = 0u; j < 8u; j++) {
            const uint plane = j % 4u;
            const uint qbase = (j / 4u) * 32u;
            const uint shift = plane * 2u;
            const uchar mbit = (uchar)(1u << ((j / 4u) * 4u + plane));
            const uint xbase = sb * 256u + (j / 4u) * 128u + plane * 32u;
            float sums[2] = {0.0f, 0.0f};
            for (uint l = 0u; l < 32u; l++) {
                const int q = (int)((qs[qbase + l] >> shift) & 3u);
                const int v = q - (((hm[l] & mbit) != 0u) ? 0 : 4);
                sums[l / 16u] += (float)v * x[xbase + l];
            }
            const uint s0 = (j / 4u) * 8u + plane * 2u;
            acc += d * (float)glm52_q3_K_scale(sc, s0) * sums[0];
            acc += d * (float)glm52_q3_K_scale(sc, s0 + 1u) * sums[1];
        }
    }
    return acc;
}

static inline float glm52_dot_q6_K_row(device const uchar *row,
                                       device const float *x,
                                       uint width) {
    float acc = 0.0f;
    for (uint sb = 0u; sb < width / 256u; sb++) {
        device const uchar *base = row + sb * 210u;
        device const uchar *ql = base;
        device const uchar *qh = base + 128u;
        device const char *scales = (device const char *)(base + 192u);
        const float d = glm52_half_at(base + 208u);
        for (uint half128 = 0u; half128 < 256u; half128 += 128u) {
            const uint qlHalf = half128 / 2u;
            const uint qhHalf = half128 / 4u;
            const uint scHalf = half128 / 16u;
            for (uint l = 0u; l < 32u; l++) {
                const uint sub = l / 16u;
                const uchar high = qh[qhHalf + l];
                const int q1 = (int)((ql[qlHalf + l] & 0x0Fu) | ((high << 4u) & 0x30u)) - 32;
                const int q2 = (int)((ql[qlHalf + l + 32u] & 0x0Fu) | ((high << 2u) & 0x30u)) - 32;
                const int q3 = (int)((ql[qlHalf + l] >> 4u) | (high & 0x30u)) - 32;
                const int q4 = (int)((ql[qlHalf + l + 32u] >> 4u) | ((high >> 2u) & 0x30u)) - 32;
                const uint out0 = sb * 256u + half128 + l;
                acc += d * (float)scales[scHalf + sub] * (float)q1 * x[out0];
                acc += d * (float)scales[scHalf + sub + 2u] * (float)q2 * x[out0 + 32u];
                acc += d * (float)scales[scHalf + sub + 4u] * (float)q3 * x[out0 + 64u];
                acc += d * (float)scales[scHalf + sub + 6u] * (float)q4 * x[out0 + 96u];
            }
        }
    }
    return acc;
}

static inline float glm52_dot_q8_0_row(device const uchar *row,
                                       device const float *x,
                                       uint width) {
    float acc = 0.0f;
    for (uint b = 0u; b < width / 32u; b++) {
        device const uchar *base = row + b * 34u;
        const float d = glm52_half_at(base);
        device const char *qs = (device const char *)(base + 2u);
        for (uint i = 0u; i < 32u; i++) {
            acc += d * (float)qs[i] * x[b * 32u + i];
        }
    }
    return acc;
}

// IQ2_XXS: 66-byte superblocks of 256 — f16 scale + 32 uint16 words. Each
// 32-value group decodes four 8-value cells through the shared grid/sign
// tables the DeepSeek prelude already embeds (same concatenated library),
// with db = d * (0.5 + top-nibble) * 0.25. Same emission order as
// Quantize.dequantIQ2_XXS, the CPU reference.
static inline float glm52_dot_iq2_xxs_row(device const uchar *row,
                                          device const float *x,
                                          uint width) {
    float acc = 0.0f;
    for (uint sb = 0u; sb < width / 256u; sb++) {
        device const uchar *base = row + sb * 66u;
        const float d = glm52_half_at(base);
        device const ushort *qs = (device const ushort *)(base + 2u);
        for (uint ib = 0u; ib < 8u; ib++) {
            const uint aux0 = (uint)qs[4u * ib]
                | ((uint)qs[4u * ib + 1u] << 16u);
            const uint aux1 = (uint)qs[4u * ib + 2u]
                | ((uint)qs[4u * ib + 3u] << 16u);
            const float db = d * (0.5f + (float)(aux1 >> 28u)) * 0.25f;
            for (uint l = 0u; l < 4u; l++) {
                constant uchar *grid = (constant uchar *)
                    (ds4_metal_iq2xxs_grid + ((aux0 >> (8u * l)) & 255u));
                const uchar signs =
                    ds4_metal_ksigns_iq2xs[(aux1 >> (7u * l)) & 127u];
                const uint out0 = sb * 256u + ib * 32u + l * 8u;
                for (uint j = 0u; j < 8u; j++) {
                    const float w = db * (float)grid[j]
                        * (((signs >> j) & 1u) != 0u ? -1.0f : 1.0f);
                    acc += w * x[out0 + j];
                }
            }
        }
    }
    return acc;
}

static inline uint glm52_kquant_row_bytes(uint type, uint width) {
    switch (type) {
        case 8u:  return (width / 32u) * 34u;
        case 10u: return (width / 256u) * 84u;
        case 11u: return (width / 256u) * 110u;
        case 12u: return (width / 256u) * 144u;
        case 13u: return (width / 256u) * 176u;
        case 14u: return (width / 256u) * 210u;
        case 16u: return (width / 256u) * 66u;
        default:  return 0u;
    }
}

static inline float glm52_dot_kquant_row(uint type,
                                         device const uchar *row,
                                         device const float *x,
                                         uint width) {
    switch (type) {
        case 8u:  return glm52_dot_q8_0_row(row, x, width);
        case 10u: return glm52_dot_q2_K_row(row, x, width);
        case 11u: return glm52_dot_q3_K_row(row, x, width);
        case 12u: return glm52_dot_q4_K_row(row, x, width);
        case 13u: return glm52_dot_q5_K_row(row, x, width);
        case 14u: return glm52_dot_q6_K_row(row, x, width);
        case 16u: return glm52_dot_iq2_xxs_row(row, x, width);
        default:  return 0.0f;
    }
}

static inline float glm52_silu(float value) {
    // Stable sigmoid, matching upstream sigmoid_stable.
    const float s = value >= 0.0f
        ? 1.0f / (1.0f + exp(-value))
        : exp(value) / (1.0f + exp(value));
    return value * s;
}

// MARK: - Cooperative (simdgroup-per-row) matvec variants
//
// The per-thread kernels above walk a WHOLE row per thread — a strided,
// cache-hostile access pattern that leaves most of the memory bandwidth on
// the table. These variants put one SIMD group on each row: the 32 lanes
// stride the row's 32-element quant groups (so consecutive lanes read
// consecutive weight blocks) and reduce with simd_sum. Same math, same
// inputs, float-accumulation order differs only by the reduction tree.
// The per-thread originals remain the reference and the DS4_GLM_SG=0
// fallback.

// Partial dot of ONE 32-element group `g` of a row: sb = g/8 selects the
// 256-wide superblock (g IS the block for q8_0), j = g%8 the 32-chunk
// inside it. Emission matches the row helpers above exactly.
// VETTORIZZATO: letture short (il blocco q8_0 è largo 34 B, qs parte a +2 —
// allineamento 2, mai 4: char4/uint sarebbero UB) spacchettate con as_type e
// FMA float4. Stessa matematica del loop scalare a meno dell'ordine delle
// somme (4 corsie parallele, d fattorizzato): entro la tolleranza dei test.
static inline float glm52_dot_q8_0_group(device const uchar *row,
                                         device const float *x,
                                         uint g) {
    device const uchar *base = row + g * 34u;
    const float d = glm52_half_at(base);
    device const short *qs = (device const short *)(base + 2u);
    device const float4 *xv = (device const float4 *)(x + g * 32u);
    float4 acc = 0.0f;
    for (uint i = 0u; i < 8u; i++) {
        const char2 a = as_type<char2>(qs[2u * i]);
        const char2 b = as_type<char2>(qs[2u * i + 1u]);
        acc += float4((float)a.x, (float)a.y,
                      (float)b.x, (float)b.y) * xv[i];
    }
    return d * (acc.x + acc.y + acc.z + acc.w);
}

static inline float glm52_dot_q2_K_group(device const uchar *row,
                                         device const float *x,
                                         uint g) {
    const uint sb = g / 8u;
    const uint j = g % 8u;
    const uint half128 = (j / 4u) * 128u;
    const uint plane = j % 4u;
    device const uchar *base = row + sb * 84u;
    device const uchar *qs = base + 16u;
    const float d = glm52_half_at(base + 80u);
    const float dmin = glm52_half_at(base + 82u);
    float acc = 0.0f;
    for (uint l = 0u; l < 32u; l++) {
        const uchar sc = base[half128 / 16u + plane * 2u + l / 16u];
        const uint q = (qs[half128 / 4u + l] >> (plane * 2u)) & 3u;
        const float w = d * (float)(sc & 0x0Fu) * (float)q -
            dmin * (float)(sc >> 4u);
        acc += w * x[sb * 256u + half128 + plane * 32u + l];
    }
    return acc;
}

static inline float glm52_dot_q3_K_group(device const uchar *row,
                                         device const float *x,
                                         uint g) {
    const uint sb = g / 8u;
    const uint j = g % 8u;
    const uint plane = j % 4u;
    device const uchar *base = row + sb * 110u;
    device const uchar *hm = base;
    device const uchar *qs = base + 32u;
    device const uchar *sc = base + 96u;
    const float d = glm52_half_at(base + 108u);

    const uint qbase = (j / 4u) * 32u;
    const uint shift = plane * 2u;
    const uchar mbit = (uchar)(1u << ((j / 4u) * 4u + plane));
    const uint xbase = sb * 256u + (j / 4u) * 128u + plane * 32u;
    float sums[2] = {0.0f, 0.0f};
    for (uint l = 0u; l < 32u; l++) {
        const int q = (int)((qs[qbase + l] >> shift) & 3u);
        const int v = q - (((hm[l] & mbit) != 0u) ? 0 : 4);
        sums[l / 16u] += (float)v * x[xbase + l];
    }
    const uint s0 = (j / 4u) * 8u + plane * 2u;
    return d * (float)glm52_q3_K_scale(sc, s0) * sums[0] +
           d * (float)glm52_q3_K_scale(sc, s0 + 1u) * sums[1];
}

static inline float glm52_dot_q4_K_group(device const uchar *row,
                                         device const float *x,
                                         uint g) {
    const uint sb = g / 8u;
    const uint j = g % 8u;
    device const uchar *base = row + sb * 144u;
    const float d = glm52_half_at(base);
    const float dmin = glm52_half_at(base + 2u);
    const uchar2 sm = glm52_scale_min_k4(j, base + 4u);
    const float dj = d * (float)sm.x;
    const float mj = dmin * (float)sm.y;
    device const uchar *qs = base + 16u;
    const uint chunk = (j / 2u) * 32u;
    const uint shift = (j & 1u) * 4u;
    // VETTORIZZATO: il blocco q4_K è largo 144 B e qs parte a +16 con chunk
    // multipli di 32 — letture uchar4 allineate; nibble e FMA in float4.
    device const uchar4 *qv = (device const uchar4 *)(qs + chunk);
    device const float4 *xv =
        (device const float4 *)(x + sb * 256u + j * 32u);
    float4 acc = 0.0f;
    for (uint l = 0u; l < 8u; l++) {
        const uchar4 q = qv[l];
        acc += (dj * float4((float)((q.x >> shift) & 0x0Fu),
                            (float)((q.y >> shift) & 0x0Fu),
                            (float)((q.z >> shift) & 0x0Fu),
                            (float)((q.w >> shift) & 0x0Fu))
                - mj) * xv[l];
    }
    return acc.x + acc.y + acc.z + acc.w;
}

static inline float glm52_dot_q5_K_group(device const uchar *row,
                                         device const float *x,
                                         uint g) {
    const uint sb = g / 8u;
    const uint j = g % 8u;
    device const uchar *base = row + sb * 176u;
    const float d = glm52_half_at(base);
    const float dmin = glm52_half_at(base + 2u);
    const uchar2 sm = glm52_scale_min_k4(j, base + 4u);
    const float dj = d * (float)sm.x;
    const float mj = dmin * (float)sm.y;
    device const uchar *qh = base + 16u;
    device const uchar *qs = base + 48u;
    const uint chunk = (j / 2u) * 32u;
    const uint shift = (j & 1u) * 4u;
    float acc = 0.0f;
    for (uint l = 0u; l < 32u; l++) {
        const uint q = ((qs[chunk + l] >> shift) & 0x0Fu) +
            ((qh[l] >> j) & 1u) * 16u;
        acc += (dj * (float)q - mj) * x[sb * 256u + j * 32u + l];
    }
    return acc;
}

static inline float glm52_dot_q6_K_group(device const uchar *row,
                                         device const float *x,
                                         uint g) {
    const uint sb = g / 8u;
    const uint j = g % 8u;
    const uint half128 = (j / 4u) * 128u;
    const uint k = j % 4u;
    device const uchar *base = row + sb * 210u;
    device const uchar *ql = base;
    device const uchar *qh = base + 128u;
    device const char *scales = (device const char *)(base + 192u);
    const float d = glm52_half_at(base + 208u);
    const uint qlHalf = half128 / 2u;
    const uint qhHalf = half128 / 4u;
    const uint scHalf = half128 / 16u;
    float acc = 0.0f;
    for (uint l = 0u; l < 32u; l++) {
        const uint sub = l / 16u;
        const uchar high = qh[qhHalf + l];
        int q;
        float scale;
        switch (k) {
            case 0u:
                q = (int)((ql[qlHalf + l] & 0x0Fu) | ((high << 4u) & 0x30u)) - 32;
                scale = (float)scales[scHalf + sub];
                break;
            case 1u:
                q = (int)((ql[qlHalf + l + 32u] & 0x0Fu) | ((high << 2u) & 0x30u)) - 32;
                scale = (float)scales[scHalf + sub + 2u];
                break;
            case 2u:
                q = (int)((ql[qlHalf + l] >> 4u) | (high & 0x30u)) - 32;
                scale = (float)scales[scHalf + sub + 4u];
                break;
            default:
                q = (int)((ql[qlHalf + l + 32u] >> 4u) | ((high >> 2u) & 0x30u)) - 32;
                scale = (float)scales[scHalf + sub + 6u];
                break;
        }
        acc += d * scale * (float)q * x[sb * 256u + half128 + k * 32u + l];
    }
    return acc;
}

static inline float glm52_dot_iq2_xxs_group(device const uchar *row,
                                            device const float *x,
                                            uint g) {
    const uint sb = g / 8u;
    const uint ib = g % 8u;
    device const uchar *base = row + sb * 66u;
    const float d = glm52_half_at(base);
    device const ushort *qs = (device const ushort *)(base + 2u);
    const uint aux0 = (uint)qs[4u * ib] | ((uint)qs[4u * ib + 1u] << 16u);
    const uint aux1 = (uint)qs[4u * ib + 2u]
        | ((uint)qs[4u * ib + 3u] << 16u);
    const float db = d * (0.5f + (float)(aux1 >> 28u)) * 0.25f;
    // VETTORIZZATO: la voce di griglia è UNA lettura ulong dalla constant
    // memory (8 byte = 8 valori), i segni un byte; unpack e FMA in float4.
    // db fattorizzato fuori dal loop: stessa matematica del percorso
    // scalare a meno dell'ordine delle somme, entro la tolleranza dei test.
    device const float4 *xv =
        (device const float4 *)(x + sb * 256u + ib * 32u);
    float4 acc = 0.0f;
    for (uint l = 0u; l < 4u; l++) {
        const ulong grid = ds4_metal_iq2xxs_grid[(aux0 >> (8u * l)) & 255u];
        const uchar signs = ds4_metal_ksigns_iq2xs[(aux1 >> (7u * l)) & 127u];
        const float4 g0 = float4((float)((grid >>  0u) & 0xFFu),
                                 (float)((grid >>  8u) & 0xFFu),
                                 (float)((grid >> 16u) & 0xFFu),
                                 (float)((grid >> 24u) & 0xFFu));
        const float4 g1 = float4((float)((grid >> 32u) & 0xFFu),
                                 (float)((grid >> 40u) & 0xFFu),
                                 (float)((grid >> 48u) & 0xFFu),
                                 (float)((grid >> 56u) & 0xFFu));
        const float4 s0 = float4((signs & 1u) ? -1.0f : 1.0f,
                                 (signs & 2u) ? -1.0f : 1.0f,
                                 (signs & 4u) ? -1.0f : 1.0f,
                                 (signs & 8u) ? -1.0f : 1.0f);
        const float4 s1 = float4((signs & 16u) ? -1.0f : 1.0f,
                                 (signs & 32u) ? -1.0f : 1.0f,
                                 (signs & 64u) ? -1.0f : 1.0f,
                                 (signs & 128u) ? -1.0f : 1.0f);
        acc += g0 * s0 * xv[2u * l];
        acc += g1 * s1 * xv[2u * l + 1u];
    }
    return db * (acc.x + acc.y + acc.z + acc.w);
}

static inline float glm52_dot_kquant_group(uint type,
                                           device const uchar *row,
                                           device const float *x,
                                           uint g) {
    switch (type) {
        case 8u:  return glm52_dot_q8_0_group(row, x, g);
        case 10u: return glm52_dot_q2_K_group(row, x, g);
        case 11u: return glm52_dot_q3_K_group(row, x, g);
        case 12u: return glm52_dot_q4_K_group(row, x, g);
        case 13u: return glm52_dot_q5_K_group(row, x, g);
        case 14u: return glm52_dot_q6_K_group(row, x, g);
        case 16u: return glm52_dot_iq2_xxs_group(row, x, g);
        default:  return 0.0f;
    }
}

// Dispatch: threadsPerThreadgroup = (32, NSG, 1) — each y-slice is exactly
// one SIMD group — threadgroups = (ceil(row_count / NSG), 1, 1).
