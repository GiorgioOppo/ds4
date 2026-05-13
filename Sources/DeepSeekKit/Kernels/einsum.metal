#include <metal_stdlib>
using namespace metal;

// Two grouped einsum kernels needed for V4 attention:
//
// 1. einsum_bshd_btd_to_bsht
//    Indexer score: index_score = einsum("bshd,btd->bsht", q, kv)
//    q  : [B, S, H, D]   (per-head queries)
//    kv : [B, T, D]      (compressed KV with single shared head)
//    out: [B, S, H, T]
//    out[b, s, h, t] = Σ_d q[b, s, h, d] * kv[b, t, d]
//
// 2. einsum_bsgd_grd_to_bsgr
//    MLA grouped output: einsum("bsgd,grd->bsgr", o_view, wo_a_view)
//    o     : [B, S, G, D]
//    wo_a  : [G, R, D]
//    out   : [B, S, G, R]
//    out[b, s, g, r] = Σ_d o[b, s, g, d] * wo_a[g, r, d]
//
// Each thread handles one output cell. Loop over the contracted axis.

kernel void einsum_bshd_btd_to_bsht_f32(
    device const float* q     [[buffer(0)]],
    device const float* kv    [[buffer(1)]],
    device float*       out_  [[buffer(2)]],
    constant uint4&     dims  [[buffer(3)]],   // (B, S, H, D)
    constant uint&      T     [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint t = gid.x, sh = gid.y, b = gid.z;
    uint B = dims.x, S = dims.y, H = dims.z, D = dims.w;
    if (b >= B || sh >= S * H || t >= T) return;
    uint s = sh / H;
    uint h = sh - s * H;

    uint qOff = ((b * S + s) * H + h) * D;
    uint kvOff = (b * T + t) * D;
    float acc = 0.0f;
    for (uint d = 0; d < D; d++) {
        acc += q[qOff + d] * kv[kvOff + d];
    }
    out_[((b * S + s) * H + h) * T + t] = acc;
}

kernel void einsum_bsgd_grd_to_bsgr_f32(
    device const float* o     [[buffer(0)]],
    device const float* wo_a  [[buffer(1)]],
    device float*       out_  [[buffer(2)]],
    constant uint4&     dims  [[buffer(3)]],   // (B, S, G, D)
    constant uint&      R     [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint r = gid.x, sg = gid.y, b = gid.z;
    uint B = dims.x, S = dims.y, G = dims.z, D = dims.w;
    if (b >= B || sg >= S * G || r >= R) return;
    uint s = sg / G;
    uint g = sg - s * G;

    uint oOff = ((b * S + s) * G + g) * D;
    uint wOff = (g * R + r) * D;
    float acc = 0.0f;
    for (uint d = 0; d < D; d++) {
        acc += o[oOff + d] * wo_a[wOff + d];
    }
    out_[((b * S + s) * G + g) * R + r] = acc;
}

// BF16 wo_a variant. o stays f32, wo_a is bf16 (per the INT8 converter,
// which skips wo_a in the INT8 path and leaves it as bf16).
kernel void einsum_bsgd_grd_to_bsgr_bf16wo(
    device const float*  o     [[buffer(0)]],
    device const bfloat* wo_a  [[buffer(1)]],
    device float*        out_  [[buffer(2)]],
    constant uint4&      dims  [[buffer(3)]],
    constant uint&       R     [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint r = gid.x, sg = gid.y, b = gid.z;
    uint B = dims.x, S = dims.y, G = dims.z, D = dims.w;
    if (b >= B || sg >= S * G || r >= R) return;
    uint s = sg / G;
    uint g = sg - s * G;

    uint oOff = ((b * S + s) * G + g) * D;
    uint wOff = (g * R + r) * D;
    float acc = 0.0f;
    for (uint d = 0; d < D; d++) {
        acc += o[oOff + d] * float(wo_a[wOff + d]);
    }
    out_[((b * S + s) * G + g) * R + r] = acc;
}

// FP8-E4M3 wo_a + UE8M0 per-block scale. Used when the loaded checkpoint
// keeps wo_a in its native FP8 form (DeepSeek-V4-HF). Weight is logically
// reshaped to [G, R, D] but stored linearly as [G*R, D] FP8; the scale is
// [G*R/128, D/128] UE8M0 (1 byte per scale), 128×128 2D blocks.
//
// Each thread computes one (b, s, g, r) output cell, iterating over D in
// blocks of 128. The active block-row in scale-space is (g*R+r)/128 and
// the active block-column is d/128. Bytes are dequantized inline so the
// scale buffer can stay 4× smaller than an f32 tensor would be.
inline float fp8wo_deq_e4m3(uchar b) {
    uint sign = (uint)(b >> 7) & 1u;
    uint exp = (uint)(b >> 3) & 0xFu;
    uint mant = (uint)b & 0x7u;
    if (exp == 0u && mant == 0u) return sign != 0u ? -0.0f : 0.0f;
    if (exp == 0xFu && mant == 0x7u) return NAN;
    if (exp == 0u) {
        float v = (float)mant * 0x1p-9f;
        return sign != 0u ? -v : v;
    }
    uint bits = (sign << 31) | ((exp + 120u) << 23) | (mant << 20);
    return as_type<float>(bits);
}
inline float fp8wo_deq_e8m0(uchar b) {
    if (b == 0xFFu) return NAN;
    return as_type<float>(((uint)b) << 23);
}
constant uint FP8WO_BLOCK = 128u;

kernel void einsum_bsgd_grd_to_bsgr_fp8wo(
    device const float* o      [[buffer(0)]],
    device const uchar* wo_a   [[buffer(1)]],   // [G*R, D] FP8
    device const uchar* w_sc   [[buffer(2)]],   // [G*R/128, D/128] UE8M0
    device float*       out_   [[buffer(3)]],
    constant uint4&     dims   [[buffer(4)]],   // (B, S, G, D)
    constant uint&      R      [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint r = gid.x, sg = gid.y, b = gid.z;
    uint B = dims.x, S = dims.y, G = dims.z, D = dims.w;
    if (b >= B || sg >= S * G || r >= R) return;
    uint s = sg / G;
    uint g = sg - s * G;

    uint flatRow      = g * R + r;
    uint scaleBlockR  = flatRow / FP8WO_BLOCK;
    uint blocksK      = D / FP8WO_BLOCK;
    uint oOff         = ((b * S + s) * G + g) * D;
    uint wOff         = flatRow * D;

    float acc = 0.0f;
    for (uint kb = 0; kb < blocksK; kb++) {
        float scl = fp8wo_deq_e8m0(w_sc[scaleBlockR * blocksK + kb]);
        float block_acc = 0.0f;
        uint k0 = kb * FP8WO_BLOCK;
        for (uint k = 0; k < FP8WO_BLOCK; k++) {
            uint d = k0 + k;
            float w = fp8wo_deq_e4m3(wo_a[wOff + d]);
            block_acc += o[oOff + d] * w;
        }
        acc += block_acc * scl;
    }
    out_[((b * S + s) * G + g) * R + r] = acc;
}
