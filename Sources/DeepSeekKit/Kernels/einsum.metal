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
