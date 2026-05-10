#include <metal_stdlib>
using namespace metal;

// Apply rotary embedding to the LAST `rope_head_dim` of each head, in place.
// This matches the reference: `apply_rotary_emb(q[..., -rd:], freqs_cis)`.
//
// Layout:
//   x:        [tokens, heads, head_dim] f32 (or BF16 — kept FP32 for now)
//   freqs:    [seqlen, rope_head_dim/2 * 2] f32 — (cos, sin) interleaved per
//             position, precomputed by YaRN.precomputeFreqsCis. The shader
//             reads `freqs[start_pos + t]`.
//   inverse:  if 1, conjugates (sin → -sin). Used to undo the rotation on the
//             output projection in the reference Attention.forward.
//
// Each thread handles one (token, head, pair) where pair indexes the rotary
// half-dim. Pairs are laid out interleaved in memory: positions
// `head_dim - rope_head_dim + 2*i` and `... + 2*i + 1`.
kernel void rope_apply_f32(
    device float*        x          [[buffer(0)]],
    device const float*  freqs      [[buffer(1)]],   // [seqlen, rope_dim/2, 2]
    constant uint4&      dims       [[buffer(2)]],   // (tokens, heads, head_dim, rope_dim)
    constant uint2&      misc       [[buffer(3)]],   // (start_pos, inverse)
    uint3 gid [[thread_position_in_grid]]
) {
    uint t = gid.x, h = gid.y, i = gid.z;
    uint T = dims.x, H = dims.y, D = dims.z, RD = dims.w;
    uint startPos = misc.x;
    uint inverse = misc.y;
    uint half_rd = RD / 2;
    if (t >= T || h >= H || i >= half_rd) return;

    uint pos = startPos + t;
    float c = freqs[2 * (pos * half_rd + i) + 0];
    float s = freqs[2 * (pos * half_rd + i) + 1];
    if (inverse != 0u) s = -s;

    uint base = (t * H + h) * D + (D - RD);
    float a = x[base + 2*i];
    float b = x[base + 2*i + 1];
    x[base + 2*i]     = a * c - b * s;
    x[base + 2*i + 1] = a * s + b * c;
}
