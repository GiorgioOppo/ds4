#include <metal_stdlib>
using namespace metal;

// act_quant — block-wise activation quantization.
// Port of `act_quant_kernel` and `fp4_quant_kernel` from
// Original/DeepSeek-V4-Pro/inference/kernel.py.
//
// One threadgroup per (row, block). Threadgroup size == block_size, so each
// thread owns one element of the block. The block runs the absmax reduction
// in threadgroup memory, then every thread encodes its element.
//
// Two FP4 packing rules are used by the reference:
//   - scale block size  = 32 (same dim as packing block)
//   - 2 nibbles per byte, packed along the last (K) axis
// Even-tid threads write the combined byte after a barrier.

inline float fast_pow2_exp(int e) {
    uint bits = ((uint)(e + 127)) << 23;
    return as_type<float>(bits);
}

// ceil(log2(x)) for x > 0. Matches kernel.py:fast_log2_ceil.
inline int fast_log2_ceil(float x) {
    uint bits = as_type<uint>(x);
    int exp_x = (int)((bits >> 23) & 0xFF);
    uint mant = bits & 0x7FFFFFu;
    return exp_x - 127 + (mant != 0 ? 1 : 0);
}

inline float round_pow2_scale(float amax, float max_inv) {
    return fast_pow2_exp(fast_log2_ceil(amax * max_inv));
}

// ---------- FP8 E4M3FN ----------------------------------------------------
//
// 1 sign + 4 exp (bias 7) + 3 mantissa. NaN encoding is exp=0xF & m=0x7 only.
// Max finite = 1.75 * 2^8 = 448.
//
// Encode: route through half-precision (1+5+10) and re-bias / re-round
// the mantissa to 3 bits (RTNE).

inline uchar f32_to_e4m3(float x) {
    if (isnan(x)) return 0x7Fu;
    x = clamp(x, -448.0f, 448.0f);
    half h = (half)x;
    ushort hb = as_type<ushort>(h);
    uchar sign = (uchar)((hb >> 15) & 1u);
    int exp16 = (int)((hb >> 10) & 0x1Fu);
    uint mant10 = (uint)(hb & 0x3FFu);

    if (exp16 == 0 && mant10 == 0u) return sign << 7;

    int new_exp = exp16 - 15 + 7;

    if (new_exp <= 0) {
        if (exp16 == 0) {
            // Half subnormals: < 2^-14 ≪ smallest e4m3 subnormal (2^-9). → 0.
            return sign << 7;
        }
        // Half normal that becomes e4m3 subnormal.
        // Implicit-1 mantissa = (1024 + mant10), 11 bits.
        // Subnormal e4m3 value = m * 2^-9, where m in [0, 7].
        int shift = 17 - exp16;             // shift right by `shift` to get m
        uint full = 1024u + mant10;         // 11 bits
        uint result = full >> shift;
        uint round_bit = (full >> (shift - 1)) & 1u;
        uint sticky = full & ((1u << (shift - 1)) - 1u);
        if (round_bit != 0u && (sticky != 0u || (result & 1u) != 0u)) result += 1u;
        if (result >= 8u) {
            // Rounded up to smallest normal: e=1, m=0.
            return (sign << 7) | (uchar)(1u << 3);
        }
        return (sign << 7) | (uchar)result;
    }

    // Normal e4m3.
    uint mant3 = mant10 >> 7;               // top 3 bits
    uint round_bit = (mant10 >> 6) & 1u;
    uint sticky = mant10 & 0x3Fu;
    if (round_bit != 0u && (sticky != 0u || (mant3 & 1u) != 0u)) {
        mant3 += 1u;
        if (mant3 == 8u) { mant3 = 0u; new_exp += 1; }
    }
    if (new_exp >= 16) return (sign << 7) | 0x7Eu;          // saturate
    if (new_exp == 15 && mant3 == 7u) return (sign << 7) | 0x7Eu;  // 0x7F is NaN
    return (sign << 7) | (uchar)((uint)new_exp << 3) | (uchar)mant3;
}

inline float e4m3_to_f32(uchar b) {
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

// ---------- FP4 E2M1 ------------------------------------------------------
//
// 1 sign + 2 exp (bias 1) + 1 mantissa. Eight magnitudes:
//   0, 0.5, 1, 1.5, 2, 3, 4, 6
// Round-to-nearest with midpoints (ties to upper) — close enough to RTNE
// for QAT-style activation noise.

inline uchar f32_to_e2m1(float x) {
    uchar sign = (x < 0.0f) ? 8u : 0u;
    float a = fabs(x);
    uchar mag;
    if      (a < 0.25f) mag = 0u;
    else if (a < 0.75f) mag = 1u;
    else if (a < 1.25f) mag = 2u;
    else if (a < 1.75f) mag = 3u;
    else if (a < 2.5f)  mag = 4u;
    else if (a < 3.5f)  mag = 5u;
    else if (a < 5.0f)  mag = 6u;
    else                mag = 7u;
    return sign | mag;
}

inline float e2m1_to_f32(uchar b) {
    uchar mag = b & 7u;
    float v;
    switch (mag) {
        case 0u: v = 0.0f; break;
        case 1u: v = 0.5f; break;
        case 2u: v = 1.0f; break;
        case 3u: v = 1.5f; break;
        case 4u: v = 2.0f; break;
        case 5u: v = 3.0f; break;
        case 6u: v = 4.0f; break;
        default: v = 6.0f; break;
    }
    return ((b & 8u) != 0u) ? -v : v;
}

// ---------- Kernels -------------------------------------------------------
//
// Buffer indices (shared across both kernels):
//   0  x_in           [M, N]            f32
//   1  y_out_qbytes   [M, N]            uchar (FP8) or [M, N/2] uchar (FP4)  — non-inplace
//   2  y_out_inplace  [M, N]            f32                                  — inplace round-trip
//   3  scales_out     [M, N/blockSize]  f32 (E8M0 stored as float for now)
//   4  block_size     uint
//   5  inplace_flag   uint

constant uint BLOCK_SIZE_FP8 [[function_constant(0)]];

kernel void act_quant_fp8(
    device const float* x_in     [[buffer(0)]],
    device uchar*       y_q      [[buffer(1)]],
    device float*       y_ip     [[buffer(2)]],
    device float*       scales   [[buffer(3)]],
    constant uint2&     dims     [[buffer(4)]],   // (M, N)
    constant uint&      inplace  [[buffer(5)]],
    threadgroup float*  shared_  [[threadgroup(0)]],
    uint2 tg      [[threadgroup_position_in_grid]],
    uint  tid     [[thread_position_in_threadgroup]],
    uint  tgsize  [[threads_per_threadgroup]]
) {
    uint M = dims.x, N = dims.y;
    uint blkSize = BLOCK_SIZE_FP8;
    uint row = tg.x;
    uint blkIdx = tg.y;
    uint blockStart = blkIdx * blkSize;
    if (row >= M || blockStart >= N) return;

    uint idx = row * N + blockStart + tid;
    float v = x_in[idx];
    float a = fabs(v);

    // Reduction in threadgroup memory.
    shared_[tid] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shared_[tid] = max(shared_[tid], shared_[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float amax = max(shared_[0], 1e-4f);
    float scale = round_pow2_scale(amax, 1.0f / 448.0f);

    if (tid == 0u) scales[row * (N / blkSize) + blkIdx] = scale;

    float clipped = clamp(v / scale, -448.0f, 448.0f);
    uchar qb = f32_to_e4m3(clipped);

    if (inplace != 0u) {
        y_ip[idx] = e4m3_to_f32(qb) * scale;
    } else {
        y_q[idx] = qb;
    }
}

constant uint BLOCK_SIZE_FP4 [[function_constant(10)]];

kernel void act_quant_fp4(
    device const float* x_in     [[buffer(0)]],
    device uchar*       y_q      [[buffer(1)]],   // packed: [M, N/2]
    device float*       y_ip     [[buffer(2)]],
    device float*       scales   [[buffer(3)]],
    constant uint2&     dims     [[buffer(4)]],
    constant uint&      inplace  [[buffer(5)]],
    threadgroup float*  shared_  [[threadgroup(0)]],
    uint2 tg      [[threadgroup_position_in_grid]],
    uint  tid     [[thread_position_in_threadgroup]],
    uint  tgsize  [[threads_per_threadgroup]]
) {
    uint M = dims.x, N = dims.y;
    uint blkSize = BLOCK_SIZE_FP4;
    uint row = tg.x;
    uint blkIdx = tg.y;
    uint blockStart = blkIdx * blkSize;
    if (row >= M || blockStart >= N) return;

    uint idx = row * N + blockStart + tid;
    float v = x_in[idx];
    float a = fabs(v);

    shared_[tid] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shared_[tid] = max(shared_[tid], shared_[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // FP4 amax floor: 6 * 2^-126 (matches the reference).
    float amax = max(shared_[0], 6.0f * 0x1p-126f);
    float scale = round_pow2_scale(amax, 1.0f / 6.0f);

    if (tid == 0u) scales[row * (N / blkSize) + blkIdx] = scale;

    float clipped = clamp(v / scale, -6.0f, 6.0f);
    uchar nib = f32_to_e2m1(clipped);

    if (inplace != 0u) {
        y_ip[idx] = e2m1_to_f32(nib) * scale;
    } else {
        // Pack two nibbles into one byte. Even-tid contributes the low nibble,
        // odd-tid the high nibble. Use threadgroup memory to exchange.
        threadgroup uchar* nibs = (threadgroup uchar*)shared_;   // reuse shared
        nibs[tid] = nib;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((tid & 1u) == 0u) {
            uchar lo = nibs[tid];
            uchar hi = nibs[tid + 1];
            uint outIdx = row * (N / 2u) + (blockStart + tid) / 2u;
            y_q[outIdx] = (uchar)(lo | (hi << 4));
        }
    }
}
