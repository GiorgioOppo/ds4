#include <metal_stdlib>
using namespace metal;

// GGUF block dequant kernels (TODO §10.2 / T2). Each kernel reads
// the raw mmap'd file bytes for one quantized tensor and produces a
// dense float tensor in the per-element layout the rest of the
// engine expects.
//
// Output dtype is f32 for now. A bfloat variant would halve the
// resident memory for the dequantized weights at the cost of a
// second kernel per format; deferred to a follow-up — the F32 path
// is enough to wire up `LlamaDecoderLayer` end-to-end.
//
// Block layouts come from llama.cpp's `ggml-quants.h` and must stay
// bit-identical so a GGUF written by `quantize` reads back the
// same numbers we'd dequant in the reference C path.
//
// One thread per output element. Coarser thread→block mappings
// (one thread per 32-element block) trade store coalescing for
// register-resident scale hoisting; the per-element shape is
// simpler and the bottleneck here is memory bandwidth, not arith.

// --------------------------------------------------------------------
// Q8_0 — block of 32 i8 values + 1×F16 scale. 34 bytes per block.
// --------------------------------------------------------------------
//   bytes [0..1]:   F16 scale d
//   bytes [2..33]:  32 × i8 quants
//   value[k] = float(quant[k]) * float(d)

constant uint Q8_0_BLOCK = 32;
constant uint Q8_0_BYTES = 34;

kernel void dequant_q8_0_to_f32(
    device const uchar* W      [[buffer(0)]],
    device float*       out    [[buffer(1)]],
    constant uint&      nElem  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= nElem) return;
    uint blockIdx = gid / Q8_0_BLOCK;
    uint inner    = gid % Q8_0_BLOCK;
    device const uchar* blk = W + blockIdx * Q8_0_BYTES;

    ushort sBits = ushort(blk[0]) | (ushort(blk[1]) << 8);
    float  scale = float(as_type<half>(sBits));
    char   q     = as_type<char>(blk[2 + inner]);

    out[gid] = float(q) * scale;
}

// --------------------------------------------------------------------
// Q4_0 — block of 32 4-bit values + 1×F16 scale. 18 bytes per block.
// --------------------------------------------------------------------
//   bytes [0..1]:    F16 scale d
//   bytes [2..17]:   16 × packed nibbles. byte[j].lo = q[j], byte[j].hi = q[j+16]
//   raw nibble n is in [0, 15]; the signed value used downstream is
//   (n - 8) (so the natural zero is the middle of the range).
//   value[k] = float(n - 8) * float(d)

constant uint Q4_0_BLOCK = 32;
constant uint Q4_0_BYTES = 18;

kernel void dequant_q4_0_to_f32(
    device const uchar* W      [[buffer(0)]],
    device float*       out    [[buffer(1)]],
    constant uint&      nElem  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= nElem) return;
    uint blockIdx = gid / Q4_0_BLOCK;
    uint inner    = gid % Q4_0_BLOCK;
    device const uchar* blk = W + blockIdx * Q4_0_BYTES;

    ushort sBits = ushort(blk[0]) | (ushort(blk[1]) << 8);
    float  scale = float(as_type<half>(sBits));

    // qs starts at byte 2. Each byte j packs lo→q[j], hi→q[j+16].
    uint byteIdx = (inner < 16) ? inner : (inner - 16);
    uchar pack = blk[2 + byteIdx];
    uint  nibble = (inner < 16) ? (pack & 0xF) : (pack >> 4);
    int   signed_q = int(nibble) - 8;
    out[gid] = float(signed_q) * scale;
}

// --------------------------------------------------------------------
// Q4_K — super-block of 256 elements, 8 sub-blocks of 32. 144 bytes.
// --------------------------------------------------------------------
//   bytes [0..1]:     F16 super-scale d
//   bytes [2..3]:     F16 super-min dmin
//   bytes [4..15]:    12 bytes packed 6-bit scales+mins per sub-block (see get_scale_min_k4 below)
//   bytes [16..143]:  128 bytes packed 4-bit quants (256 nibbles)
//
// For sub-block j (0..7) the dequantized value of element l (0..31) is:
//   value = (d * sc_j) * nibble(j,l) − (dmin * m_j)
//
// Nibble layout inside qs[] (128 bytes): the 256 elements are visited
// in pairs of sub-blocks. For outer iteration j_pair = 0..3:
//   qs bytes [32*j_pair … 32*j_pair + 31] each carry:
//     lo nibble → element (64*j_pair + l)      → sub-block (2*j_pair)
//     hi nibble → element (64*j_pair + 32 + l) → sub-block (2*j_pair + 1)
//   where l = 0..31.
//
// Scales / mins live in the 12-byte `scales[]` array under the
// "K_SCALE_SIZE = 12" packing scheme that llama.cpp's
// `get_scale_min_k4` decodes:
//
//   for j in 0..3:
//     sc[j] = scales[j]   & 0x3F   (low 6 bits)
//     m [j] = scales[j+4] & 0x3F
//   for j in 4..7:
//     sc[j] = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
//     m [j] = (scales[j+4] >>  4) | ((scales[j]   >> 6) << 4)

constant uint Q4_K_BLOCK = 256;
constant uint Q4_K_BYTES = 144;

static inline void q4k_scale_min(uint j,
                                  device const uchar* scales,
                                  thread uchar& sc,
                                  thread uchar& m)
{
    if (j < 4) {
        sc = scales[j]   & 0x3F;
        m  = scales[j+4] & 0x3F;
    } else {
        sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4);
        m  = (scales[j+4] >>  4)  | ((scales[j]   >> 6) << 4);
    }
}

kernel void dequant_q4_k_m_to_f32(
    device const uchar* W      [[buffer(0)]],
    device float*       out    [[buffer(1)]],
    constant uint&      nElem  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= nElem) return;
    uint blockIdx = gid / Q4_K_BLOCK;
    uint inner    = gid % Q4_K_BLOCK;
    device const uchar* blk = W + blockIdx * Q4_K_BYTES;

    ushort dBits   = ushort(blk[0]) | (ushort(blk[1]) << 8);
    ushort dmBits  = ushort(blk[2]) | (ushort(blk[3]) << 8);
    float  d       = float(as_type<half>(dBits));
    float  dmin    = float(as_type<half>(dmBits));
    device const uchar* scales = blk + 4;        // 12 bytes
    device const uchar* qs     = blk + 16;       // 128 bytes

    // Resolve which sub-block this element belongs to.
    uint j_pair = inner / 64;             // 0..3  (outer iteration)
    uint within = inner % 64;             // 0..63
    bool isHi   = within >= 32;           // hi nibble → sub-block (2*j_pair+1)
    uint l      = within % 32;            // 0..31 inside the 32-element sub-block
    uint subIdx = 2 * j_pair + (isHi ? 1u : 0u);

    uchar sc, m;
    q4k_scale_min(subIdx, scales, sc, m);
    float scaleSub = d * float(sc);
    float minSub   = dmin * float(m);

    uint  qsByte = qs[32 * j_pair + l];
    uint  nibble = isHi ? (qsByte >> 4) : (qsByte & 0xF);
    out[gid] = scaleSub * float(nibble) - minSub;
}

// --------------------------------------------------------------------
// BF16 output variants. Same arithmetic as the F32 kernels above —
// only the store dtype differs. Halves the resident weight memory
// (BF16 keeps full F32 dynamic range, just lower mantissa precision)
// at no measurable accuracy hit in practice for these GGUF quants.
// --------------------------------------------------------------------

kernel void dequant_q8_0_to_bf16(
    device const uchar* W      [[buffer(0)]],
    device bfloat*      out    [[buffer(1)]],
    constant uint&      nElem  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= nElem) return;
    uint blockIdx = gid / Q8_0_BLOCK;
    uint inner    = gid % Q8_0_BLOCK;
    device const uchar* blk = W + blockIdx * Q8_0_BYTES;
    ushort sBits = ushort(blk[0]) | (ushort(blk[1]) << 8);
    float  scale = float(as_type<half>(sBits));
    char   q     = as_type<char>(blk[2 + inner]);
    out[gid] = bfloat(float(q) * scale);
}

kernel void dequant_q4_0_to_bf16(
    device const uchar* W      [[buffer(0)]],
    device bfloat*      out    [[buffer(1)]],
    constant uint&      nElem  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= nElem) return;
    uint blockIdx = gid / Q4_0_BLOCK;
    uint inner    = gid % Q4_0_BLOCK;
    device const uchar* blk = W + blockIdx * Q4_0_BYTES;
    ushort sBits = ushort(blk[0]) | (ushort(blk[1]) << 8);
    float  scale = float(as_type<half>(sBits));
    uint byteIdx = (inner < 16) ? inner : (inner - 16);
    uchar pack   = blk[2 + byteIdx];
    uint  nibble = (inner < 16) ? (pack & 0xF) : (pack >> 4);
    int   signed_q = int(nibble) - 8;
    out[gid] = bfloat(float(signed_q) * scale);
}

kernel void dequant_q4_k_m_to_bf16(
    device const uchar* W      [[buffer(0)]],
    device bfloat*      out    [[buffer(1)]],
    constant uint&      nElem  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= nElem) return;
    uint blockIdx = gid / Q4_K_BLOCK;
    uint inner    = gid % Q4_K_BLOCK;
    device const uchar* blk = W + blockIdx * Q4_K_BYTES;
    ushort dBits  = ushort(blk[0]) | (ushort(blk[1]) << 8);
    ushort dmBits = ushort(blk[2]) | (ushort(blk[3]) << 8);
    float  d     = float(as_type<half>(dBits));
    float  dmin  = float(as_type<half>(dmBits));
    device const uchar* scales = blk + 4;
    device const uchar* qs     = blk + 16;
    uint j_pair = inner / 64;
    uint within = inner % 64;
    bool isHi   = within >= 32;
    uint l      = within % 32;
    uint subIdx = 2 * j_pair + (isHi ? 1u : 0u);
    uchar sc, m;
    q4k_scale_min(subIdx, scales, sc, m);
    float scaleSub = d * float(sc);
    float minSub   = dmin * float(m);
    uint  qsByte = qs[32 * j_pair + l];
    uint  nibble = isHi ? (qsByte >> 4) : (qsByte & 0xF);
    out[gid] = bfloat(scaleSub * float(nibble) - minSub);
}

// --------------------------------------------------------------------
// Q5_K — 5-bit per element, 256-element super-block (176 bytes).
// Layout:
//   bytes [0..1]:     F16 super-scale d
//   bytes [2..3]:     F16 super-min dmin
//   bytes [4..15]:    12 bytes packed 6-bit scales+mins (same get_scale_min_k4 layout as Q4_K)
//   bytes [16..47]:   32 bytes of high-bit qh (one bit per quant; 256 bits)
//   bytes [48..175]:  128 bytes of low 4-bit qs (same Q4_K-style packing)
// For sub-block j (= 2*j_pair + is_hi), element l:
//   low nibble       = (is_hi ? qs[32*j_pair+l] >> 4 : qs[32*j_pair+l] & 0xF)
//   high bit         = (qh[l + 32*l_block] >> (sub_idx_within_pair)) & 1
//     — concretely: high bit for sub-block `j` at element `l` is
//       `(qh[l] >> j) & 1` (j in 0..7), packing all 8 sub-blocks of
//       the same `l` into the same byte.
//   5-bit value      = low_nibble | (high_bit << 4)        (0..31)
//   value            = (d * sc_j) * 5bit_value - (dmin * m_j)
// --------------------------------------------------------------------

constant uint Q5_K_BLOCK = 256;
constant uint Q5_K_BYTES = 176;

kernel void dequant_q5_k_to_f32(
    device const uchar* W      [[buffer(0)]],
    device float*       out    [[buffer(1)]],
    constant uint&      nElem  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= nElem) return;
    uint blockIdx = gid / Q5_K_BLOCK;
    uint inner    = gid % Q5_K_BLOCK;
    device const uchar* blk = W + blockIdx * Q5_K_BYTES;
    ushort dBits   = ushort(blk[0]) | (ushort(blk[1]) << 8);
    ushort dmBits  = ushort(blk[2]) | (ushort(blk[3]) << 8);
    float  d       = float(as_type<half>(dBits));
    float  dmin    = float(as_type<half>(dmBits));
    device const uchar* scales = blk + 4;
    device const uchar* qh     = blk + 16;      // 32 bytes
    device const uchar* qs     = blk + 48;      // 128 bytes

    uint j_pair = inner / 64;
    uint within = inner % 64;
    bool isHi   = within >= 32;
    uint l      = within % 32;
    uint subIdx = 2 * j_pair + (isHi ? 1u : 0u);

    uchar sc, m;
    q4k_scale_min(subIdx, scales, sc, m);
    float scaleSub = d * float(sc);
    float minSub   = dmin * float(m);

    uint qsByte    = qs[32 * j_pair + l];
    uint lowNibble = isHi ? (qsByte >> 4) : (qsByte & 0xF);
    uint highBit   = (uint(qh[l]) >> subIdx) & 1u;
    uint fiveBit   = lowNibble | (highBit << 4);
    out[gid] = scaleSub * float(fiveBit) - minSub;
}

// --------------------------------------------------------------------
// Q6_K — 6-bit per element, 256-element super-block (210 bytes).
// Layout (offsets reverse-engineered from `ggml-quants.h`):
//   bytes [0..127]:   128 bytes ql — lower 4 bits of each quant
//   bytes [128..191]: 64 bytes qh  — upper 2 bits, packed (4 quants per byte)
//   bytes [192..207]: 16 bytes scales — int8 per-16-element scale
//   bytes [208..209]: F16 super-scale d
//
// llama.cpp processes Q6_K in 128-element halves (n = 0, 128). Within
// each half, elements come in 4 groups of 32, with this addressing:
//   element offsets relative to half start: l + 0, l + 32, l + 64, l + 96
//   (for l = 0..31)
// The low 4 bits live in ql[l] (for groups 0+2) and ql[l+32] (groups 1+3),
// with lo nibble feeding groups 0,1 and hi nibble feeding groups 2,3.
// The high 2 bits come from qh[l] bit-groups (qh[l]>>0&3, >>2&3, >>4&3, >>6&3).
// Each 16-element strip uses one int8 scale entry; per half there are 8
// strips (`sc + 0…7`).
// signed_q = ((4_lo | (2_hi << 4)) − 32) — center the 6-bit value at zero.
// value    = d * sc * signed_q.
// --------------------------------------------------------------------

constant uint Q6_K_BLOCK = 256;
constant uint Q6_K_BYTES = 210;

kernel void dequant_q6_k_to_f32(
    device const uchar* W      [[buffer(0)]],
    device float*       out    [[buffer(1)]],
    constant uint&      nElem  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= nElem) return;
    uint blockIdx = gid / Q6_K_BLOCK;
    uint inner    = gid % Q6_K_BLOCK;
    device const uchar* blk = W + blockIdx * Q6_K_BYTES;

    ushort dBits = ushort(blk[208]) | (ushort(blk[209]) << 8);
    float  d     = float(as_type<half>(dBits));
    device const uchar* qlAll    = blk;                  // 128
    device const uchar* qhAll    = blk + 128;            // 64
    device const char*  scAll    = (device const char*)(blk + 192);  // signed 16

    // Which 128-element half? n_half in {0, 1}.
    uint nHalf = inner / 128;
    uint local = inner % 128;       // 0..127 inside the half
    // Local layout: groups of 32 at offsets 0/32/64/96.
    uint groupIdx = local / 32;     // 0..3
    uint l        = local % 32;     // 0..31

    device const uchar* ql = qlAll + 64 * nHalf;   // 64 bytes per half
    device const uchar* qh = qhAll + 32 * nHalf;   // 32 bytes per half
    device const char*  sc = scAll +  8 * nHalf;   //  8 scales per half

    // Pick the right pair of ql byte + qh bit-group depending on the
    // group index, mirroring the C reference loop.
    uchar qlByte;
    uint  qhBits;
    if (groupIdx == 0) {
        qlByte = ql[l];        // lo nibble feeds offset l + 0
        qhBits = (uint(qh[l]) >> 0u) & 3u;
    } else if (groupIdx == 1) {
        qlByte = ql[l + 32];   // lo nibble feeds offset l + 32
        qhBits = (uint(qh[l]) >> 2u) & 3u;
    } else if (groupIdx == 2) {
        qlByte = ql[l];        // hi nibble feeds offset l + 64
        qhBits = (uint(qh[l]) >> 4u) & 3u;
    } else /* groupIdx == 3 */ {
        qlByte = ql[l + 32];   // hi nibble feeds offset l + 96
        qhBits = (uint(qh[l]) >> 6u) & 3u;
    }
    uint qlBits = (groupIdx >= 2) ? (uint(qlByte) >> 4) : (uint(qlByte) & 0xFu);
    int  sixBit = int(qlBits | (qhBits << 4)) - 32;

    // Scale: 8 scales per half, indexed by (l/16) + 2*groupIdx.
    int scIdx = int(l / 16) + 2 * int(groupIdx);
    float scale = d * float(sc[scIdx]);

    out[gid] = scale * float(sixBit);
}
