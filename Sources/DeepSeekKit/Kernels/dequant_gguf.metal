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
