#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// FP8-E4M3FN dequant — duplicated from act_quant.metal (each .metal is its own
// compilation unit). See dtype semantics in Quantization.swift.
inline float deq_e4m3(uchar b) {
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

// E8M0 (unsigned-exponent-only) dequant: value = 2^(b - 127). Encoded as a
// Float32 whose mantissa is zero and whose biased exponent equals `b`, i.e.
// the byte goes straight into the 8 exponent bits of an IEEE 754 single-
// precision word. Mirrors `deqE8M0` in DTypePacking.swift.
inline float deq_e8m0(uchar b) {
    if (b == 0xFFu) return NAN;
    return as_type<float>(((uint)b) << 23);
}

// fp8_gemm — FP8 × FP8 matrix multiply with per-128 block scaling.
//
// Mirrors `fp8_gemm_kernel` from
// Reference/inference/kernel.py lines 203–273.
//
// Layout:
//   A:        [M, K] fp8_e4m3
//   A_scale:  [M, K/128] f32   — produced by ActQuant (dynamic, runtime)
//   B:        [N, K] fp8_e4m3
//   B_scale:  [N/128, K/128] uchar — UE8M0 weight scale from disk
//   C:        [M, N] f32 output
//
// Naive tiled implementation: one thread per output cell, scalar loop over
// K-blocks of 128 elements. Apple Silicon has no native FP8 GEMM so each
// FP8 byte is dequantized through `deq_e4m3` in-shader before multiplying.
//
// Weight scales come from disk as UE8M0 (1 byte each — `scale_fmt: ue8m0`
// in DeepSeek-V4-HF). Reading them as `device const float*` would mis-
// interpret 4 consecutive bytes as one f32, producing silent garbage and
// out-of-bounds reads near the end of the buffer. Dequant inline via
// `deq_e8m0`.

constant uint BLOCK_K = 128;
constant uint BLOCK_N_FP8 = 128;

kernel void gemm_fp8_to_f32(
    device const uchar*  A      [[buffer(0)]],
    device const float*  A_sc   [[buffer(1)]],
    device const uchar*  B      [[buffer(2)]],
    device const uchar*  B_sc   [[buffer(3)]],
    device float*        C      [[buffer(4)]],
    constant uint3&      dims   [[buffer(5)]],   // (M, N, K)
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gid.y, col = gid.x;
    if (row >= M || col >= N) return;

    uint blocksK = K / BLOCK_K;
    float acc = 0.0f;

    for (uint kb = 0; kb < blocksK; kb++) {
        float a_scale = A_sc[row * blocksK + kb];
        float b_scale = deq_e8m0(B_sc[(col / BLOCK_N_FP8) * blocksK + kb]);

        float block_acc = 0.0f;
        uint k0 = kb * BLOCK_K;
        for (uint k = 0; k < BLOCK_K; k++) {
            float a = deq_e4m3(A[row * K + k0 + k]);
            float b = deq_e4m3(B[col * K + k0 + k]);
            block_acc += a * b;
        }
        acc += block_acc * a_scale * b_scale;
    }
    C[row * N + col] = acc;
}

// ============================================================================
// gemm_fp8_to_f32_sg — simdgroup_matrix variant of gemm_fp8_to_f32.
// ============================================================================
//
// One threadgroup = one SIMD group (32 threads). Each threadgroup produces a
// 32×32 block of C as a 4×4 grid of `simdgroup_matrix<float, 8, 8>`
// accumulators kept in registers across the K reduction. A and B are
// FP8-E4M3FN on disk / in device memory; we dequant to bfloat in threadgroup
// memory (and bake in the per-block scales while we're at it) before issuing
// the simdgroup matmul.
//
// Requirements (caller's `canUseSG` plus the FP8-specific K alignment):
//   M % 32 == 0, N % 32 == 0, K % 128 == 0.
// The K%128 constraint is tighter than the bf16 kernel's K%8 because the FP8
// weight scales (B_sc) live on a 128×128 block grid — anything not a multiple
// of 128 lands the SG matmul mid-block and forces a partial-block scale
// branch we don't currently implement. Linear.swift falls back to the
// `gemm_fp8_to_f32` scalar kernel when K%128 != 0.
//
// SG_TN=32 divides BLOCK_N_FP8=128 so every 32-col output tile is guaranteed
// to lie entirely within a single N-side weight-scale block — i.e., the
// `b_scale` value is uniform across the 32 output columns within one
// threadgroup. We exploit that to read it ONCE per K-block (instead of
// once per col) and broadcast it through threadgroup memory.

constant uint SG_TM_FP8 = 32;
constant uint SG_TN_FP8 = 32;
constant uint SG_TK_FP8 = 8;

kernel void gemm_fp8_to_f32_sg(
    device const uchar*  A      [[buffer(0)]],
    device const float*  A_sc   [[buffer(1)]],
    device const uchar*  B      [[buffer(2)]],
    device const uchar*  B_sc   [[buffer(3)]],
    device float*        C      [[buffer(4)]],
    constant uint3&      dims   [[buffer(5)]],   // (M, N, K)
    uint  lid  [[thread_index_in_simdgroup]],
    uint2 gpos [[threadgroup_position_in_grid]]
) {
    // M is implicit in the grid dispatch (gpos.y) so it isn't read here.
    uint N = dims.y, K = dims.z;
    uint row0 = gpos.y * SG_TM_FP8;
    uint col0 = gpos.x * SG_TN_FP8;
    uint blocksK = K / BLOCK_K;
    uint n_block = col0 / BLOCK_N_FP8;

    // 4×4 grid of 8×8 F32 accumulators. Stays in registers across the
    // entire K reduction (the matmul's "natural" precision on Apple
    // Silicon simdgroups; the unit promotes the bfloat inputs to FP32
    // internally for the FMA).
    simdgroup_matrix<float, 8, 8> Cmat[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            Cmat[i][j] = simdgroup_matrix<float, 8, 8>(0);

    // Threadgroup staging for the dequantized + pre-scaled bfloat tiles.
    // 32 rows × 8 K-elements each = 256 entries per tile, 512 bytes per
    // tile, plus 32+1 floats for the scales = ~1.2 KB total. Well under
    // the Apple Silicon 32 KB threadgroup memory limit.
    threadgroup bfloat As[SG_TM_FP8][SG_TK_FP8];
    threadgroup bfloat Bs[SG_TN_FP8][SG_TK_FP8];
    threadgroup float  a_scale_buf[SG_TM_FP8];
    threadgroup float  b_scale_buf;

    simdgroup_matrix<bfloat, 8, 8> Atile[4];
    simdgroup_matrix<bfloat, 8, 8> Btile[4];

    // Iterate over K in steps of BLOCK_K (=128). One scale-set per step.
    for (uint kb = 0; kb < blocksK; kb++) {
        // Stage scales for this K-block. 32 SIMD lanes handle the 32
        // row-side a_scales in parallel; lane 0 also dequantizes the
        // single UE8M0 b_scale and writes it to the broadcast slot.
        a_scale_buf[lid] = A_sc[(row0 + lid) * blocksK + kb];
        if (lid == 0) {
            b_scale_buf = deq_e8m0(B_sc[n_block * blocksK + kb]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float a_s = a_scale_buf[lid];
        float b_s = b_scale_buf;

        uint k0_block = kb * BLOCK_K;

        // Inner reduction: 16 sub-steps of 8 K-elements each. Each
        // sub-step stages a fresh 32×8 As/Bs tile and runs one
        // 4×4 grid of simdgroup matmuls.
        for (uint ks = 0; ks < BLOCK_K; ks += SG_TK_FP8) {
            uint k_global = k0_block + ks;

            // Cooperative dequant + scale bake-in. Thread `lid`
            // handles its row of A (row=row0+lid) and its col of B
            // (col=col0+lid). Each writes SG_TK_FP8=8 elements.
            //
            // Memory pattern is strided across simdgroup lanes (each
            // thread reads from a different row, K apart), which is
            // not coalesced — matches the gemm_f32_bf16_to_f32_sg
            // staging pattern (the bf16 kernel's bottleneck is the
            // matmul, not the load). If FP8 turns out to be load-bound
            // a follow-up can restructure to lane-coalesced 32-wide
            // reads with a transpose at use time.
            for (uint k = 0; k < SG_TK_FP8; k++) {
                uchar abyte = A[(row0 + lid) * K + k_global + k];
                As[lid][k] = bfloat(deq_e4m3(abyte) * a_s);
            }
            for (uint k = 0; k < SG_TK_FP8; k++) {
                uchar bbyte = B[(col0 + lid) * K + k_global + k];
                Bs[lid][k] = bfloat(deq_e4m3(bbyte) * b_s);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Load fragments + run the 4×4 grid of 8×8 matmuls.
            // Btile uses transpose=true: Bs is staged as [N_tile,
            // K_tile] (row-major over output cols × K), the matmul
            // wants [K_tile, N_tile], so the loader does the swap.
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(Atile[i],
                               &As[i * 8][0], SG_TK_FP8,
                               ulong2(0, 0), false);
                simdgroup_load(Btile[i],
                               &Bs[i * 8][0], SG_TK_FP8,
                               ulong2(0, 0), true);
            }
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(Cmat[i][j],
                                                    Atile[i], Btile[j],
                                                    Cmat[i][j]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Store the final 32×32 F32 output. C is the host-allocated [M, N]
    // f32 buffer; simdgroup_store writes one 8×8 tile per call into
    // (row0 + i*8, col0 + j*8).
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            simdgroup_store(Cmat[i][j],
                            C + (row0 + i * 8) * N + col0 + j * 8, N,
                            ulong2(0, 0), false);
}
