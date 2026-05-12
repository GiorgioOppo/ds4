#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// simdgroup_matrix variants — Apple Silicon tensor-core path.
// ============================================================================
// One threadgroup = one SIMD group (32 threads on Apple GPUs). Each threadgroup
// produces a 32×32 block of C as a 4×4 grid of simdgroup_matrix<8, 8>
// accumulators kept in registers across the K reduction.
//
// Requirements (caller's responsibility):
//   M % 32 == 0, N % 32 == 0, K % 8 == 0.
// Smaller / unaligned dims must use the legacy gemm_*_to_f32 kernels above.
// Linear.swift picks the right one based on (M, N, K).
//
// A is [M, K] row-major, B is [N, K] row-major (rows are output features).
// B is logically transposed during the load via simdgroup_load(transpose=true)
// so each fragment lands as [K_tile, N_tile] for the matmul.

constant uint SG_TM = 32;
constant uint SG_TN = 32;
constant uint SG_TK = 8;

kernel void gemm_bf16_to_f32_sg(
    device const bfloat* A     [[buffer(0)]],
    device const bfloat* B     [[buffer(1)]],
    device float*        C     [[buffer(2)]],
    constant uint3&      dims  [[buffer(3)]],   // (M, N, K)
    uint2 gpos [[threadgroup_position_in_grid]]
) {
    // M is implicit in the grid dispatch (gpos.y) so we don't read it here.
    uint N = dims.y, K = dims.z;
    uint row0 = gpos.y * SG_TM;
    uint col0 = gpos.x * SG_TN;

    simdgroup_matrix<float, 8, 8> Cmat[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            Cmat[i][j] = simdgroup_matrix<float, 8, 8>(0);

    simdgroup_matrix<bfloat, 8, 8> Atile[4];
    simdgroup_matrix<bfloat, 8, 8> Btile[4];

    for (uint k0 = 0; k0 < K; k0 += SG_TK) {
        for (uint i = 0; i < 4; i++) {
            simdgroup_load(Atile[i],
                           A + (row0 + i * 8) * K + k0, K,
                           ulong2(0, 0), false);
            // B is [N, K]; we load 8x8 tile starting at row=col0+j*8, col=k0
            // and transpose to get [K_tile, N_tile] orientation.
            simdgroup_load(Btile[i],
                           B + (col0 + i * 8) * K + k0, K,
                           ulong2(0, 0), true);
        }
        for (uint i = 0; i < 4; i++)
            for (uint j = 0; j < 4; j++)
                simdgroup_multiply_accumulate(Cmat[i][j], Atile[i], Btile[j], Cmat[i][j]);
    }

    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            simdgroup_store(Cmat[i][j],
                            C + (row0 + i * 8) * N + col0 + j * 8, N,
                            ulong2(0, 0), false);
}

// F32 activations, BF16 weights. simdgroup_matrix is bf16-typed, so we
// reinterpret incoming f32 by reading it through a thread-local cast.
// Apple's runtime offers a direct path: the multiply_accumulate accepts
// mixed-precision inputs when one operand is f16/bf16. We promote f32 → bf16
// at load time. This loses 8 bits of mantissa for the activation; the
// reference also runs activations through act_quant for the FP8/FP4 paths,
// so a bf16-equivalent precision is acceptable on the dense path.
kernel void gemm_f32_bf16_to_f32_sg(
    device const float*  A     [[buffer(0)]],
    device const bfloat* B     [[buffer(1)]],
    device float*        C     [[buffer(2)]],
    constant uint3&      dims  [[buffer(3)]],
    uint  lid  [[thread_index_in_simdgroup]],
    uint2 gpos [[threadgroup_position_in_grid]]
) {
    // M is implicit in the grid dispatch (gpos.y) so we don't read it here.
    uint N = dims.y, K = dims.z;
    uint row0 = gpos.y * SG_TM;
    uint col0 = gpos.x * SG_TN;

    threadgroup bfloat Acast[SG_TM][SG_TK];

    simdgroup_matrix<float, 8, 8> Cmat[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            Cmat[i][j] = simdgroup_matrix<float, 8, 8>(0);

    simdgroup_matrix<bfloat, 8, 8> Atile[4];
    simdgroup_matrix<bfloat, 8, 8> Btile[4];

    for (uint k0 = 0; k0 < K; k0 += SG_TK) {
        // Stage the 32×8 A slab as bfloat in threadgroup memory.
        // 32 threads, 32×8 = 256 entries → 8 entries per thread.
        for (uint r = 0; r < 8; r++) {
            uint idx = lid * 8 + r;
            uint mi = idx / SG_TK;
            uint ki = idx % SG_TK;
            float v = A[(row0 + mi) * K + k0 + ki];
            Acast[mi][ki] = bfloat(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < 4; i++) {
            simdgroup_load(Atile[i],
                           &Acast[i * 8][0], SG_TK,
                           ulong2(0, 0), false);
            simdgroup_load(Btile[i],
                           B + (col0 + i * 8) * K + k0, K,
                           ulong2(0, 0), true);
        }
        for (uint i = 0; i < 4; i++)
            for (uint j = 0; j < 4; j++)
                simdgroup_multiply_accumulate(Cmat[i][j], Atile[i], Btile[j], Cmat[i][j]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            simdgroup_store(Cmat[i][j],
                            C + (row0 + i * 8) * N + col0 + j * 8, N,
                            ulong2(0, 0), false);
}

// ============================================================================
// Legacy tiled GEMMs — used for unaligned M/N/K and for the f32×f32 path.
// ============================================================================

// BF16 weight × BF16 activation → F32 GEMM.
//
// Mirrors the F.linear path in
// Reference/inference/model.py (the unquantized fallback in `linear()`,
// model.py:108-120). Used by Compressor's wkv/wgate (FP32 in checkpoint,
// cast to BF16 here) and by any other dense Linear layers.
//
// Layout:
//   A: [M, K] bf16        — activations
//   B: [N, K] bf16        — weights, rows are output features
//   C: [M, N] f32         — output (FP32 to match the reference's accum dtype)
//
// Naive tiled implementation. One thread per output cell. Threadgroups are
// 16×16 tiles of C and load 16×16 tiles of A and B^T into threadgroup memory
// for reuse across the K reduction. Slow but correct; can be replaced with
// simdgroup_matrix later.

constant uint TILE = 16;

kernel void gemm_bf16_to_f32(
    device const bfloat* A     [[buffer(0)]],
    device const bfloat* B     [[buffer(1)]],
    device float*        C     [[buffer(2)]],
    constant uint3&      dims  [[buffer(3)]],   // (M, N, K)
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 gpos [[threadgroup_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gpos.y * TILE + lid.y;
    uint col = gpos.x * TILE + lid.x;

    threadgroup bfloat As[TILE][TILE];
    threadgroup bfloat Bs[TILE][TILE];

    float acc = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += TILE) {
        uint kx = k0 + lid.x;
        uint ky = k0 + lid.y;
        As[lid.y][lid.x] = (row < M && kx < K) ? A[row * K + kx] : bfloat(0);
        // B is [N, K] so column `col` of C uses row `col` of B at offset ky.
        Bs[lid.y][lid.x] = (col < N && ky < K) ? B[col * K + ky] : bfloat(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            acc += float(As[lid.y][k]) * float(Bs[k][lid.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// F32 input variant — convenience for tests / activations that haven't been
// down-cast to BF16 yet. Same math, A is f32.
kernel void gemm_f32_bf16_to_f32(
    device const float*  A     [[buffer(0)]],
    device const bfloat* B     [[buffer(1)]],
    device float*        C     [[buffer(2)]],
    constant uint3&      dims  [[buffer(3)]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 gpos [[threadgroup_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gpos.y * TILE + lid.y;
    uint col = gpos.x * TILE + lid.x;

    threadgroup float  As[TILE][TILE];
    threadgroup bfloat Bs[TILE][TILE];

    float acc = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += TILE) {
        uint kx = k0 + lid.x;
        uint ky = k0 + lid.y;
        As[lid.y][lid.x] = (row < M && kx < K) ? A[row * K + kx] : 0.0f;
        Bs[lid.y][lid.x] = (col < N && ky < K) ? B[col * K + ky] : bfloat(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            acc += As[lid.y][k] * float(Bs[k][lid.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// F32 × F32 — matches the reference's compressor weights which the upstream
// promotes from BF16 to FP32 at load time for numerical stability.
kernel void gemm_f32_to_f32(
    device const float*  A     [[buffer(0)]],
    device const float*  B     [[buffer(1)]],
    device float*        C     [[buffer(2)]],
    constant uint3&      dims  [[buffer(3)]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 gpos [[threadgroup_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gpos.y * TILE + lid.y;
    uint col = gpos.x * TILE + lid.x;

    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];

    float acc = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += TILE) {
        uint kx = k0 + lid.x;
        uint ky = k0 + lid.y;
        As[lid.y][lid.x] = (row < M && kx < K) ? A[row * K + kx] : 0.0f;
        Bs[lid.y][lid.x] = (col < N && ky < K) ? B[col * K + ky] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            acc += As[lid.y][k] * Bs[k][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
