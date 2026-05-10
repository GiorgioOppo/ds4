#include <metal_stdlib>
using namespace metal;

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
