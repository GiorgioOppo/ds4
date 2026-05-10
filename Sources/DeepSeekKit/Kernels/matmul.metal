#include <metal_stdlib>
using namespace metal;

// Naive tiled GEMM: y = x @ wT  where x:[M,K], w:[N,K] (rows are output features), y:[M,N]
// Suitable for prefill. For incremental decoding (M=1) use matvec_q4 below.
constant uint TILE = 16;

kernel void matmul_f32(
    device const float* x   [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    device float*       y   [[buffer(2)]],
    constant uint3&     dims [[buffer(3)]],   // (M, N, K)
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 gpos [[threadgroup_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gpos.y * TILE + lid.y;
    uint col = gpos.x * TILE + lid.x;

    threadgroup float xs[TILE][TILE];
    threadgroup float ws[TILE][TILE];

    float acc = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += TILE) {
        uint kx = k0 + lid.x;
        uint ky = k0 + lid.y;
        xs[lid.y][lid.x] = (row < M && kx < K) ? x[row * K + kx] : 0.0f;
        ws[lid.y][lid.x] = (col < N && ky < K) ? w[col * K + ky] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; k++) {
            acc += xs[lid.y][k] * ws[lid.x][k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) y[row * N + col] = acc;
}

// int4 block-quantized matvec: y[n] = sum_k x[k] * dequant(w[n,k])
// Block layout per row of w: [scale(f16), zero(f16), nibbles(blockSize/2 bytes)] repeated K/blockSize times.
// blockSize is fixed at 32 to match Quant.blockSize on the Swift side.
constant uint Q4_BLOCK = 32;

kernel void matvec_q4(
    device const float*  x       [[buffer(0)]],
    device const uchar*  w       [[buffer(1)]],   // packed
    device float*        y       [[buffer(2)]],
    constant uint2&      dims    [[buffer(3)]],   // (N, K)
    uint                 n       [[thread_position_in_grid]]
) {
    uint N = dims.x, K = dims.y;
    if (n >= N) return;

    uint blocksPerRow = K / Q4_BLOCK;
    uint bytesPerBlock = 2 + 2 + Q4_BLOCK / 2;        // scale + zero + nibbles
    device const uchar* row = w + n * blocksPerRow * bytesPerBlock;

    float acc = 0.0f;
    for (uint b = 0; b < blocksPerRow; b++) {
        device const uchar* blk = row + b * bytesPerBlock;
        ushort sH = (ushort)blk[0] | ((ushort)blk[1] << 8);
        ushort zH = (ushort)blk[2] | ((ushort)blk[3] << 8);
        float scale = float(as_type<half>(sH));
        float zero  = float(as_type<half>(zH));
        device const uchar* nibs = blk + 4;
        uint kbase = b * Q4_BLOCK;
        for (uint i = 0; i < Q4_BLOCK / 2; i++) {
            uchar pair = nibs[i];
            float a = float(pair & 0x0F) * scale + zero;
            float bv = float(pair >> 4)   * scale + zero;
            acc += x[kbase + 2*i]     * a;
            acc += x[kbase + 2*i + 1] * bv;
        }
    }
    y[n] = acc;
}
