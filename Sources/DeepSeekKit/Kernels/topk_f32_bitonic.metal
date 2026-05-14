#include <metal_stdlib>
using namespace metal;

// Top-K via in-place bitonic sort over threadgroup-shared memory.
//
// Used when the small register-resident kernel (topk_f32) can't carry
// K because K > 32. One threadgroup per row of [N, V]; each thread in
// the group handles V_padded / T elements per stage. Sort is
// descending, so the top-K elements end up at positions 0..K-1.
//
// Limits:
//   - V <= MAX_V (4096). The Swift wrapper enforces this; rows beyond
//     this width need a tiled/multipass approach we haven't written.
//   - Threadgroup size <= 1024 (Metal hard limit). For V_padded > 1024
//     each thread does V_padded / 1024 compare-swaps per inner step.
//
// Shared memory: MAX_V * (4 + 4) bytes = 32 KiB. Apple-silicon GPUs
// expose >= 32 KiB threadgroup memory; if a future config raises
// MAX_V past 4096 the host-side allocator needs to query
// `maxThreadgroupMemoryLength` first.

constant constexpr uint MAX_V = 4096;

kernel void topk_f32_bitonic(
    device const float* x        [[buffer(0)]],     // [N, V]
    device float*       outVals  [[buffer(1)]],     // [N, K]
    device int*         outIdxs  [[buffer(2)]],     // [N, K]
    constant uint3&     dims     [[buffer(3)]],     // (N, V, K)
    uint3               tgPos    [[threadgroup_position_in_grid]],
    uint3               lid      [[thread_position_in_threadgroup]],
    uint3               tgSize   [[threads_per_threadgroup]]
) {
    uint N = dims.x, V = dims.y, K = dims.z;
    uint row = tgPos.z;
    if (row >= N) return;

    threadgroup float sVals[MAX_V];
    threadgroup int   sIdxs[MAX_V];

    // Pad V up to the next power of two; bitonic sort needs that.
    uint Vp = 1;
    while (Vp < V) Vp <<= 1;
    // Caller guarantees V <= MAX_V, so Vp <= MAX_V also.

    uint T = tgSize.x;
    uint tid = lid.x;

    // 1) Load row into shared memory, padding with -INF (so the
    //    padding never bubbles up into the top-K window).
    for (uint i = tid; i < Vp; i += T) {
        if (i < V) {
            sVals[i] = x[row * V + i];
            sIdxs[i] = (int)i;
        } else {
            sVals[i] = -INFINITY;
            sIdxs[i] = -1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) Bitonic sort, descending.
    //    Outer subsequence size k = 2, 4, 8, ..., Vp.
    //    Inner bit distance     j = k/2, k/4, ..., 1.
    //    At each (k, j) step every (i, i^j) pair with i < i^j compares
    //    once. The "descending half" flag (i & k) flips the sense so
    //    the final pass leaves the array sorted in descending order.
    for (uint k = 2; k <= Vp; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = tid; i < Vp; i += T) {
                uint ixj = i ^ j;
                if (ixj > i) {
                    bool descendingHalf = (i & k) == 0;
                    float a = sVals[i];
                    float b = sVals[ixj];
                    bool swapIt = descendingHalf ? (a < b) : (a > b);
                    if (swapIt) {
                        sVals[i]   = b;        sVals[ixj] = a;
                        int ai = sIdxs[i];     int bi = sIdxs[ixj];
                        sIdxs[i]   = bi;       sIdxs[ixj] = ai;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // 3) Emit the first K (the row is now sorted descending).
    for (uint i = tid; i < K; i += T) {
        outVals[row * K + i] = sVals[i];
        outIdxs[row * K + i] = sIdxs[i];
    }
}
