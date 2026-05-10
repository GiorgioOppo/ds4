#include <metal_stdlib>
using namespace metal;

// Top-K along the last axis. Returns both values and (Int32) indices.
// One thread per row, with the top-K running heap held in registers.
//
// Used by Indexer's `score.topk(min(self.index_topk, end_pos // ratio), dim=-1)`
// from Reference/inference/model.py:427.
//
// Limitation: K must be ≤ MAX_K (32). The Swift wrapper enforces this with
// a precondition. For the V4 prefill case where index_topk = 512, this
// kernel needs replacing with a tile/merge approach; until then decode
// (M=1) and small-K paths use it directly.

kernel void topk_f32(
    device const float* x        [[buffer(0)]],   // [N, V]
    device float*       outVals  [[buffer(1)]],   // [N, K]
    device int*         outIdxs  [[buffer(2)]],   // [N, K]
    constant uint3&     dims     [[buffer(3)]],   // (N, V, K)
    uint                row      [[thread_position_in_grid]]
) {
    uint N = dims.x, V = dims.y, K = dims.z;
    if (row >= N) return;

    constexpr uint MAX_K = 32;
    float bestV[MAX_K];
    int   bestI[MAX_K];
    for (uint k = 0; k < K; k++) { bestV[k] = -INFINITY; bestI[k] = -1; }

    device const float* rowPtr = x + row * V;
    for (uint v = 0; v < V; v++) {
        float val = rowPtr[v];
        // find smallest entry currently in the top-K
        uint  smallI = 0;
        float smallV = bestV[0];
        for (uint k = 1; k < K; k++) {
            if (bestV[k] < smallV) { smallV = bestV[k]; smallI = k; }
        }
        if (val > smallV) {
            bestV[smallI] = val;
            bestI[smallI] = (int)v;
        }
    }

    // Sort the K results in descending order so the caller can read them
    // in rank order (insertion sort, K is tiny).
    for (uint k = 1; k < K; k++) {
        float v = bestV[k]; int i = bestI[k];
        uint j = k;
        while (j > 0u && bestV[j - 1u] < v) {
            bestV[j] = bestV[j - 1u];
            bestI[j] = bestI[j - 1u];
            j--;
        }
        bestV[j] = v;
        bestI[j] = i;
    }

    for (uint k = 0; k < K; k++) {
        outVals[row * K + k] = bestV[k];
        outIdxs[row * K + k] = bestI[k];
    }
}
