#include <metal_stdlib>
using namespace metal;

// hc_split_sinkhorn — port of `hc_split_sinkhorn_kernel`
// in Reference/inference/kernel.py lines 371–427.
//
// Inputs (n = b * s, hc = hcMult, mix_hc = (2 + hc) * hc):
//   mixes:    [n, mix_hc]  f32
//   hcScale:  [3]          f32
//   hcBase:   [mix_hc]     f32
//
// Outputs:
//   pre:   [n, hc]       pre[i,j]  = sigmoid(mixes[i,j]      * scale[0] + base[j])     + eps
//   post:  [n, hc]       post[i,j] = 2 * sigmoid(mixes[i,hc+j] * scale[1] + base[hc+j])
//   comb:  [n, hc, hc]   start[j,k] = mixes[i, j*hc + k + 2*hc] * scale[2] + base[j*hc + k + 2*hc]
//                        comb = softmax(start, dim=-1) + eps
//                        comb /= comb.sum(dim=-2) + eps          (one extra col-norm to start)
//                        repeat (sinkhorn_iters - 1) times:
//                            comb /= comb.sum(dim=-1) + eps
//                            comb /= comb.sum(dim=-2) + eps
//
// Constants resolved at pipeline-creation time via function constants:
//   HC               — hyper-connection multiplicity (4 in V4 configs)
//   SINKHORN_ITERS   — number of Sinkhorn iterations (20 in V4)
//   HC_EPS           — additive eps (1e-6 in V4)
//
// Threadgroup layout: one threadgroup per output token (i in [0, n)),
// HC*HC threads. Each thread owns one (j,k) cell of `comb`. The hc values
// are tiny (4×4 = 16 cells) so everything lives in threadgroup memory.

constant uint  HC             [[function_constant(4)]];
constant uint  SINKHORN_ITERS [[function_constant(5)]];
constant float HC_EPS         [[function_constant(6)]];

inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + exp(-x));
}

kernel void hc_split_sinkhorn_f32(
    device const float* mixes    [[buffer(0)]],
    device const float* hcScale  [[buffer(1)]],
    device const float* hcBase   [[buffer(2)]],
    device float*       pre      [[buffer(3)]],
    device float*       post     [[buffer(4)]],
    device float*       comb     [[buffer(5)]],
    threadgroup float*  shared_  [[threadgroup(0)]],
    uint   row [[threadgroup_position_in_grid]],
    uint   tid [[thread_position_in_threadgroup]]
) {
    uint hc = HC;
    uint mix_hc = (2u + hc) * hc;
    device const float* mix_row = mixes + row * mix_hc;

    // shared_ layout:
    //   [0 .. hc*hc)        — comb tile
    //   [hc*hc .. hc*hc+hc) — row-sum / col-sum scratch
    threadgroup float* combT = shared_;
    threadgroup float* rowSum = shared_ + hc * hc;
    threadgroup float* colSum = shared_ + hc * hc + hc;

    uint j = tid / hc;
    uint k = tid - j * hc;
    bool active = j < hc && k < hc;

    // pre / post — first thread of each j writes them out.
    if (tid < hc) {
        uint jj = tid;
        float pv = sigmoid_f(mix_row[jj] * hcScale[0] + hcBase[jj]) + HC_EPS;
        float qv = 2.0f * sigmoid_f(mix_row[jj + hc] * hcScale[1] + hcBase[jj + hc]);
        pre [row * hc + jj] = pv;
        post[row * hc + jj] = qv;
    }

    // Initialise comb tile.
    if (active) {
        uint idx = j * hc + k + 2u * hc;
        combT[j * hc + k] = mix_row[idx] * hcScale[2] + hcBase[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Row-softmax: subtract row-max then exp.
    if (active && k == 0) {
        float m = combT[j * hc + 0];
        for (uint kk = 1; kk < hc; kk++) m = max(m, combT[j * hc + kk]);
        rowSum[j] = m;          // re-use rowSum to broadcast row-max
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
        combT[j * hc + k] = exp(combT[j * hc + k] - rowSum[j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active && k == 0) {
        float s = 0.0f;
        for (uint kk = 0; kk < hc; kk++) s += combT[j * hc + kk];
        rowSum[j] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
        combT[j * hc + k] = combT[j * hc + k] / rowSum[j] + HC_EPS;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Initial col-norm (matches the reference's `comb = comb / (comb.sum(-2) + eps)`
    // before the iter loop).
    if (active && j == 0) {
        float s = 0.0f;
        for (uint jj = 0; jj < hc; jj++) s += combT[jj * hc + k];
        colSum[k] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
        combT[j * hc + k] = combT[j * hc + k] / (colSum[k] + HC_EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // (sinkhorn_iters - 1) alternations: row, col.
    for (uint iter = 1; iter < SINKHORN_ITERS; iter++) {
        if (active && k == 0) {
            float s = 0.0f;
            for (uint kk = 0; kk < hc; kk++) s += combT[j * hc + kk];
            rowSum[j] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) combT[j * hc + k] = combT[j * hc + k] / (rowSum[j] + HC_EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active && j == 0) {
            float s = 0.0f;
            for (uint jj = 0; jj < hc; jj++) s += combT[jj * hc + k];
            colSum[k] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) combT[j * hc + k] = combT[j * hc + k] / (colSum[k] + HC_EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        comb[(row * hc + j) * hc + k] = combT[j * hc + k];
    }
}
