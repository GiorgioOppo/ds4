#include <metal_stdlib>
using namespace metal;

// MoE routing: top-k softmax gate over expert logits.
// gate_logits: [tokens, n_experts] f32
// out_idx:     [tokens, top_k] u32
// out_w:       [tokens, top_k] f32 (renormalised softmax weights)
//
// This is the standard DeepSeek-V3 routing scheme; V4 keeps it modulo
// auxiliary-loss-free balancing, which is a training-time concern only.
kernel void moe_topk_gate(
    device const float* gate_logits [[buffer(0)]],
    device uint*        out_idx     [[buffer(1)]],
    device float*       out_w       [[buffer(2)]],
    constant uint3&     dims        [[buffer(3)]],   // (tokens, n_experts, top_k)
    uint                t           [[thread_position_in_grid]]
) {
    uint T = dims.x, E = dims.y, K = dims.z;
    if (t >= T) return;

    device const float* row = gate_logits + t * E;

    // Selection sort top-K — fine for E up to a few hundred, which is what
    // DeepSeek MoE uses (V3: 256 routed experts).
    float bestV[16];
    uint  bestI[16];
    for (uint k = 0; k < K; k++) { bestV[k] = -INFINITY; bestI[k] = 0; }

    for (uint e = 0; e < E; e++) {
        float v = row[e];
        // find smallest in bestV
        uint  smallI = 0;
        float smallV = bestV[0];
        for (uint k = 1; k < K; k++) if (bestV[k] < smallV) { smallV = bestV[k]; smallI = k; }
        if (v > smallV) { bestV[smallI] = v; bestI[smallI] = e; }
    }

    // softmax over the K kept logits
    float m = -INFINITY;
    for (uint k = 0; k < K; k++) m = max(m, bestV[k]);
    float s = 0.0f;
    for (uint k = 0; k < K; k++) { bestV[k] = exp(bestV[k] - m); s += bestV[k]; }
    for (uint k = 0; k < K; k++) bestV[k] /= s;

    for (uint k = 0; k < K; k++) {
        out_idx[t * K + k] = bestI[k];
        out_w  [t * K + k] = bestV[k];
    }
}

// Per-token expert dispatch is deliberately left to host-orchestrated calls
// to matvec_q4: for batch=1 decode, the active set is just top_k experts and
// we issue top_k MLP matvecs sequentially. Full batched scatter/gather MoE
// dispatch (needed for prefill of long prompts) requires a token-permutation
// kernel + grouped GEMM, which is its own project — see README "Roadmap".
