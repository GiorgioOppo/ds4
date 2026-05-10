#include <metal_stdlib>
using namespace metal;

// MoE Gate scoring + top-k selection.
//
// Mirrors `Gate.forward` in
// Reference/inference/model.py lines 564–584.
//
// Input:
//   logits:       [tokens, n_experts] f32  — produced by linear(x, gate.weight)
//   bias:         [n_experts] f32          — additive shift used only for selection
// Outputs:
//   indices:      [tokens, top_k] u32      — top-k expert indices
//   weights:      [tokens, top_k] f32      — gating weights, renormalized
//
// score_func (selected via constant `SCORE`):
//   0 = softmax     scores = softmax(logits, dim=-1)
//   1 = sigmoid     scores = sigmoid(logits)
//   2 = sqrtsoftplus  scores = sqrt(softplus(logits))   (DeepSeek-V4 default)
//
// For sigmoid / sqrtsoftplus, `weights` is divided by its top-k sum so the
// gating weights normalize to 1 even though the raw scores don't.
// `route_scale` is applied at the end.

constant uint  SCORE      [[function_constant(0)]];
constant float ROUTE_SCALE [[function_constant(1)]];

inline float score_fn(float x) {
    if (SCORE == 0u) {
        return x;                                // softmax handled outside
    } else if (SCORE == 1u) {
        return 1.0f / (1.0f + exp(-x));
    } else {
        // sqrt(softplus(x)) = sqrt(log(1 + exp(x)))
        // numerically stable: softplus(x) = max(x, 0) + log(1 + exp(-|x|))
        // MSL has no log1p; substitute log(1 + .).
        float sp = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
        return sqrt(sp);
    }
}

kernel void moe_gate(
    device const float* logits   [[buffer(0)]],
    device const float* bias     [[buffer(1)]],   // may be null when SCORE==0 with hash routing
    device uint*        indices  [[buffer(2)]],
    device float*       weights  [[buffer(3)]],
    constant uint3&     dims     [[buffer(4)]],   // (tokens, n_experts, top_k)
    uint                t        [[thread_position_in_grid]]
) {
    uint T = dims.x, E = dims.y, K = dims.z;
    if (t >= T) return;

    device const float* row = logits + t * E;

    // 1) score per expert + remember the original (un-biased) score for the gating weight.
    float orig[256];   // n_experts up to 256 in V4 configs; oversize is fine
    float sel[256];
    for (uint e = 0; e < E; e++) {
        float l = row[e];
        if (SCORE == 0u) {
            orig[e] = l;
        } else {
            orig[e] = score_fn(l);
        }
    }
    if (SCORE == 0u) {
        // softmax in place
        float m = -INFINITY;
        for (uint e = 0; e < E; e++) m = max(m, orig[e]);
        float s = 0;
        for (uint e = 0; e < E; e++) { orig[e] = exp(orig[e] - m); s += orig[e]; }
        for (uint e = 0; e < E; e++) orig[e] /= s;
    }

    for (uint e = 0; e < E; e++) {
        float v = orig[e];
        if (SCORE != 0u && bias != nullptr) v += bias[e];
        sel[e] = v;
    }

    // 2) selection sort top-K from sel[]
    float bestV[16]; uint bestI[16];
    for (uint k = 0; k < K; k++) { bestV[k] = -INFINITY; bestI[k] = 0; }
    for (uint e = 0; e < E; e++) {
        float v = sel[e];
        uint smallI = 0; float smallV = bestV[0];
        for (uint k = 1; k < K; k++) if (bestV[k] < smallV) { smallV = bestV[k]; smallI = k; }
        if (v > smallV) { bestV[smallI] = v; bestI[smallI] = e; }
    }

    // 3) gather original scores for the selected experts and renormalise
    float gw[16];
    float sumGw = 0;
    for (uint k = 0; k < K; k++) {
        gw[k] = orig[bestI[k]];
        sumGw += gw[k];
    }
    if (SCORE != 0u) {
        for (uint k = 0; k < K; k++) gw[k] /= max(sumGw, 1e-12f);
    }
    for (uint k = 0; k < K; k++) gw[k] *= ROUTE_SCALE;

    for (uint k = 0; k < K; k++) {
        indices[t * K + k] = bestI[k];
        weights[t * K + k] = gw[k];
    }
}
