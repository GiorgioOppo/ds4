#include <metal_stdlib>
using namespace metal;

// hc_split_sinkhorn — splits the HC mixing tensor into pre/post/comb with
// Sinkhorn doubly-stochastic normalization on `comb`.
//
// Port of `hc_split_sinkhorn_kernel` from
// Original/DeepSeek-V4-Pro/inference/kernel.py lines 371–427.
//
// Inputs:
//   mixes:    [n, mix_hc]            FP32   where mix_hc = (2 + hc) * hc
//   hc_scale: [3]                    FP32
//   hc_base:  [mix_hc]               FP32
// Outputs:
//   pre:      [n, hc]                FP32   pre[i,j] = sigmoid(mixes[j]*scale[0] + base[j]) + eps
//   post:     [n, hc]                FP32   post[i,j] = 2*sigmoid(mixes[j+hc]*scale[1] + base[j+hc])
//   comb:     [n, hc, hc]            FP32   start = mixes[j*hc+k+2*hc]*scale[2] + base[j*hc+k+2*hc]
//                                            comb = softmax(start, dim=-1) + eps
//                                            for _ in range(sinkhorn_iters):
//                                              comb /= comb.sum(dim=-2) + eps
//                                              comb /= comb.sum(dim=-1) + eps
//                                            (with one extra row-norm at the end of init)
//
// One threadgroup per token (n outer dim). hc is small (4 in V4-Pro), so the
// hc×hc tile fits comfortably in registers.
//
// NOT IMPLEMENTED. Required machinery:
//   - row-softmax via simdgroup max + exp + sum
//   - alternating row-sum / col-sum reductions for Sinkhorn iterations
//   - serial loop of length `hc_sinkhorn_iters` (=20 in V4-Pro)
//
// This kernel is logistically simpler than sparse_attn (small fixed dims,
// no flash tiling) and would be a good first-implementation target.

kernel void hc_split_sinkhorn_unimplemented(
    device float* dst [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) dst[0] = NAN;
}
