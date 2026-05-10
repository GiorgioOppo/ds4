#include <metal_stdlib>
using namespace metal;

// sparse_attn — sparse multi-head attention via index gathering + online
// softmax (FlashAttention-style).
//
// Port of `sparse_attn_kernel` from
// Original/DeepSeek-V4-Pro/inference/kernel.py lines 277–352.
//
// Inputs:
//   q:          [b, m, h, d]    BF16 (m = query length, h = heads)
//   kv:         [b, n, d]       BF16 (n = total keys including window + compressed)
//   attn_sink:  [h]             FP32 — per-head learnable sink logit
//   topk_idxs:  [b, m, topk]    INT32 — KV positions to gather (-1 = padding)
//   o:          [b, m, h, d]    BF16 output
//   scale:      float
//
// Algorithm per (batch, query_pos):
//   acc_o = 0;  sum_exp = 0;  scores_max = -inf
//   for each block of `block` (=64) topk indices:
//     gather kv_block from kv at the indices (zero where idx == -1)
//     acc_s = (idx != -1) ? 0 : -inf
//     acc_s += q_block @ kv_block^T
//     acc_s *= scale
//     online softmax update of (scores_max, sum_exp, acc_o)
//   sum_exp += exp(attn_sink[h] - scores_max)   // sink logit absorbed
//   o = acc_o / sum_exp
//
// NOT IMPLEMENTED. Required machinery:
//   - simdgroup matrix BF16 GEMM for the inner Q@K^T and softmax@V
//   - threadgroup-shared running max + exp scaling
//   - gather-with-mask: kv[idx], with idx == -1 producing 0 KV row + -inf logit
//   - h<16 zero-padding to align with simdgroup widths (reference does this
//     in the host wrapper at lines 358–367)
//
// This is the single most complex kernel in the project. Plan: implement it
// last, after all other pieces are numerically verified, and validate against
// a Python reference forward pass on a small toy config.

kernel void sparse_attn_unimplemented(
    device float* dst [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) dst[0] = NAN;
}
