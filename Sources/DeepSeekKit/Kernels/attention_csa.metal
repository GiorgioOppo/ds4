#include <metal_stdlib>
using namespace metal;

// Compressed Sparse Attention (CSA) — DeepSeek-V4 hybrid attention.
//
// NOT IMPLEMENTED. Per the V4 technical report, CSA combines:
//   1. a learnable compression of the K/V projections by csa_compression_ratio
//      across the sequence axis, producing compact summary tokens
//   2. a sparse attention pattern over those summary tokens (block-sparse
//      and/or top-k), so per-token attention cost scales sub-linearly in
//      sequence length
//
// Implementing it correctly requires:
//   - the exact compression operator from modeling_deepseek_v4.py
//     (linear? strided conv? learned pooling?)
//   - the sparse pattern definition (block size, sink/window tokens, top-k)
//   - the masking rules for causal generation with compressed K/V
//
// Until the reference implementation is downloaded from
// huggingface.co/deepseek-ai/DeepSeek-V4-Pro, this kernel is a placeholder
// that traps when invoked. Do NOT replace with a guess.
kernel void attention_csa_unimplemented(
    device float* dst [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    // Touching the buffer is what surfaces the fault host-side.
    if (gid == 0) dst[0] = NAN;
}
