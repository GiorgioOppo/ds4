#include <metal_stdlib>
using namespace metal;

// Heavily Compressed Attention (HCA) — DeepSeek-V4 hybrid attention.
//
// NOT IMPLEMENTED. HCA is applied on every `hca_layer_stride`-th layer and
// uses a much larger compression ratio than CSA (paper claims ~10% of V3.2
// KV cache at 1M context). The exact operator is unspecified in the public
// summary and must be ported from modeling_deepseek_v4.py.
//
// Open questions to resolve before writing this kernel:
//   - Is the compression done jointly with CSA's compressor or independent?
//   - Does HCA preserve full Q resolution and only compress K/V, or also Q?
//   - How is RoPE applied across the compressed positions?
kernel void attention_hca_unimplemented(
    device float* dst [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) dst[0] = NAN;
}
