#include <metal_stdlib>
using namespace metal;

// Hadamard transform — applied before FP4 quantization to spread information
// across dimensions, used by `rotate_activation` in the reference Indexer
// and Compressor (when rotate=True).
//
// Reference uses `fast_hadamard_transform.hadamard_transform(x, scale=d^-0.5)`.
// For dimensions that are powers of 2, this is a fast Walsh-Hadamard transform
// (FWHT). For non-power-of-2 dimensions, the reference library uses block
// decomposition (e.g., for d = m * 2^k, applies FWHT on each 2^k group then
// mixes via the m-dim Hadamard matrix).
//
// For DeepSeek-V4 the dims that get rotated are:
//   - index_head_dim = 128 (Indexer Q and KV)
//   - head_dim = 512 (Indexer KV via Compressor with rotate=True)
//
// Both are powers of 2, so a pure FWHT suffices. Algorithm:
//
//   for stride = 1, 2, 4, ..., d/2:
//     for each pair (i, i+stride) with (i & stride) == 0:
//       a = x[i]; b = x[i+stride]
//       x[i]        = a + b
//       x[i+stride] = a - b
//   x *= 1/sqrt(d)
//
// In Metal: do all log2(d) passes inside a single threadgroup with
// barriers between passes, threadgroup_size = d / 2.
//
// NOT IMPLEMENTED — straightforward to write but requires care on the
// barrier placement and on using `threadgroup_barrier(mem_flags::mem_threadgroup)`
// between passes.

kernel void hadamard_unimplemented(
    device float* dst [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) dst[0] = NAN;
}
