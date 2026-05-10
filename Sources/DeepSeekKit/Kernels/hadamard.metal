#include <metal_stdlib>
using namespace metal;

// Walsh-Hadamard transform for power-of-2 dims, applied in place along the
// last axis. Used by `rotate_activation` in the reference Indexer/Compressor
// (model.py:247–251) before FP4 quantization.
//
// Layout:
//   x: [rows, dim] f32  (rows runs across all leading axes flattened together)
// Each threadgroup handles one row. Threadgroup size must be dim/2 so every
// pair-swap in the FWHT happens in parallel within one barrier interval.
//
// Algorithm (see e.g. https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform):
//   for stride in 1, 2, 4, ..., dim/2:
//     for each pair (a_idx, a_idx+stride) within the threadgroup:
//       (x[a_idx], x[a_idx+stride]) = (x[a_idx]+x[a_idx+stride],
//                                      x[a_idx]-x[a_idx+stride])
//   x *= 1 / sqrt(dim)         // matches reference's scale=d^-0.5
//
// `dim` must be a power of 2. For DeepSeek-V4 the only dims rotated are 128
// and 512, both pow2. The kernel does NOT validate this.
//
// Threadgroup memory size (`tile`) must be set host-side via
// setThreadgroupMemoryLength = dim * sizeof(float).

kernel void hadamard_f32(
    device float*           x         [[buffer(0)]],
    constant uint&          dim       [[buffer(1)]],
    threadgroup float*      tile      [[threadgroup(0)]],
    uint                    row       [[threadgroup_position_in_grid]],
    uint                    tid       [[thread_position_in_threadgroup]],
    uint                    tgsize    [[threads_per_threadgroup]]
) {
    device float* xrow = x + row * dim;

    // 1) Cooperative load row -> tile.
    for (uint i = tid; i < dim; i += tgsize) tile[i] = xrow[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) FWHT butterflies. tid in [0, dim/2) handles one pair per stride.
    //    Pair indices for stride s:
    //      group_size = 2*s
    //      group  = tid / s
    //      local  = tid % s
    //      a = group * group_size + local
    //      b = a + s
    for (uint stride = 1; stride < dim; stride <<= 1) {
        uint group_size = stride << 1;
        uint group = tid / stride;
        uint local = tid - group * stride;
        uint a = group * group_size + local;
        uint b = a + stride;
        if (b < dim) {
            float va = tile[a];
            float vb = tile[b];
            tile[a] = va + vb;
            tile[b] = va - vb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3) Normalize by 1/sqrt(dim) and write back.
    float norm = rsqrt((float)dim);
    for (uint i = tid; i < dim; i += tgsize) xrow[i] = tile[i] * norm;
}
