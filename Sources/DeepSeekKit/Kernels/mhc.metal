#include <metal_stdlib>
using namespace metal;

// Manifold-Constrained Hyper-Connections (mHC).
//
// NOT IMPLEMENTED. The V4 report describes mHC as a generalisation of
// hyper-connections (https://arxiv.org/abs/2409.19606) that constrains the
// per-layer mixing matrix to a low-rank manifold (rank = mhc_rank,
// expansion factor = mhc_expansion). Concretely each residual update is
//
//     h_{l+1} = sum_i alpha_i^{(l)} * Block_i(h_l)
//
// where alpha is a learned matrix factored as U @ V with U:[expansion,rank]
// and V:[rank,expansion], constrained to lie on a Stiefel manifold.
//
// Inference-time math is just a matmul of the rank-decomposed alpha against
// the stack of per-block outputs — but the layout (whether alpha is per-token
// or shared, whether expansion happens before or after the block, etc.)
// must come from the reference model code.
kernel void mhc_unimplemented(
    device float* dst [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) dst[0] = NAN;
}
