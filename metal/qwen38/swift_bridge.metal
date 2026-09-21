// Swift orchestration-only pointwise operation. Qwen's stable sigmoid matches
// the upstream routed-expert implementation above.
kernel void kernel_qwen38_swiglu(constant uint &n, device const float *gate,
                                device const float *up, device float *out,
                                uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = qwen4_silu(gate[i]) * up[i];
}
