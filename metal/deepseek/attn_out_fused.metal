// Decode attention output-B + HC=4, ported from GiorgioOppo/ds4
// b8507c9f848abb8f24020b136a3db2834ffd600a:metal/moe.metal.
// Concatenate after moe.metal and dsv4_hc.metal. Reuse the Swift baseline's
// Q4_K matvec implementation and HC argument layout; no new quantization math.
// Q8_0 uses the existing kernel_dsv4_q8_hc_expand4_q8_0 in dsv4_hc.metal.
kernel void kernel_dsv4_attn_out_q4_K_hc_expand4(
        constant ds4_metal_args_mul_mv & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device const char * weight,
        device const char * input,
        device       char * block_out,
        device const char * residual,
        device const char * post,
        device const char * comb,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    // moe.metal undefines N_R0_Q4_K after defining the shared helper.
    // Keep its two-row specialization local to this separate kernel unit.
    constexpr short rows_per_simdgroup = 2;
    if (hc.n_hc != 4 || hc.n_tokens != 1 || hc.has_add != 0 ||
        mv.ne0 != hc.n_embd || (mv.ne0 & 1) != 0 ||
        hc.nb_block0 != sizeof(float)) {
        return;
    }

    const int first_row =
        (tgpig.x * FC_mul_mv_nsg + sgitg) * rows_per_simdgroup;
    if (first_row < mv.ne0) {
        kernel_mul_mv_q4_K_f32_impl<rows_per_simdgroup,
                                  constant ds4_metal_args_mul_mv &>(
            mv, weight, input, block_out, shmem,
            tgpig, tiisg, sgitg);
    }

    // Every simdgroup, including an inactive tail group, reaches this barrier.
    // Read the materialized F32 value exactly as the standalone HC dispatch
    // does; keep block_out available for diagnostics and parity comparisons.
    threadgroup_barrier(mem_flags::mem_device);

    if (tiisg != 0) return;
    FOR_UNROLL(short row = 0; row < rows_per_simdgroup; ++row) {
        const int d = first_row + row;
        if (d >= mv.ne0) continue;

        const float block_v = *((device const float *)(
            block_out + (uint64_t)d * hc.nb_block0));
        const float r0 = *((device const float *)(
            residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
        const float r1 = *((device const float *)(
            residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
        const float r2 = *((device const float *)(
            residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
        const float r3 = *((device const float *)(
            residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

        FOR_UNROLL(short dst_hc = 0; dst_hc < 4; ++dst_hc) {
            float acc = block_v * *((device const float *)(
                post + (uint64_t)dst_hc * hc.nb_post0));
            acc += *((device const float *)(
                comb + (uint64_t)dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
            acc += *((device const float *)(
                comb + (uint64_t)dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
            acc += *((device const float *)(
                comb + (uint64_t)dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
            acc += *((device const float *)(
                comb + (uint64_t)dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;
            *((device float *)(dst + (uint64_t)d * hc.nb0 +
                                (uint64_t)dst_hc * hc.nb1)) = acc;
        }
    }
}
