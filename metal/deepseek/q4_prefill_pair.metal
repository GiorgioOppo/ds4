// Q-A/KV prefill specialization from ds4 b8507c9f848abb8f24020b136a3db2834ffd600a.
// Same M32 x N32 x K64 staging and K accumulation as the C F16-RHS kernel.
// Dequantization keeps Swift's existing Q4_K arithmetic, including half d/16,
// so this optimization does not change the meaning of subnormal scale blocks.
// The host admits only complete tiles and the production 4096 -> 1024/512 shape.

kernel void kernel_dsv4_q4_prefill_rhs_f16(
        constant uint &count [[buffer(0)]],
        device const float *src [[buffer(1)]],
        device half4 *dst [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const uint base = 4 * gid;
    if (base >= count) return; // host requires count divisible by four
    // Scalar input access also permits F32 views with a non-vector-aligned offset.
    dst[gid] = half4(src[base], src[base + 1], src[base + 2], src[base + 3]);
}

kernel void kernel_dsv4_q4_prefill_f16_rhs_m32_k64(
        constant ds4_metal_args_mul_mm &args [[buffer(0)]],
        device const char *src0 [[buffer(1)]],
        device const char *src1 [[buffer(2)]],
        device float *dst [[buffer(3)]],
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 32, NR1 = 32, NK = 64, NL0 = 4, NL1 = 4, MA = 2;
    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;
    const short il0 = tiitg % NL0;
    short il = il0;
    device const block_q4_K *x =
        (device const block_q4_K *)(src0 + args.nb01 * (r0 + tiitg / NL0));
    const short iy = 8 * (tiitg % NL1);
    device const half *y = (device const half *)(src1
        + args.nb11 * (r1 + tiitg / NL1) + args.nb10 * iy);
    simdgroup_half8x8 ma[MA], mb[2];
    simdgroup_float8x8 mc[2 * MA];
    for (short i = 0; i < 2 * MA; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loopK = 0; loopK < args.ne00; loopK += NK) {
        half4x4 a;
        dequantize_q4_K(x, il, a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = (NR0 / 8) * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = a[i / 4][i % 4];
        }
        #pragma unroll
        for (short pass = 0; pass < 2; ++pass) {
            const short sx = tiitg % NL1;
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 1024 * pass + 64 * ib + 8 * ly) =
                *((device const half2x4 *)(y + 32 * pass));
        }
        il = il + 4 < 16 ? il + 4 : il % NL0;
        x = il < NL0 ? x + 1 : x;
        y += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + MA * 64 * (sgitg % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);
        #pragma unroll
        for (short ik = 0; ik < NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < MA; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 2 * MA; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / MA], ma[i % MA], mc[i]);
            }
            lsma += (NR0 / 8) * 64;
            lsmb += 4 * 64;
        }
    }
    device float *out = dst + r0 + (NR0 / 2) * (sgitg & 1)
        + (r1 + 16 * (sgitg >> 1)) * args.ne0;
    for (short i = 0; i < 2 * MA; ++i) {
        simdgroup_store(mc[i], out + 8 * (i % MA) + 8 * args.ne0 * (i / MA), args.ne0, 0, false);
    }
}
