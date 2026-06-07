// AUTO-GENERATED from metal/*.metal — do not edit by hand.
// Regenerate with: make embed-kernels  (scripts/embed_kernels.sh).
// Embeds the kernel sources in the binary so the Metal runtime needs no
// on-disk kernel folder (works in SwiftPM, the .xcodeproj, and a shipped .app).

extension MetalRuntime {
    static let embeddedKernels: [String: String] = [
        "flash_attn": ###"""
#define FC_FLASH_ATTN_EXT_PAD 100
#define FC_FLASH_ATTN_EXT_BLK 200
#define FC_FLASH_ATTN_EXT 300
#define FC_FLASH_ATTN_EXT_VEC 400
#define FC_FLASH_ATTN_EXT_VEC_REDUCE 500
#define OP_FLASH_ATTN_EXT_NQPSG 8
#define OP_FLASH_ATTN_EXT_NCPSG 64
#define OP_FLASH_ATTN_EXT_VEC_NQPSG 1
#define OP_FLASH_ATTN_EXT_VEC_NCPSG 32

#ifndef PAD2
#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))
#endif

template <typename type4>
void dequantize_f32_t4(device const float4 * src, short il, thread type4 & reg) {
    reg = (type4)(*src);
}

template <typename type4>
void dequantize_f16_t4(device const half4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src));
}

template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg);

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg);

struct ds4_metal_args_flash_attn_ext_pad {
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
};

struct ds4_metal_args_flash_attn_ext_blk {
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
};

struct ds4_metal_args_flash_attn_ext {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
};

struct ds4_metal_args_flash_attn_ext_vec {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
};

struct ds4_metal_args_flash_attn_ext_vec_reduce {
    int32_t nrows;
};

constant bool FC_flash_attn_ext_pad_has_mask [[function_constant(FC_FLASH_ATTN_EXT_PAD + 0)]];
constant int32_t FC_flash_attn_ext_pad_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_PAD + 25)]];

// DS4 FlashAttention padding: pads the final partial K/V/mask cache block so the
// vector FlashAttention kernel can read full 32-row chunks.
kernel void kernel_flash_attn_ext_pad(
        constant ds4_metal_args_flash_attn_ext_pad & args,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const int32_t C = FC_flash_attn_ext_pad_ncpsg;

    device char * k_pad    = dst;
    device char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;
    device char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;

    const int32_t icp = args.ne11 % C;
    const int32_t ic0 = args.ne11 - icp;

    const int32_t i1 = tgpig[0];
    const int32_t i2 = tgpig[1];
    const int32_t i3 = tgpig[2];

    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
        device const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;
        device const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;

        device char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;
        device char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;

        if (i1 >= icp) {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = 0;
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = 0;
            }
        } else {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = k_src[i];
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = v_src[i];
            }
        }
    }

    if (FC_flash_attn_ext_pad_has_mask) {
        if (i2 < args.ne32 && i3 < args.ne33) {
            for (int ib = i1; ib < args.ne31; ib += C) {
                device const half * mask_src = (device const half *)(mask      + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;
                device       half * mask_dst = (device       half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;

                for (int i = tiitg; i < C; i += ntg.x) {
                    if (i >= icp) {
                        mask_dst[i] = -MAXHALF;
                    } else {
                        mask_dst[i] = mask_src[i];
                    }
                }
            }
        }
    }
}

constant int32_t FC_flash_attn_ext_blk_nqptg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 24)]];
constant int32_t FC_flash_attn_ext_blk_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 25)]];

// DS4 FlashAttention mask scan: marks blocks so the non-vector kernel can skip
// blocks that are entirely masked or entirely zero.
kernel void kernel_flash_attn_ext_blk(
        constant ds4_metal_args_flash_attn_ext_blk & args,
        device const char * mask,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int32_t Q = FC_flash_attn_ext_blk_nqptg;
    const int32_t C = FC_flash_attn_ext_blk_ncpsg;

    constexpr short NW  = N_SIMDWIDTH;

    const int32_t i3 = tgpig[2]/args.ne32;
    const int32_t i2 = tgpig[2]%args.ne32;
    const int32_t i1 = tgpig[1];
    const int32_t i0 = tgpig[0];

    char res = i0*C + C > args.ne30 ? 1 : 0;

    if ((C > NW || Q > 1) && res == 0) {
        half mmin =  MAXHALF;
        half mmax = -MAXHALF;
        const int32_t q0 = i1*Q;

        FOR_UNROLL (short j = 0; j < Q; ++j) {
            if (q0 + j < args.ne31) {
                device const half * mask_src =
                    (device const half *) (mask + (q0 + j)*args.nb31 + i2*args.nb32 + i3*args.nb33) +
                    i0*C + tiisg;

                FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {
                    mmin = min(mmin, mask_src[ii*NW]);
                    mmax = max(mmax, mask_src[ii*NW]);
                }
            }
        }

        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        if (mmax > -MAXHALF) {
            if (mmin == 0.0 && mmax == 0.0) {
                res = 2;
            } else {
                res = 1;
            }
        }
    }

    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
    const int32_t nblk0 = ((args.ne30 + C - 1)/C);

    if (tiisg == 0) {
        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
    }
}

constant bool FC_flash_attn_ext_has_mask  [[function_constant(FC_FLASH_ATTN_EXT + 0)]];
constant bool FC_flash_attn_ext_has_sinks [[function_constant(FC_FLASH_ATTN_EXT + 1)]];
constant bool FC_flash_attn_ext_has_bias  [[function_constant(FC_FLASH_ATTN_EXT + 2)]];
constant bool FC_flash_attn_ext_has_scap  [[function_constant(FC_FLASH_ATTN_EXT + 3)]];
constant bool FC_flash_attn_ext_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT + 4)]];

constant bool FC_flash_attn_ext_bc_mask [[function_constant(FC_FLASH_ATTN_EXT + 10)]];

constant int32_t FC_flash_attn_ext_ns10 [[function_constant(FC_FLASH_ATTN_EXT + 20)]];
constant int32_t FC_flash_attn_ext_ns20 [[function_constant(FC_FLASH_ATTN_EXT + 21)]];
constant int32_t FC_flash_attn_ext_nsg  [[function_constant(FC_FLASH_ATTN_EXT + 22)]];

// DS4 non-vector FlashAttention. The only exported instance uses the model's
// 512-wide F16 K/V rows; keeping the template body generic preserves the same
// arithmetic for dense and compressed-attention prefill.
template<
    typename q_t,
    typename q4_t,
    typename q8x8_t,
    typename k_t,
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,
    typename qk8x8_t,
    typename s_t,
    typename s2_t,
    typename s8x8_t,
    typename o_t,
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t,
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,
    short DV,
    short Q,
    short C,
    short NSG>
void kernel_flash_attn_ext_impl(
        constant ds4_metal_args_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0]*Q;

#define NS10 (FC_flash_attn_ext_ns10)
#define NS20 (FC_flash_attn_ext_ns20)

    constexpr short KV   = 8;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;
    constexpr short DV4  = DV/4;
    constexpr short DV16 = DV/16;

    constexpr short PV   = PAD2(DV, 64);
    constexpr short PV4  = PV/4;
    constexpr short PV8  = PV/8;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NQ  = Q/NSG;
    constexpr short SH  = 2*C;

    constexpr short TS = 2*SH;
    constexpr short T  = DK + 2*PV;

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T);
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T);
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK);
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T);
    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T);

    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);

    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);

    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);

    device const half2 * pm2[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm2[jj] = (device const half2 *) ((device const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
        const int32_t nblk0 = ((args.ne11 + C - 1)/C);

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        device const float4 * q4 = (device const float4 *) ((device const char *) q + j*args.nb01);

        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*DK4 + i] = (q4_t) q4[i];
            } else {
                sq4[j*DK4 + i] = 0;
            }
        }
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        for (short i = tiisg; i < DV4; i += NW) {
            so4[j*PV4 + i] = 0;
        }

        for (short i = tiisg; i < SH; i += NW) {
            ss[j*SH + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };

    {
        float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

        float slope = 1.0f;

        if (FC_flash_attn_ext_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        for (int ic0 = 0; ; ++ic0) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            if (FC_flash_attn_ext_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_has_mask) {
                    threadgroup half * sm = (threadgroup half *) (sm2);

                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        for (short i = tiisg; i < C; i += NW) {
                            if (ic + i >= args.ne11) {
                                sm[2*j*SH + i] = -MAXHALF;
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        pm2[jj] = (device const half2 *) ((device const half *) mask +
                                (iq1 + j)*C +
                                (iq2%args.ne32)*(C*args.ne31) +
                                (iq3%args.ne33)*(C*args.ne31*args.ne32));
                    }
                }

                ic = 0;
            }

            char blk_cur = 1;

            if (FC_flash_attn_ext_has_mask) {
                blk_cur = blk[ic0];

                if (blk_cur == 0) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }

                    continue;
                }

                if (blk_cur == 1) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        if (FC_flash_attn_ext_bc_mask) {
                            sm2[j*SH + tiisg] = (iq1 + j) < args.ne31 ? pm2[jj][tiisg] : half2(-MAXHALF, -MAXHALF);
                        } else {
                            sm2[j*SH + tiisg] = pm2[jj][tiisg];
                        }

                        pm2[jj] += NW;
                    }
                } else if (blk_cur == 2) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }
                }
            }

            if (is_same<kd4x4_t, k4x4_t>::value) {
                device      const k_t * pk = (device const k_t *) (k + ic*args.nb11);
                threadgroup const q_t * pq = sq;
                threadgroup       s_t * ps = ss;

                pk += sgitg*(8*NS10);
                ps += sgitg*(8*1);

                static_assert((C/8) % NSG == 0, "");

                constexpr short NC = (C/8)/NSG;

                FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    if (DK % 16 != 0) {
                        k8x8_t mk;
                        q8x8_t mq;

                        FOR_UNROLL (short i = 0; i < DK8; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mk, pk + 8*i, NS10, 0, true);
                            simdgroup_load(mq, pq + 8*i, DK);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        }
                    } else {
                        k8x8_t mk[2];
                        q8x8_t mq[2];

                        #pragma unroll (MIN(DK8/2, 4*NSG))
                        for (short i = 0; i < DK8/2; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                            simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);

                            simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                            simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                        }
                    }

                    simdgroup_store(mqk, ps, SH, 0, false);

                    pk += 8*(NSG*NS10);
                    ps += 8*(NSG);
                }
            } else {
                for (short ccc = 0; ccc < (C/8)/NSG; ++ccc) {
                    const short cc = ccc*NSG + sgitg;

                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    for (short ii = 0; ii < DK16; ii += 4) {
                        device const kd4x4_t * pk4x4 = (device const kd4x4_t *) (k + ((ic + 8*cc + ty)*args.nb11));

                        if (DK16%4 == 0) {
                            {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            FOR_UNROLL (short k = 0; k < 4; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        } else {
                            if (ii + tx < DK16) {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            for (short k = 0; k < 4 && ii + k < DK16; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        }
                    }

                    simdgroup_store(mqk, ss + 8*cc, SH, 0, false);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];

                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;

                if (FC_flash_attn_ext_has_scap) {
                    s2 = args.logit_softcap*precise::tanh(s2);
                }

                if (blk_cur != 2) {
                    if (FC_flash_attn_ext_has_bias) {
                        s2 += s2_t(sm2[j*SH + tiisg])*slope;
                    } else {
                        s2 += s2_t(sm2[j*SH + tiisg]);
                    }
                }

                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

                const float  ms  = exp(m  - M[jj]);
                const float2 vs2 = exp(s2 - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

                ss2[j*SH/2 + tiisg] = vs2;

                if (DV4 % NW == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                        const short i = ii*NW + tiisg;

                        so4[j*PV4 + i] *= ms;
                    }
                } else {
                    for (short i = tiisg; i < DV4; i += NW) {
                        so4[j*PV4 + i] *= ms;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            {
                if (is_same<vd4x4_t, v4x4_t>::value) {
                    static_assert(PV8 % NSG == 0, "");

                    constexpr short NO = PV8/NSG;

                    o8x8_t lo[NO];

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_load(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }

                    {
                        device const v_t * pv = (device const v_t *) (v + ic*args.nb21);

                        pv += 8*sgitg;

                        if (DV <= 64) {
                            FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
                                s8x8_t vs;
                                simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[2];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv[1], lo[2*ii + 1]);
                                }

                                pv  += 8*NS20;
                            }
                        } else {
                            constexpr short NC = (C/8)/2;

                            FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                                s8x8_t vs[2];

                                simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);
                                simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[4];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                                }

                                pv  += 2*8*NS20;
                            }
                        }
                    }

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_store(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }
                } else {
                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    for (short cc = 0; cc < C/8; ++cc) {
                        s8x8_t vs;
                        simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                        for (short ii = 4*sgitg; ii < DV16; ii += 4*NSG) {
                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) (v + ((ic + 8*cc + ty)*args.nb21));

                            if (DV16%4 == 0) {
                                {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                FOR_UNROLL (short k = 0; k < 4; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            } else {
                                if (ii + tx < DV16) {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < DV16; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            }
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (FC_flash_attn_ext_has_sinks) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

                M[jj] = simd_max(max(M[jj], s));

                const float ms = exp(m - M[jj]);
                const float vs = exp(s - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs);

                for (short i = tiisg; i < DV4; i += NW) {
                    so4[j*PV4 + i] *= ms;
                }
            }
        }
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        if (iq1 + j >= args.ne01) {
            break;
        }

        device float4 * dst4 = (device float4 *) dst + ((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*DV4;

        const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];

        if (DV4 % NW == 0) {
            FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                const short i = ii*NW + tiisg;

                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        } else {
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        }
    }

#undef NS10
#undef NS20
}

// Batched FlashAttention for prompt/prefill rows. It computes QK, applies mask,
// sinks, ALiBi/softcap options when enabled, and multiplies by V without
// materializing the full attention matrix.
template<
    typename q_t,
    typename q4_t,
    typename q8x8_t,
    typename k_t,
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,
    typename qk8x8_t,
    typename s_t,
    typename s2_t,
    typename s8x8_t,
    typename o_t,
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t,
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,
    short DV,
    short Q  = OP_FLASH_ATTN_EXT_NQPSG,
    short C  = OP_FLASH_ATTN_EXT_NCPSG>
kernel void kernel_flash_attn_ext(
        constant ds4_metal_args_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
#define FWD_TMPL q_t, q4_t, q8x8_t, k_t, k4x4_t, k8x8_t, v_t, v4x4_t, v8x8_t, qk_t, qk8x8_t, s_t, s2_t, s8x8_t, o_t, o4_t, o8x8_t, kd4x4_t, nl_k, deq_k, vd4x4_t, nl_v, deq_v, DK, DV, Q, C
#define FWD_ARGS args, q, k, v, mask, sinks, pad, blk, dst, shmem_f16, tgpig, tiisg, sgitg
    switch (FC_flash_attn_ext_nsg) {
        case 4: kernel_flash_attn_ext_impl<FWD_TMPL, 4>(FWD_ARGS); break;
        case 8: kernel_flash_attn_ext_impl<FWD_TMPL, 8>(FWD_ARGS); break;
    }
#undef FWD_TMPL
#undef FWD_ARGS
}

#define FA_NONVEC_TYPES \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8

typedef decltype(kernel_flash_attn_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 512, 512>) flash_attn_ext_dk512_t;

// Host-visible prefill FlashAttention variant for DS4's 512-wide F16 K/V rows.
template [[host_name("kernel_flash_attn_ext_f16_dk512_dv512")]]
kernel flash_attn_ext_dk512_t kernel_flash_attn_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 512, 512>;

#undef FA_NONVEC_TYPES

constant bool FC_flash_attn_ext_vec_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 0)]];
constant bool FC_flash_attn_ext_vec_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_VEC + 1)]];
constant bool FC_flash_attn_ext_vec_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 2)]];
constant bool FC_flash_attn_ext_vec_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 3)]];
constant bool FC_flash_attn_ext_vec_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT_VEC + 4)]];
constant int32_t FC_flash_attn_ext_vec_ns10 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 20)]];
constant int32_t FC_flash_attn_ext_vec_ns20 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 21)]];
constant int32_t FC_flash_attn_ext_vec_nsg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 22)]];
constant int32_t FC_flash_attn_ext_vec_nwg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 23)]];

// Decode FlashAttention for one query row. DS4 uses this in generation to scan
// raw and compressed KV cache chunks, optionally splitting long contexts across
// workgroups and writing partial softmax state for a later reduction.
template<
    typename q4_t,
    typename k4_t,
    typename v4_t,
    typename qk_t,
    typename s_t,
    typename s4_t,
    typename o4_t,
    typename kd4_t,
    short nl_k,
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t,
    short nl_v,
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,
    short DV,
    short NE = 4,
    short Q  = OP_FLASH_ATTN_EXT_VEC_NQPSG,
    short C  = OP_FLASH_ATTN_EXT_VEC_NCPSG>
kernel void kernel_flash_attn_ext_vec(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");

#define NWG  (FC_flash_attn_ext_vec_nwg)
#define NSG  (FC_flash_attn_ext_vec_nsg)
#define NS10 (FC_flash_attn_ext_vec_ns10)
#define NS20 (FC_flash_attn_ext_vec_ns20)

    const short iwg = tgpig[2]%NWG;

    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;
    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;
    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE;
    constexpr short SH  = 4*C;

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                      0*PK);
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*C + NSG*PK);
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*PV       + NSG*PK + NSG*SH);

    so4 += tiisg;

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    device const float4 * q4 = (device const float4 *) ((device const char *) q);

    if (iq1 < args.ne01) {
        for (short i = tiisg; i < PK4; i += NW) {
            if (i < DK4) {
                sq4[i] = (q4_t) q4[i];
            } else {
                sq4[i] = (q4_t) 0.0f;
            }
        }
    }

    for (short i = 0; i < DV4/NL; ++i) {
        so4[i*NL] = (o4_t) 0.0f;
    }

    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S = 0.0f;
        float M = -FLT_MAX/2;

        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        device const half * pm = (device const half *) (mask + iq1*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

        float slope = 1.0f;

        if (FC_flash_attn_ext_vec_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_vec_has_mask) {
                    if (ic + tiisg >= args.ne11) {
                        sm[tiisg] = -MAXHALF;
                    }
                } else {
                    pm = (device const half *) (mask) +
                        iq1*C +
                        (iq2%args.ne32)*(C*args.ne31) +
                        (iq3%args.ne33)*(C*args.ne31*args.ne32);
                }

                ic = 0;
            }

            if (FC_flash_attn_ext_vec_has_mask) {
                sm[tiisg] = pm[ic + tiisg];
            }

            if (simd_max(sm[tiisg]) <= -MAXHALF) {
                continue;
            }

            {
                device      const k4_t * pk4 = (device const k4_t *) (k + ic*args.nb11);
                threadgroup const q4_t * pq4 = sq4;

                pk4 += ty*NS10/4 + tx;
                pq4 += tx;

                qk_t mqk[C/NE] = { [ 0 ... C/NE - 1] = 0.0f };

                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    if (is_same<kd4_t, k4_t>::value) {
                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            mqk[cc] += dot((float4) pk4[cc*NE*NS10/4 +  ii*NL], (float4) pq4[ii*NL]);
                        }
                    } else {
                        device const kd4_t * pk = (device const kd4_t *) (k + ((ic + NE*cc + ty)*args.nb11));

                        k4_t mk;

                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                            mqk[cc] += dot((float4) mk, (float4) sq4[i]);
                        }
                    }

                    if (NE == 1) {
                        mqk[cc] = simd_sum(mqk[cc]);
                    } else {
                        if (NE <= 1) {
                            mqk[cc] += simd_shuffle_down(mqk[cc], 16);
                        }
                        if (NE <= 2) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  8);
                        }
                        if (NE <= 4) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  4);
                        }
                        if (NE <= 8) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  2);
                        }
                        if (NE <= 16) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  1);
                        }

                        mqk[cc] = simd_shuffle(mqk[cc], NL*ty);
                    }
                }

                if (FC_flash_attn_ext_vec_has_mask &&
                   !FC_flash_attn_ext_vec_has_scap &&
                   !FC_flash_attn_ext_vec_has_bias) {
                    ss[NE*tx + ty] = fma(mqk[tx], args.scale, (qk_t) sm[NE*tx + ty]);
                } else {
                    mqk[tx] *= args.scale;

                    if (FC_flash_attn_ext_vec_has_scap) {
                        mqk[tx] = args.logit_softcap*precise::tanh(mqk[tx]);
                    }

                    if (FC_flash_attn_ext_vec_has_bias) {
                        mqk[tx] += (qk_t) sm[NE*tx + ty]*slope;
                    } else {
                        mqk[tx] += (qk_t) sm[NE*tx + ty];
                    }

                    ss[NE*tx + ty] = mqk[tx];
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                const float m = M;
                const float s = ss[tiisg];

                M = simd_max(max(M, s));

                const float ms = exp(m - M);
                const float vs = exp(s - M);

                S = S*ms + simd_sum(vs);

                ss[tiisg] = vs;

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] *= ms;
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                o4_t lo[DV4/NL];
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    lo[ii] = 0.0f;
                }

                if (is_same<vd4_t, v4_t>::value) {
                    device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);

                    pv4 += ty*NS20/4 + tx;

                    const auto sst = ss + ty;

                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            lo[ii] += o4_t(float4(pv4[cc*NE*NS20/4 + ii*NL])*float4(sst[cc*NE]));
                        }
                    }
                } else {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        device const vd4_t * pv4 = (device const vd4_t *) (v + ((ic + NE*cc + ty)*args.nb21));

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            v4_t mv;
                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                            lo[ii] += o4_t(float4(mv)*float4(ss[NE*cc + ty]));
                        }
                    }
                }

                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    if (NE > 1) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0], 16);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1], 16);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2], 16);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3], 16);
                    }

                    if (NE > 2) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  8);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  8);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  8);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  8);
                    }

                    if (NE > 4) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  4);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  4);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  4);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  4);
                    }

                    if (NE > 8) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  2);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  2);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  2);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  2);
                    }

                    if (NE > 16) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  1);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  1);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  1);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  1);
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] += lo[ii];
                    }
                }
            }
        }

        if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {
            const float m = M;
            const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

            M = simd_max(max(M, s));

            const float ms = exp(m - M);
            const float vs = exp(s - M);

            S = S*ms + simd_sum(vs);

            if ((DV4/NL % NW == 0) || ty == 0) {
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    so4[ii*NL] *= ms;
                }
            }
        }

        if (tiisg == 0) {
            ss[0] = (s_t) S;
            ss[1] = (s_t) M;
        }
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const float S0 = ss[           0];
            const float S1 = ss[r*(SH/2) + 0];

            const float M0 = ss[           1];
            const float M1 = ss[r*(SH/2) + 1];

            const float M = max(M0, M1);

            const float ms0 = exp(M0 - M);
            const float ms1 = exp(M1 - M);

            const float S = S0*ms0 + S1*ms1;

            if (tiisg == 0) {
                ss[0] = S;
                ss[1] = M;
            }

            for (short i = tiisg; i < DV4; i += NW) {
                so4[i] = so4[i]*ms0 + so4[i + r*PV4]*ms1;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;
        const int64_t rid   = iq3*args.ne2*args.ne1 + iq2 + iq1*args.ne1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG;

        const float S = NWG == 1 ? (ss[0] == 0.0f ? 0.0f : 1.0f/ss[0]) : 1.0f;

        for (short i = tiisg; i < DV4; i += NW) {
            dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[i]*S;
        }

        if (NWG > 1) {
            if (tiisg == 0) {
                dst1[rid*(2*NWG) + 2*iwg + 0] = ss[0];
                dst1[rid*(2*NWG) + 2*iwg + 1] = ss[1];
            }
        }
    }

#undef NWG
#undef NSG
#undef NS10
#undef NS20
}

#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#define FA_TYPES_F32 \
           half4,  \
           float4, \
           float4, \
    float,         \
    float, float4, \
           float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

// Host-visible decode FlashAttention variant for DS4's 512-wide F16 K/V rows.
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,  1, dequantize_f16_t4, half4,  1, dequantize_f16_t4, 512, 512, 1>;

#undef FA_TYPES
#undef FA_TYPES_F32

constant int32_t FC_flash_attn_ext_vec_reduce_DV  [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 0)]];
constant int32_t FC_flash_attn_ext_vec_reduce_NWG [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 1)]];

// Reduces split-K decode FlashAttention partials. It combines each workgroup's
// output vector and softmax (sum,max) pair into the final attention result.
kernel void kernel_flash_attn_ext_vec_reduce(
        constant ds4_metal_args_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t rid = tgpig;

    const short iwg = tiisg;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV*NWG;

    float S = ss[rid*(2*NWG) + 2*iwg + 0];
    float M = ss[rid*(2*NWG) + 2*iwg + 1];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV/4;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);

        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }

#undef NWG
#undef DV
}
"""###,
        "dense": ###"""
// DS4 Metal matvec kernels used by generation.

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];

struct ds4_metal_args_mul_mv {
    int ne00;
    int ne01;
    int ne02;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    int ne10;
    int ne11;
    int ne12;
    ulong nb10;
    ulong nb11;
    ulong nb12;
    ulong nb13;
    int ne0;
    int ne1;
    int nr0;
    short r2;
    short r3;
};

struct ds4_metal_args_mul_mm {
    int32_t ne00;
    int32_t ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
};

struct ds4_metal_args_mul_mv_ext {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
};

template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

template<short NR0, typename args_t>
void kernel_mul_mv_q8_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);

    const int ib0 = sgitg*NQ + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NQ*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// Decode-time Q8_0 matrix-vector multiply. DS4 uses this for Q8_0 dense
// projections such as shared experts and output-side small matvecs.
[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// Decode shared-expert gate/up projections followed by SwiGLU:
//
//     mid = silu(min(gate, limit)) * clamp(up, -limit, limit)
//
// DS4's shared expert uses two Q8_0 matrices with the same input row.  This
// kernel preserves the exact Q8_0 dot-product reduction shape for both
// projections, still writes gate/up for diagnostics, and derives `mid` in the
// same lane that owns the reduced output row.  The point is not to fuse two
// independent weight streams into one matmul; it is to remove the separate
// activation pass and its reread of the two 2048-wide rows.
[[host_name("kernel_dsv4_shared_gate_up_swiglu_q8_0")]]
kernel void kernel_dsv4_shared_gate_up_swiglu_q8_0(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        constant     float &clamp_value,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = args.ne00 / QK8_0;
    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float *y = (device const float *)(src1 + offset1);

    device const block_q8_0 *ag[NR0];
    device const block_q8_0 *au[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 +
                                 (i12 / args.r2) * args.nb02 +
                                 (i13 / args.r3) * args.nb03;
        ag[row] = (device const block_q8_0 *)((device const char *)src0_gate + offset0);
        au[row] = (device const block_q8_0 *)((device const char *)src0_up   + offset0);
    }

    float sumg[NR0] = { 0.f };
    float sumu[NR0] = { 0.f };

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const int8_t *qg = ag[row][ib].qs + il * NQ;
            device const int8_t *qu = au[row][ib].qs + il * NQ;

            float sg = 0.f;
            float su = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sg += qg[i] * yl[i];
                su += qu[i] * yl[i];
            }

            sumg[row] += sg * ag[row][ib].d;
            sumu[row] += su * au[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32 = (threadgroup float *)shmem;
    threadgroup float *sh_gate[NR0];
    threadgroup float *sh_up[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sh_gate[row] = shmem_f32 + NW * row;
        sh_up[row]   = shmem_f32 + NW * (NR0 + row);
        if (sgitg == 0) {
            sh_gate[row][tiisg] = 0.0f;
            sh_up[row][tiisg] = 0.0f;
        }
        sumg[row] = simd_sum(sumg[row]);
        sumu[row] = simd_sum(sumu[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            sh_gate[row][sgitg] = sumg[row];
            sh_up[row][sgitg] = sumu[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *gate_f32 = (device float *)dst_gate +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    device float *up_f32 = (device float *)dst_up +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    device float *mid_f32 = (device float *)dst_mid +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    FOR_UNROLL (short row = 0; row < NR0 && r0 + row < args.ne01; ++row) {
        const float gate = simd_sum(sh_gate[row][tiisg]);
        const float up = simd_sum(sh_up[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            const uint out_row = r0 + row;
            gate_f32[out_row] = gate;
            up_f32[out_row] = up;
            float g = gate;
            float u = up;
            if (clamp_value > 1.0e-6f) {
                g = min(g, clamp_value);
                u = clamp(u, -clamp_value, clamp_value);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u;
        }
    }
}

template<typename T0, typename T1, short NR0, typename args_t>
void kernel_mul_mv_t_t_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB = 32;
    constexpr short NF = 8;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const T1 * y = (device const T1 *) (src1 + offset1);

    device const T0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const T0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T1 yl[NF];

    device const T1 * yb = y + (ib0*NB + il*NF);

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T0 * xb = ax[row] + (ib*NB + il*NF);

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF; ++i) {
                sumq += xb[i] * yl[i];
            }

            sumf[row] += sumq;
        }

        yb += NSG*NF*NW;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 2: kernel_mul_mv_t_t_impl<T0, T1, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 4: kernel_mul_mv_t_t_impl<T0, T1, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    }
}

// Decode-time dense F32/F16 matrix-vector multiply. The instantiated kernels
// handle unquantized DS4 weights and activations that are already float rows.
template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_disp<T0, T1, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t<half, half>) mul_mv_t_t;

// Host-visible dense matvec variants used by the graph for F32 and F16 weights.
template [[host_name("kernel_mul_mv_f32_f32")]] kernel mul_mv_t_t kernel_mul_mv_t_t<float, float>;
template [[host_name("kernel_mul_mv_f16_f32")]] kernel mul_mv_t_t kernel_mul_mv_t_t<half,  float>;

template<typename T0, typename T04, typename T1, typename T14, short NR0, typename args_t>
void kernel_mul_mv_t_t_4_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const T1  * y  = (device const T1  *) (src1 + offset1);
    device const T14 * y4 = (device const T14 *) (src1 + offset1);

    device const T0  * ax [NR0];
    device const T04 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax [row] = (device const T0  *) ((device char *) src0 + offset0);
        ax4[row] = (device const T04 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T14 yl4[NF4];

    device const T14 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T04 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), float4(yl4[i]));
            }

            sumf[row] += sumq;
        }

        yb4 += NSG*NF*NW/4;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T04, typename T1, typename T14, typename args_t>
void kernel_mul_mv_t_t_4_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 2: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 4: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    };
}

// Vectorized dense matvec using float4/half4 loads. DS4 uses this where the
// inner dimension and alignment make vector loads cheaper than scalar lanes.
template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv_t_t_4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_4_disp<T0, T04, T1, T14, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t_4<half, half4, half, half4>) mul_mv_t_t_4;

// Host-visible vectorized dense matvec variants for F32 and F16 weights.
template [[host_name("kernel_mul_mv_f32_f32_4")]] kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<float, float4, float, float4>;
template [[host_name("kernel_mul_mv_f16_f32_4")]] kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<half,  half4,  float, float4>;

// DS4 compressor projections always compute two same-shaped F16 matvecs from
// the same normalized activation: one for projected KV and one for pooling
// scores.  This paired variant keeps the exact dense F16 row-reduction shape
// for each matrix, but shares one dispatch and one activation stream.
template<short NR0, typename args_t>
void kernel_mul_mv_f16_f32_pair_4_impl(
        args_t args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float  * y  = (device const float  *) (src1 + offset1);
    device const float4 * y4 = (device const float4 *) (src1 + offset1);

    device const half  * ax_a [NR0];
    device const half4 * ax4_a[NR0];
    device const half  * ax_b [NR0];
    device const half4 * ax4_b[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax_a [row] = (device const half  *) ((device char *) src0_a + offset0);
        ax4_a[row] = (device const half4 *) ((device char *) src0_a + offset0);
        ax_b [row] = (device const half  *) ((device char *) src0_b + offset0);
        ax4_b[row] = (device const half4 *) ((device char *) src0_b + offset0);
    }

    float sum_a[NR0] = { 0.f };
    float sum_b[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    float4 yl4[NF4];

    device const float4 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const half4 * xb4_a = ax4_a[row] + (ib*NB + il*NF)/4;
            device const half4 * xb4_b = ax4_b[row] + (ib*NB + il*NF)/4;

            float suma = 0.f;
            float sumb = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                const float4 yv = float4(yl4[i]);
                suma += dot(float4(xb4_a[i]), yv);
                sumb += dot(float4(xb4_b[i]), yv);
            }

            sum_a[row] += suma;
            sum_b[row] += sumb;
        }

        yb4 += NSG*NF*NW/4;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            const float yi = y[i];
            sum_a[row] += ax_a[row][i] * yi;
            sum_b[row] += ax_b[row][i] * yi;
        }
    }

    device float * dst_a_f32 = (device float *) dst_a + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    device float * dst_b_f32 = (device float *) dst_b + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_a_f32, sum_a, r0, args.ne01, tiisg, sgitg, shmem);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    helper_mv_reduce_and_write<NR0>(dst_b_f32, sum_b, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename args_t>
void kernel_mul_mv_f16_f32_pair_4_disp(
        args_t args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 2: kernel_mul_mv_f16_f32_pair_4_impl<2>(args, src0_a, src0_b, src1, dst_a, dst_b, shmem, tgpig, tiisg, sgitg); break;
        case 4: kernel_mul_mv_f16_f32_pair_4_impl<4>(args, src0_a, src0_b, src1, dst_a, dst_b, shmem, tgpig, tiisg, sgitg); break;
    }
}

kernel void kernel_mul_mv_f16_f32_pair_4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_f16_f32_pair_4_disp<constant ds4_metal_args_mul_mv &>(
            args, src0_a, src0_b, src1, dst_a, dst_b, shmem, tgpig, tiisg, sgitg);
}

template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_short_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg) {
    const int r0 = tgpig.x*32 + tiisg;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    if (r0 >= args.ne01) {
        return;
    }

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const T0 * x = (device const T0 *) (src0 + offset0);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const T1 * y = (device const T1 *) (src1 + offset1);

    float res = 0.0f;

    for (int i = 0; i < args.ne00; ++i) {
        res += (float) x[i] * (float) y[i];
    }

    dst_f32[(uint64_t)r1*args.ne0 + r0] = res;
}

// Scalar fallback for short rows. It trades parallelism for lower dispatch and
// reduction overhead when DS4 asks for tiny dense matvecs.
template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t_short(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_t_t_short_impl<T0, T1, constant ds4_metal_args_mul_mv &>(
        args,
        src0,
        src1,
        dst,
        tgpig,
        tiisg);
}

typedef decltype(kernel_mul_mv_t_t_short<half, half>) mul_mv_t_t_short_t;

// Host-visible short-row dense matvec variants.
template [[host_name("kernel_mul_mv_f32_f32_short")]] kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<float, float>;
template [[host_name("kernel_mul_mv_f16_f32_short")]] kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<half,  float>;

template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    for (int i = 0; i < 4; i++) {
        reg[i] = (qs[4*(il%4) + i + 16*(il/4)] * d);
    }
}

// DS4 small-batch mat-vec kernel used for 2..8 prompt tokens.
template<short r1ptg, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &) >
void kernel_mul_mv_ext_q4_f32_impl(
        constant ds4_metal_args_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 4; // chunks per thread

    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%args.ne12;
    const int i13 = i1m/args.ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb; // current chunk index

    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    // reduce only the threads in each row
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// Small-batch prompt matvec for 2..5 tokens. It bridges decode-style matvec and
// full matmul when DS4 prefill chunks are too small to amortize matrix tiles.
template<short r1ptg, typename q_t, short epb, void (*deq_t4)(device const q_t *, short, thread float4 &)>
kernel void kernel_mul_mv_ext_q4_f32_disp(
        constant ds4_metal_args_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4_f32_impl<r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0, 32, dequantize_q8_0_t4>) mul_mv_ext_q4_f32_t;

// Host-visible small-batch variants. DS4 currently needs F16 and Q8_0 weights
// for r1=2..5 during the prompt path.
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_2")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, half4,      4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_3")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, half4,      4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_4")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, half4,      4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_5")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, half4,      4,  dequantize_f16_t4>;

template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q8_0, 32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q8_0, 32, dequantize_q8_0_t4>;

constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];

#ifdef DS4_METAL_HAS_TENSOR
template<
    short NR0, short NR1,
    typename SA, typename SA_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1>
kernel void kernel_mul_mm_mpp(
        constant ds4_metal_args_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;
    const int im = tgpig.z;
    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    threadgroup SA *sa = (threadgroup SA *)shmem;
    threadgroup SA *sb = sa + NR0*NK;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(sb, dextents<int32_t, 2>(NK, NR1));

    device const T1 *ptrB = (device const T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11/sizeof(T1);

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (!FC_mul_mm_bc_out || r0 + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    device const T0 *row_ptr = (device const T0 *)(srcA + args.nb01*(r0 + row) + offset0);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? (SA)row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos/(16*nl);
                    const short il = (k_pos/16)%nl;
                    device const block_q *row_ptr = (device const block_q *)(srcA + args.nb01*(r0 + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (SA)0;
                }
            }
        }
        for (int work = tiitg; work < NK*NR1; work += NUM_THREADS) {
            const int col = work/NK;
            const int k = work%NK;
            if ((!FC_mul_mm_bc_out && !FC_mul_mm_bc_inp) ||
                (r1 + col < N && loop_k + k < K)) {
                sb[col*NK + k] = (SA)ptrB[(uint64_t)(r1 + col)*strideB + loop_k + k];
            } else {
                sb[col*NK + k] = (SA)0;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(0, 0);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_batch = (device float *)dst + im*N*M;
    if (!FC_mul_mm_bc_out) {
        device float *dst_tile = dst_batch + r0 + (uint64_t)r1*M;
        auto tD = tensor(dst_tile, dextents<int32_t, 2>(NR0, NR1), array<int, 2>({1, M}));
        cT.store(tD);
    } else {
        auto tD = tensor(dst_batch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
        auto mD = tD.slice(r0, r1);
        cT.store(mD);
    }
}

typedef decltype(kernel_mul_mm_mpp<64, 32, half, half4x4, float4x4, 1, dequantize_f32, float, float4x4, float>) mul_mm_mpp_t;

template [[host_name("kernel_mul_mm_f16_f32_mpp")]]  kernel mul_mm_mpp_t kernel_mul_mm_mpp<64, 32, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;

// Retained Metal4/TensorOps dense prefill kernel.  The legacy MPP prototype
// staged both operands in threadgroup memory; this version stages only the
// model weight tile and lets MPP read the dense RHS activation matrix directly
// from device memory.  That direct-RHS shape was the clear win for DS4's large
// aligned F16/Q8_0 prompt matmuls.  The host selects the widest token tile that
// evenly divides the batch, with 128-token tiles retained after the 64-token
// retest was neutral or slower.
template<
    short NR1,
    typename SA, typename SA_4x4, typename block_q, short nl,
    void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1>
kernel void kernel_mul_mm_mpp_direct_rhs(
        constant ds4_metal_args_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NR0 = 64;
    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;
    const int im = tgpig.z;
    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    threadgroup SA *sa = (threadgroup SA *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));

    device T1 *ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11/sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (r0 + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    device const T0 *row_ptr = (device const T0 *)(srcA + args.nb01*(r0 + row) + offset0);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? (SA)row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos/(16*nl);
                    const short il = (k_pos/16)%nl;
                    device const block_q *row_ptr = (device const block_q *)(srcA + args.nb01*(r0 + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);
                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row*NK + k_base + i] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (SA)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_batch = (device float *)dst + im*N*M;
    auto tD = tensor(dst_batch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
    auto mD = tD.slice(r0, r1);
    cT.store(mD);
}

typedef decltype(kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, float4x4, 1, dequantize_f32, float, float4x4, float>) mul_mm_mpp_direct_rhs_t;

template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs_n64")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<64, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_f16_f32_mpp_direct_rhs_n128")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<128, half, half4x4, half4x4, 1, dequantize_f16,  half,  half4x4,  float>;
template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<32, half, half4x4, block_q8_0, 2, dequantize_q8_0, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs_n64")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<64, half, half4x4, block_q8_0, 2, dequantize_q8_0, float, float4x4, float>;
template [[host_name("kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<128, half, half4x4, block_q8_0, 2, dequantize_q8_0, float, float4x4, float>;
#endif

// Tiled matrix-matrix kernel used for prompt batches larger than 8. DS4 uses
// this to turn prefill into large simdgroup matrix operations; each block_q
// contains 16*nl weights.
template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ds4_metal_args_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // Pointer-form store avoids a slower address-lowering path in
                // current Apple Metal compilers for this dequantized tile write.
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // if no bounds checks on the output are needed, we can directly write to device memory
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) + \
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

kernel void kernel_mul_mm_f16_f32_pair(
        constant ds4_metal_args_mul_mm & args,
        device const char * src0_a,
        device const char * src0_b,
        device const char * src1,
        device       char * dst_a,
        device       char * dst_b,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_a = (threadgroup half *)(shmem);
    threadgroup half * sa_b = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il0;

    device const half4x4 * xa = (device const half4x4 *)(src0_a + args.nb01*(r0 + lr0) + offset0) + offset1;
    device const half4x4 * xb = (device const half4x4 *)(src0_b + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];

    simdgroup_float8x8 mc_a[8];
    simdgroup_float8x8 mc_b[8];

    for (short i = 0; i < 8; i++) {
        mc_a[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        mc_b[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        half4x4 temp_a;
        half4x4 temp_b;
        dequantize_f16(xa, il, temp_a);
        dequantize_f16(xb, il, temp_b);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;

            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;

            const short ib = 8*sx + sy;

            *(sa_a + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            *(sa_b + 64*ib + 8*ly + lx) = temp_b[i/4][i%4];
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (half) *((device float *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = (half2x4)(*((device float2x4 *) y));
        }

        il = (il + 2 < 1) ? il + 2 : il % 2;
        xa = (il < 2) ? xa + 2 : xa;
        xb = (il < 2) ? xb + 2 : xb;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_a = (sa_a + 4*64*(sgitg%2));
        threadgroup const half * lsma_b = (sa_b + 4*64*(sgitg%2));
        threadgroup const half * lsmb   = (sb   + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_a + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_a[i], mb[i/4], ma[i%4], mc_a[i]);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_b + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_b[i], mb[i/4], ma[i%4], mc_b[i]);
            }

            lsma_a += 8*64;
            lsma_b += 8*64;
            lsmb   += 4*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        device float * C_a = (device float *) dst_a +
            (r0 + 32*(sgitg &  1)) +
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;
        device float * C_b = (device float *) dst_b +
            (r0 + 32*(sgitg &  1)) +
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc_a[i], C_a + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
            simdgroup_store(mc_b[i], C_b + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = (threadgroup float *) shmem;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc_a[i],
                            temp_str + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0 + 8*(i%4) + 8*NR0*(i/4),
                            NR0,
                            0,
                            false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float *) dst_a + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc_b[i],
                            temp_str + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0 + 8*(i%4) + 8*NR0*(i/4),
                            NR0,
                            0,
                            false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float *) dst_b + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_t;

// Host-visible prefill matmul variants for F16 and Q8_0 weights.
template [[host_name("kernel_mul_mm_f16_f32")]]  kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, half4x4, 1, dequantize_f16,  half,  half4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32")]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, float, float2x4>;
"""###,
        "moe": ###"""
// DS4 Metal routed-MoE matvec kernels.

#define QK_K 256
#define N_R0_Q2_K 4
#define N_R0_Q4_K 2
#define N_R0_IQ2_XXS 4

static constant uchar ds4_metal_kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static constant uchar ds4_metal_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static constant ulong ds4_metal_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

#define kmask_iq2xs ds4_metal_kmask_iq2xs
#define ksigns_iq2xs ds4_metal_ksigns_iq2xs
#define iq2xxs_grid ds4_metal_iq2xxs_grid

struct block_q2_K {
    uchar scales[QK_K/16];
    uchar qs[QK_K/4];
    half d;
    half dmin;
};

struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[QK_K/2];
};

struct block_iq2_xxs {
    half d;
    ushort qs[QK_K/8];
};

struct ds4_metal_dsv4_moe_swiglu_weight_args {
    uint32_t width;
    uint32_t rows;
    uint64_t gate_row_stride;
    uint64_t up_row_stride;
    uint64_t mid_row_stride;
    uint64_t weight_stride;
    uint32_t write_clamped;
    float clamp_value;
};

struct ds4_metal_dsv4_moe_sum6_args {
    uint32_t width;
    uint32_t tokens;
    uint64_t src_token_stride;
    uint64_t dst_token_stride;
};

// Routed-MoE activation for the selected experts:
// clamp(gate), clamp(up), silu(gate) * up * route_weight.  Normal inference
// does not consume gate/up after this point, so the fast path avoids writing the
// clamped intermediates back.  A diagnostic env switch can restore those writes
// when comparing the old multi-kernel intermediate tensors.
kernel void kernel_dsv4_moe_swiglu_weight(
        constant ds4_metal_dsv4_moe_swiglu_weight_args &args,
        device char *gate,
        device char *up,
        device char *mid,
        device const char *weights,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;

    device float *gate_row = (device float *)(gate + (uint64_t)row * args.gate_row_stride);
    device float *up_row   = (device float *)(up   + (uint64_t)row * args.up_row_stride);
    device float *mid_row  = (device float *)(mid  + (uint64_t)row * args.mid_row_stride);
    device const float *w  = (device const float *)(weights + (uint64_t)row * args.weight_stride);
    const float route_weight = w[0];
    const float c = args.clamp_value;

    for (uint i = tid; i < args.width; i += ntg) {
        float g = gate_row[i];
        float u = up_row[i];
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
            if (args.write_clamped != 0) {
                gate_row[i] = g;
                up_row[i] = u;
            }
        }
        const float silu = g / (1.0f + exp(-g));
        mid_row[i] = silu * u * route_weight;
    }
}

// Same routed-MoE activation as above, but stores the down-projection input in
// half precision. The grouped matmul path converts F32 activations to half
// before MMA anyway, so this cuts the large mid write/read traffic without
// changing the effective matmul input precision.
kernel void kernel_dsv4_moe_swiglu_weight_f16(
        constant ds4_metal_dsv4_moe_swiglu_weight_args &args,
        device char *gate,
        device char *up,
        device char *mid,
        device const char *weights,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;

    device float *gate_row = (device float *)(gate + (uint64_t)row * args.gate_row_stride);
    device float *up_row   = (device float *)(up   + (uint64_t)row * args.up_row_stride);
    device half  *mid_row  = (device half  *)(mid  + (uint64_t)row * args.mid_row_stride);
    device const float *w  = (device const float *)(weights + (uint64_t)row * args.weight_stride);
    const float route_weight = w[0];
    const float c = args.clamp_value;

    for (uint i = tid; i < args.width; i += ntg) {
        float g = gate_row[i];
        float u = up_row[i];
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
            if (args.write_clamped != 0) {
                gate_row[i] = g;
                up_row[i] = u;
            }
        }
        const float silu = g / (1.0f + exp(-g));
        mid_row[i] = (half)(silu * u * route_weight);
    }
}

kernel void kernel_dsv4_moe_sum6_f32(
        constant ds4_metal_dsv4_moe_sum6_args &args,
        device const char *src,
        device       char *dst,
        uint token[[threadgroup_position_in_grid]],
        uint tid[[thread_position_in_threadgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    if (token >= args.tokens) return;

    device const float *s =
        (device const float *)(src + (uint64_t)token * args.src_token_stride);
    device float *d =
        (device float *)(dst + (uint64_t)token * args.dst_token_stride);

    for (uint col = tid; col < args.width; col += ntg) {
        float v = s[col];
        v += s[args.width + col];
        v += s[2u * args.width + col];
        v += s[3u * args.width + col];
        v += s[4u * args.width + col];
        v += s[5u * args.width + col];
        d[col] = v;
    }
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 &reg) {
    device const uchar *q = xb->qs;

    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

struct ds4_metal_args_mul_mv_id {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
};

struct ds4_metal_moe_expert_group_args {
    uint32_t expert_base;
    uint32_t expert_count;
    uint32_t accumulate;
    uint32_t pad0;
};

struct ds4_metal_q4_gather_slots6_args {
    uint64_t expert_bytes;
    uint32_t group_size;
    uint32_t n_slots;
};

struct ds4_metal_q4_expert_table {
    array<device const char *, 384> experts [[id(0)]];
};

struct ds4_metal_expert_address_table {
    device const uint64_t *addrs;
};

struct ds4_metal_stream_expert_validate_args {
    uint32_t n_total_expert;
    uint32_t n_expert;
};

struct ds4_metal_stream_expert_split_args {
    uint32_t active_mask;
    uint32_t accumulate;
};

struct ds4_metal_args_mul_mm_id_map0 {
    int32_t  ne02;
    int32_t  ne10;
    int32_t  ne11;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21;
    int32_t  ne20;
    uint64_t nb21;
};

struct ds4_metal_args_mul_mm_id {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
};

template<int nr0, typename args_t>
void kernel_mul_mv_q2_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3
    const short is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += args.nb01/2;
            sc += args.nb01;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_q4_K_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q4_K *x = (device const block_q4_K *)(src0 + offset0);
    device const float *y = (device const float *)(src1 + offset1);

    float yl[16];
    float yh[16];
    float sumf[nr0] = {0.f};

    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half *dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t *q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                  (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01 / 2;
            sc += args.nb01 / 2;
            dh += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    device float *dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }

    (void)shmem;
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ds4_metal_ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += args.nb01/2;
            q2 += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

template<int nr0>
void kernel_mul_mv_iq2_xxs_pair_f32_impl(
        ds4_metal_args_mul_mv args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * xg = (device const block_iq2_xxs *) (src0_gate + offset0);
    device const block_iq2_xxs * xu = (device const block_iq2_xxs *) (src0_up   + offset0);
    device const float         * y  = (device const float         *) (src1      + offset1);

    float yl[32];
    float sumg[nr0]={0.f};
    float sumu[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ds4_metal_ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xgr = xg + ibl;
        device const block_iq2_xxs * xur = xu + ibl;
        device const uint16_t * qg = xgr->qs + 4 * ib;
        device const uint16_t * qu = xur->qs + 4 * ib;
        device const half * dhg = &xgr->d;
        device const half * dhu = &xur->d;

        for (short row = 0; row < nr0; row++) {
            device const uint8_t * aux8g = (device const uint8_t *)qg;
            device const uint8_t * aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));

            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t * gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint8_t signg = ssigns[(aux32g >> 7*l) & 127];
                const uint8_t signu = ssigns[(aux32u >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8*l + j];
                    sg += v * gridg[j] * (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;

            dhg += args.nb01/2;
            dhu += args.nb01/2;
            qg  += args.nb01/2;
            qu  += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_gate_f32 = (device float *) dst_gate + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    device float * dst_up_f32   = (device float *) dst_up   + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_gate = simd_sum(sumg[row]);
        const float sum_up   = simd_sum(sumu[row]);
        if (tiisg == 0) {
            dst_gate_f32[first_row + row] = sum_gate * 0.25f;
            dst_up_f32[first_row + row]   = sum_up   * 0.25f;
        }
    }
}

typedef void (kernel_mul_mv2_disp_t)(
        ds4_metal_args_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg);

template<kernel_mul_mv2_disp_t disp_fn>
void mmv_fn(
        ds4_metal_args_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    disp_fn(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>) mul_mv_id_disp_fn_t;

// Decode-time expert matvec. The ids tensor selects the routed expert for each
// slot, then this wrapper invokes the quantized row kernel for Q8_0, Q2_K, or
// IQ2_XXS weights without materializing per-expert dispatches on the CPU.
template<mul_mv_id_disp_fn_t disp_fn>
kernel void kernel_mul_mv_id(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    (void)tiitg;

    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device const char * src0_cur = src0s + i02*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;

    device char * dst_cur = dst + (i1*args.ne0 + i2*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    disp_fn(
        args0,
        /* src0 */ src0_cur,
        /* src1 */ src1_cur,
        /* dst  */ dst_cur,
        shmem,
        tgpig,
        tiitg,
        tiisg,
        sgitg);
}

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>>) kernel_mul_mv_id_q_t;
typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>>>) kernel_mul_mv_id_q8_0_t;

// Host-visible decode MoE matvec variants for the DS4 quant formats.
template [[host_name("kernel_mul_mv_id_q8_0_f32")]]    kernel kernel_mul_mv_id_q8_0_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>>>;
template [[host_name("kernel_mul_mv_id_q2_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>>>;
template [[host_name("kernel_mul_mv_id_iq2_xxs_f32")]] kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS>>>;

// DS4 attention output low projection, specialized for the fixed block
// diagonal mapping used by the model:
//
//     low[token, group, rank] = heads[token, group, :] * Woa[group, rank, :]
//
// The generic GGML-style id matvec supports arbitrary routed expert ids.  Here
// the id is always equal to the group number, so this wrapper keeps the exact
// Q8_0 dot kernel but removes the id-buffer load and the CPU-side id table.
kernel void kernel_dsv4_attn_out_low_q8_0_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char * src0_cur = src0s + idx*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;
    device       char * dst_cur  = dst   + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, thread ds4_metal_args_mul_mv &>(
        args0,
        src0_cur,
        src1_cur,
        dst_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

kernel void kernel_mul_mv_id_iq2_xxs_pair_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char * src0_gate_cur = src0_gate + i02*args.nb02;
    device const char * src0_up_cur   = src0_up   + i02*args.nb02;
    device const char * src1_cur      = src1      + i11*args.nb11 + i12*args.nb12;

    device char * dst_gate_cur = dst_gate + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);
    device char * dst_up_cur   = dst_up   + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    (void)tiitg;
    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

// Decode-only routed expert gate/up projection fused with the DS4 activation:
//
//     mid = silu(clamp(gate)) * clamp(up) * route_weight
//
// The quantized dot products are intentionally the same IQ2_XXS paired path as
// `kernel_mul_mv_id_iq2_xxs_pair_f32`.  The only extra work is done by lane 0
// after each exact reduced row has been produced.  This removes the separate
// routed activation dispatch and avoids rereading the gate/up rows before the
// down projection.  The host uses this only for the normal release path where
// diagnostics do not request clamped gate/up intermediates.
kernel void kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    const int nb32 = nb * (QK_K / 32);

    device const block_iq2_xxs *xg =
        (device const block_iq2_xxs *)(src0_gate + i02 * args.nb02 + (uint64_t)first_row * args.nb01);
    device const block_iq2_xxs *xu =
        (device const block_iq2_xxs *)(src0_up + i02 * args.nb02 + (uint64_t)first_row * args.nb01);
    device const float *y =
        (device const float *)(src1 + i11 * args.nb11 + i12 * args.nb12);

    float yl[32];
    float sumg[N_R0_IQ2_XXS] = {0.f};
    float sumu[N_R0_IQ2_XXS] = {0.f};

    threadgroup uint64_t *svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  *ssigns  = (threadgroup uint8_t *)(svalues + 256);
    {
        int nval = 4;
        int pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ds4_metal_ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float *y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs *xgr = xg + ibl;
        device const block_iq2_xxs *xur = xu + ibl;
        device const uint16_t *qg = xgr->qs + 4 * ib;
        device const uint16_t *qu = xur->qs + 4 * ib;
        device const half *dhg = &xgr->d;
        device const half *dhu = &xur->d;

        for (short row = 0; row < N_R0_IQ2_XXS; row++) {
            device const uint8_t *aux8g = (device const uint8_t *)qg;
            device const uint8_t *aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));

            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t *gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t *gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint8_t signg = ssigns[(aux32g >> 7 * l) & 127];
                const uint8_t signu = ssigns[(aux32u >> 7 * l) & 127];
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8 * l + j];
                    sg += v * gridg[j] * (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;

            dhg += args.nb01 / 2;
            dhu += args.nb01 / 2;
            qg  += args.nb01 / 2;
            qu  += args.nb01 / 2;
        }

        y4 += 32 * 32;
    }

    device float *dst_gate_f32 =
        (device float *)dst_gate + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    device float *dst_up_f32 =
        (device float *)dst_up + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *dst_mid_f32 =
        (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w =
        (device const float *)(weights + pair_row * act.weight_stride);

    const float c = act.clamp_value;
    const float route_weight = route_w[0];
    for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
        const float sum_gate = simd_sum(sumg[row]);
        const float sum_up   = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + row;
            const float gate = sum_gate * 0.25f;
            const float up = sum_up * 0.25f;
            float g = gate;
            float u = up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            dst_gate_f32[out_row] = gate;
            dst_up_f32[out_row] = up;
            const float silu = g / (1.0f + exp(-g));
            dst_mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate0,
        device const char * src0_gate1,
        device const char * src0_gate2,
        device const char * src0_gate3,
        device const char * src0_gate4,
        device const char * src0_gate5,
        device const char * src0_up0,
        device const char * src0_up1,
        device const char * src0_up2,
        device const char * src0_up3,
        device const char * src0_up4,
        device const char * src0_up5,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate0;
    device const char *src0_up_cur = src0_up0;
    switch (idx) {
    case 1: src0_gate_cur = src0_gate1; src0_up_cur = src0_up1; break;
    case 2: src0_gate_cur = src0_gate2; src0_up_cur = src0_up2; break;
    case 3: src0_gate_cur = src0_gate3; src0_up_cur = src0_up3; break;
    case 4: src0_gate_cur = src0_gate4; src0_up_cur = src0_up4; break;
    case 5: src0_gate_cur = src0_gate5; src0_up_cur = src0_up5; break;
    default: break;
    }

    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const uint64_t * gate_addrs,
        device const uint64_t * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (i02 < 0 || i02 >= args.ne02 || i02 >= 384) {
        return;
    }
    const uint64_t gate_addr = gate_addrs[(uint)i02];
    const uint64_t up_addr = up_addrs[(uint)i02];
    if (gate_addr == 0 || up_addr == 0) {
        return;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur =
        reinterpret_cast<device const char *>(gate_addr);
    device const char *src0_up_cur =
        reinterpret_cast<device const char *>(up_addr);
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_masked_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        constant ds4_metal_stream_expert_split_args & split,
        device const uint64_t * gate_addrs,
        device const uint64_t * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;
    if ((split.active_mask & (1u << (uint)idx)) == 0) {
        return;
    }

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (i02 < 0 || i02 >= args.ne02 || i02 >= 384) {
        return;
    }
    const uint64_t gate_addr = gate_addrs[(uint)i02];
    const uint64_t up_addr = up_addrs[(uint)i02];
    if (gate_addr == 0 || up_addr == 0) {
        return;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur =
        reinterpret_cast<device const char *>(gate_addr);
    device const char *src0_up_cur =
        reinterpret_cast<device const char *>(up_addr);
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_stream_expert_cache_validate(
        constant ds4_metal_stream_expert_validate_args & args,
        device const char * ids,
        device const uint64_t * gate_addrs,
        device const uint64_t * up_addrs,
        device const uint64_t * down_addrs,
        device uint32_t * status,
        uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;

    uint32_t miss_mask = 0;
    uint32_t invalid_mask = 0;
    const uint32_t n_expert = min(args.n_expert, (uint32_t)6);
    device const int32_t *selected = (device const int32_t *)ids;

    status[3] = n_expert;
    for (uint32_t i = 0; i < 6; i++) {
        const int32_t expert = i < n_expert ? selected[i] : -1;
        status[4 + i] = as_type<uint32_t>(expert);
        if (i >= n_expert) continue;
        if (expert < 0 ||
            (uint32_t)expert >= args.n_total_expert ||
            (uint32_t)expert >= 384) {
            invalid_mask |= (1u << i);
            continue;
        }
        const uint32_t e = (uint32_t)expert;
        if (gate_addrs[e] == 0 || up_addrs[e] == 0 || down_addrs[e] == 0) {
            miss_mask |= (1u << i);
        }
    }

    status[0] = (miss_mask == 0 && invalid_mask == 0) ? 1u : 0u;
    status[1] = miss_mask;
    status[2] = invalid_mask;
}

kernel void kernel_mul_mv_id_q4_K_pair_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate + i02 * args.nb02;
    device const char *src0_up_cur   = src0_up   + i02 * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    (void)tiitg;
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

// Same release-path fusion as the IQ2_XXS kernel above for the Q4_K expert
// variant.  The Q4 pair path reuses the existing exact matvec implementation
// for gate and up, then the same lane that wrote each row derives the routed
// SwiGLU input.  This keeps Q4 behavior aligned with the Q2 optimization while
// preserving the old pair projection arithmetic.
kernel void kernel_mul_mv_id_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate + i02 * args.nb02;
    device const char *src0_up_cur   = src0_up   + i02 * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    const short NSG = FC_mul_mv_nsg;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    device const block_q4_K *xg =
        (device const block_q4_K *)(src0_gate_cur + (uint64_t)first_row * args.nb01);
    device const block_q4_K *xu =
        (device const block_q4_K *)(src0_up_cur + (uint64_t)first_row * args.nb01);
    device const float *y = (device const float *)src1_cur;
    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    float sumg[N_R0_Q4_K] = {0.f};
    float sumu[N_R0_Q4_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[16];
        float yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *scg = (device const uint16_t *)xg[ib].scales + iq;
        device const uint16_t *qg1 = (device const uint16_t *)xg[ib].qs + 16 * iq + 4 * ir;
        device const half *dhg = &xg[ib].d;
        device const uint16_t *scu = (device const uint16_t *)xu[ib].scales + iq;
        device const uint16_t *qu1 = (device const uint16_t *)xu[ib].qs + 16 * iq + 4 * ir;
        device const half *dhu = &xu[ib].d;

        for (short row = 0; row < N_R0_Q4_K; row++) {
            sc16[0] = scg[0] & kmask1;
            sc16[1] = scg[2] & kmask1;
            sc16[2] = ((scg[4] >> 0) & kmask2) | ((scg[0] & kmask3) >> 2);
            sc16[3] = ((scg[4] >> 4) & kmask2) | ((scg[2] & kmask3) >> 2);

            device const uint16_t *qg2 = qg1 + 32;
            float4 acc1g = {0.f, 0.f, 0.f, 0.f};
            float4 acc2g = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1g[0] += yl[2 * i + 0] * (qg1[i] & 0x000F);
                acc1g[1] += yl[2 * i + 1] * (qg1[i] & 0x0F00);
                acc1g[2] += yl[2 * i + 8] * (qg1[i] & 0x00F0);
                acc1g[3] += yl[2 * i + 9] * (qg1[i] & 0xF000);
                acc2g[0] += yh[2 * i + 0] * (qg2[i] & 0x000F);
                acc2g[1] += yh[2 * i + 1] * (qg2[i] & 0x0F00);
                acc2g[2] += yh[2 * i + 8] * (qg2[i] & 0x00F0);
                acc2g[3] += yh[2 * i + 9] * (qg2[i] & 0xF000);
            }

            sumg[row] += dhg[0] * ((acc1g[0] + 1.f / 256.f * acc1g[1]) * sc8[0] +
                                   (acc1g[2] + 1.f / 256.f * acc1g[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2g[0] + 1.f / 256.f * acc2g[1]) * sc8[4] +
                                   (acc2g[2] + 1.f / 256.f * acc2g[3]) * sc8[5] * 1.f / 16.f) -
                         dhg[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            sc16[0] = scu[0] & kmask1;
            sc16[1] = scu[2] & kmask1;
            sc16[2] = ((scu[4] >> 0) & kmask2) | ((scu[0] & kmask3) >> 2);
            sc16[3] = ((scu[4] >> 4) & kmask2) | ((scu[2] & kmask3) >> 2);

            device const uint16_t *qu2 = qu1 + 32;
            float4 acc1u = {0.f, 0.f, 0.f, 0.f};
            float4 acc2u = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1u[0] += yl[2 * i + 0] * (qu1[i] & 0x000F);
                acc1u[1] += yl[2 * i + 1] * (qu1[i] & 0x0F00);
                acc1u[2] += yl[2 * i + 8] * (qu1[i] & 0x00F0);
                acc1u[3] += yl[2 * i + 9] * (qu1[i] & 0xF000);
                acc2u[0] += yh[2 * i + 0] * (qu2[i] & 0x000F);
                acc2u[1] += yh[2 * i + 1] * (qu2[i] & 0x0F00);
                acc2u[2] += yh[2 * i + 8] * (qu2[i] & 0x00F0);
                acc2u[3] += yh[2 * i + 9] * (qu2[i] & 0xF000);
            }

            sumu[row] += dhu[0] * ((acc1u[0] + 1.f / 256.f * acc1u[1]) * sc8[0] +
                                   (acc1u[2] + 1.f / 256.f * acc1u[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2u[0] + 1.f / 256.f * acc2u[1]) * sc8[4] +
                                   (acc2u[2] + 1.f / 256.f * acc2u[3]) * sc8[5] * 1.f / 16.f) -
                         dhu[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            qg1 += args.nb01 / 2;
            scg += args.nb01 / 2;
            dhg += args.nb01 / 2;
            qu1 += args.nb01 / 2;
            scu += args.nb01 / 2;
            dhu += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
        const float gate = simd_sum(sumg[row]);
        const float up = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + row;
            float g = gate;
            float u = up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            gate_f32[out_row] = gate;
            up_f32[out_row] = up;
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_table_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ds4_metal_q4_expert_table & gate_table,
        device const ds4_metal_q4_expert_table & up_table,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (i02 < 0 || i02 >= args.ne02 || i02 >= 384) {
        return;
    }
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = gate_table.experts[(uint)i02];
    device const char *src0_up_cur   = up_table.experts[(uint)i02];
    device const char *src1_cur      = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    const short NSG = FC_mul_mv_nsg;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    device const block_q4_K *xg =
        (device const block_q4_K *)(src0_gate_cur + (uint64_t)first_row * args.nb01);
    device const block_q4_K *xu =
        (device const block_q4_K *)(src0_up_cur + (uint64_t)first_row * args.nb01);
    device const float *y = (device const float *)src1_cur;
    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    float sumg[N_R0_Q4_K] = {0.f};
    float sumu[N_R0_Q4_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[16];
        float yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *scg = (device const uint16_t *)xg[ib].scales + iq;
        device const uint16_t *qg1 = (device const uint16_t *)xg[ib].qs + 16 * iq + 4 * ir;
        device const half *dhg = &xg[ib].d;
        device const uint16_t *scu = (device const uint16_t *)xu[ib].scales + iq;
        device const uint16_t *qu1 = (device const uint16_t *)xu[ib].qs + 16 * iq + 4 * ir;
        device const half *dhu = &xu[ib].d;

        for (short row = 0; row < N_R0_Q4_K; row++) {
            sc16[0] = scg[0] & kmask1;
            sc16[1] = scg[2] & kmask1;
            sc16[2] = ((scg[4] >> 0) & kmask2) | ((scg[0] & kmask3) >> 2);
            sc16[3] = ((scg[4] >> 4) & kmask2) | ((scg[2] & kmask3) >> 2);

            device const uint16_t *qg2 = qg1 + 32;
            float4 acc1g = {0.f, 0.f, 0.f, 0.f};
            float4 acc2g = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1g[0] += yl[2 * i + 0] * (qg1[i] & 0x000F);
                acc1g[1] += yl[2 * i + 1] * (qg1[i] & 0x0F00);
                acc1g[2] += yl[2 * i + 8] * (qg1[i] & 0x00F0);
                acc1g[3] += yl[2 * i + 9] * (qg1[i] & 0xF000);
                acc2g[0] += yh[2 * i + 0] * (qg2[i] & 0x000F);
                acc2g[1] += yh[2 * i + 1] * (qg2[i] & 0x0F00);
                acc2g[2] += yh[2 * i + 8] * (qg2[i] & 0x00F0);
                acc2g[3] += yh[2 * i + 9] * (qg2[i] & 0xF000);
            }

            sumg[row] += dhg[0] * ((acc1g[0] + 1.f / 256.f * acc1g[1]) * sc8[0] +
                                   (acc1g[2] + 1.f / 256.f * acc1g[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2g[0] + 1.f / 256.f * acc2g[1]) * sc8[4] +
                                   (acc2g[2] + 1.f / 256.f * acc2g[3]) * sc8[5] * 1.f / 16.f) -
                         dhg[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            sc16[0] = scu[0] & kmask1;
            sc16[1] = scu[2] & kmask1;
            sc16[2] = ((scu[4] >> 0) & kmask2) | ((scu[0] & kmask3) >> 2);
            sc16[3] = ((scu[4] >> 4) & kmask2) | ((scu[2] & kmask3) >> 2);

            device const uint16_t *qu2 = qu1 + 32;
            float4 acc1u = {0.f, 0.f, 0.f, 0.f};
            float4 acc2u = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1u[0] += yl[2 * i + 0] * (qu1[i] & 0x000F);
                acc1u[1] += yl[2 * i + 1] * (qu1[i] & 0x0F00);
                acc1u[2] += yl[2 * i + 8] * (qu1[i] & 0x00F0);
                acc1u[3] += yl[2 * i + 9] * (qu1[i] & 0xF000);
                acc2u[0] += yh[2 * i + 0] * (qu2[i] & 0x000F);
                acc2u[1] += yh[2 * i + 1] * (qu2[i] & 0x0F00);
                acc2u[2] += yh[2 * i + 8] * (qu2[i] & 0x00F0);
                acc2u[3] += yh[2 * i + 9] * (qu2[i] & 0xF000);
            }

            sumu[row] += dhu[0] * ((acc1u[0] + 1.f / 256.f * acc1u[1]) * sc8[0] +
                                   (acc1u[2] + 1.f / 256.f * acc1u[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2u[0] + 1.f / 256.f * acc2u[1]) * sc8[4] +
                                   (acc2u[2] + 1.f / 256.f * acc2u[3]) * sc8[5] * 1.f / 16.f) -
                         dhu[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            qg1 += args.nb01 / 2;
            scg += args.nb01 / 2;
            dhg += args.nb01 / 2;
            qu1 += args.nb01 / 2;
            scu += args.nb01 / 2;
            dhu += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
        const float gate = simd_sum(sumg[row]);
        const float up = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + row;
            float g = gate;
            float u = up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            gate_f32[out_row] = gate;
            up_f32[out_row] = up;
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_addr_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ulong * gate_addrs,
        device const ulong * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (i02 < 0 || i02 >= args.ne02 || i02 >= 384) {
        return;
    }
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur =
        reinterpret_cast<device const char *>(gate_addrs[(uint)i02]);
    device const char *src0_up_cur =
        reinterpret_cast<device const char *>(up_addrs[(uint)i02]);
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_q4_gather_slots6(
        constant ds4_metal_q4_gather_slots6_args &args,
        device const char *src_group0,
        device const char *src_group1,
        device const char *src_group2,
        device const char *src_group3,
        device const char *src_group4,
        device const char *src_group5,
        device const int32_t *ids,
        device char *dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.n_slots || args.group_size == 0) return;

    const int32_t expert = ids[slot];
    if (expert < 0) return;

    const uint expert_u = (uint)expert;
    const uint group = expert_u / args.group_size;
    if (group >= 6) return;

    const uint local_expert = expert_u - group * args.group_size;
    device const char *src_group = src_group0;
    switch (group) {
    case 1: src_group = src_group1; break;
    case 2: src_group = src_group2; break;
    case 3: src_group = src_group3; break;
    case 4: src_group = src_group4; break;
    case 5: src_group = src_group5; break;
    default: break;
    }

    const uint64_t chunk = (uint64_t)tgpig.x * 256ul + (uint64_t)tiitg;
    const uint64_t n_chunks = args.expert_bytes >> 4;
    if (chunk >= n_chunks) return;

    device const uint4 *src =
        (device const uint4 *)(src_group + (uint64_t)local_expert * args.expert_bytes);
    device uint4 *out =
        (device uint4 *)(dst + (uint64_t)slot * args.expert_bytes);
    out[chunk] = src[chunk];
}

kernel void kernel_mul_mv_slots6_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate0,
        device const char * src0_gate1,
        device const char * src0_gate2,
        device const char * src0_gate3,
        device const char * src0_gate4,
        device const char * src0_gate5,
        device const char * src0_up0,
        device const char * src0_up1,
        device const char * src0_up2,
        device const char * src0_up3,
        device const char * src0_up4,
        device const char * src0_up5,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate0;
    device const char *src0_up_cur = src0_up0;
    switch (idx) {
    case 1: src0_gate_cur = src0_gate1; src0_up_cur = src0_up1; break;
    case 2: src0_gate_cur = src0_gate2; src0_up_cur = src0_up2; break;
    case 3: src0_gate_cur = src0_gate3; src0_up_cur = src0_up3; break;
    case 4: src0_gate_cur = src0_gate4; src0_up_cur = src0_up4; break;
    case 5: src0_gate_cur = src0_gate5; src0_up_cur = src0_up5; break;
    default: break;
    }

    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

static inline device const char *ds4_q4_group24_select(
        uint32_t group_id,
        device const char *src00,
        device const char *src01,
        device const char *src02,
        device const char *src03,
        device const char *src04,
        device const char *src05,
        device const char *src06,
        device const char *src07,
        device const char *src08,
        device const char *src09,
        device const char *src10,
        device const char *src11,
        device const char *src12,
        device const char *src13,
        device const char *src14,
        device const char *src15,
        device const char *src16,
        device const char *src17,
        device const char *src18,
        device const char *src19,
        device const char *src20,
        device const char *src21,
        device const char *src22,
        device const char *src23) {
    switch (group_id) {
    case 1:  return src01;
    case 2:  return src02;
    case 3:  return src03;
    case 4:  return src04;
    case 5:  return src05;
    case 6:  return src06;
    case 7:  return src07;
    case 8:  return src08;
    case 9:  return src09;
    case 10: return src10;
    case 11: return src11;
    case 12: return src12;
    case 13: return src13;
    case 14: return src14;
    case 15: return src15;
    case 16: return src16;
    case 17: return src17;
    case 18: return src18;
    case 19: return src19;
    case 20: return src20;
    case 21: return src21;
    case 22: return src22;
    case 23: return src23;
    default: return src00;
    }
}

kernel void kernel_mul_mv_group6_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate0,
        device const char * src0_gate1,
        device const char * src0_gate2,
        device const char * src0_gate3,
        device const char * src0_gate4,
        device const char * src0_gate5,
        device const char * src0_up0,
        device const char * src0_up1,
        device const char * src0_up2,
        device const char * src0_up3,
        device const char * src0_up4,
        device const char * src0_up5,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 64;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t expert = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (expert < 0) {
        return;
    }
    const uint32_t expert_u = (uint32_t)expert;
    const uint32_t group_id = expert_u / expert_group_size;
    if (group_id >= 6) {
        return;
    }
    const uint32_t expert_local = expert_u - group_id * expert_group_size;

    device const char *src0_gate_cur = src0_gate0;
    device const char *src0_up_cur = src0_up0;
    switch (group_id) {
    case 1: src0_gate_cur = src0_gate1; src0_up_cur = src0_up1; break;
    case 2: src0_gate_cur = src0_gate2; src0_up_cur = src0_up2; break;
    case 3: src0_gate_cur = src0_gate3; src0_up_cur = src0_up3; break;
    case 4: src0_gate_cur = src0_gate4; src0_up_cur = src0_up4; break;
    case 5: src0_gate_cur = src0_gate5; src0_up_cur = src0_up5; break;
    default: break;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    src0_gate_cur += (uint64_t)expert_local * args.nb02;
    src0_up_cur   += (uint64_t)expert_local * args.nb02;
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_group8_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate0,
        device const char * src0_gate1,
        device const char * src0_gate2,
        device const char * src0_gate3,
        device const char * src0_gate4,
        device const char * src0_gate5,
        device const char * src0_gate6,
        device const char * src0_gate7,
        device const char * src0_up0,
        device const char * src0_up1,
        device const char * src0_up2,
        device const char * src0_up3,
        device const char * src0_up4,
        device const char * src0_up5,
        device const char * src0_up6,
        device const char * src0_up7,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 48;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t expert = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (expert < 0) {
        return;
    }
    const uint32_t expert_u = (uint32_t)expert;
    const uint32_t group_id = expert_u / expert_group_size;
    if (group_id >= 8) {
        return;
    }
    const uint32_t expert_local = expert_u - group_id * expert_group_size;

    device const char *src0_gate_cur = src0_gate0;
    device const char *src0_up_cur = src0_up0;
    switch (group_id) {
    case 1: src0_gate_cur = src0_gate1; src0_up_cur = src0_up1; break;
    case 2: src0_gate_cur = src0_gate2; src0_up_cur = src0_up2; break;
    case 3: src0_gate_cur = src0_gate3; src0_up_cur = src0_up3; break;
    case 4: src0_gate_cur = src0_gate4; src0_up_cur = src0_up4; break;
    case 5: src0_gate_cur = src0_gate5; src0_up_cur = src0_up5; break;
    case 6: src0_gate_cur = src0_gate6; src0_up_cur = src0_up6; break;
    case 7: src0_gate_cur = src0_gate7; src0_up_cur = src0_up7; break;
    default: break;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    src0_gate_cur += (uint64_t)expert_local * args.nb02;
    src0_up_cur   += (uint64_t)expert_local * args.nb02;
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_group24_q4_K_id_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src06,
        device const char * src07,
        device const char * src08,
        device const char * src09,
        device const char * src10,
        device const char * src11,
        device const char * src12,
        device const char * src13,
        device const char * src14,
        device const char * src15,
        device const char * src16,
        device const char * src17,
        device const char * src18,
        device const char * src19,
        device const char * src20,
        device const char * src21,
        device const char * src22,
        device const char * src23,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 16;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t expert = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (expert < 0) {
        return;
    }
    const uint32_t expert_u = (uint32_t)expert;
    const uint32_t group_id = expert_u / expert_group_size;
    if (group_id >= 24) {
        return;
    }
    const uint32_t expert_local = expert_u - group_id * expert_group_size;

    device const char *src0_cur = ds4_q4_group24_select(group_id,
                                                        src00, src01, src02, src03,
                                                        src04, src05, src06, src07,
                                                        src08, src09, src10, src11,
                                                        src12, src13, src14, src15,
                                                        src16, src17, src18, src19,
                                                        src20, src21, src22, src23);
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    src0_cur += (uint64_t)expert_local * args.nb02;
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;
    device char *dst_cur = dst + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_cur,
        src1_cur,
        dst_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    (void)tiitg;
}

kernel void kernel_mul_mv_group_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        constant ds4_metal_moe_expert_group_args & group,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t expert_global = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (expert_global < 0) {
        return;
    }
    const uint32_t expert_u = (uint32_t)expert_global;
    if (expert_u < group.expert_base ||
        expert_u >= group.expert_base + group.expert_count) {
        return;
    }
    const uint32_t expert_local = expert_u - group.expert_base;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate + (uint64_t)expert_local * args.nb02;
    device const char *src0_up_cur   = src0_up   + (uint64_t)expert_local * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_id_q2_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        device const block_q2_K * x = (device const block_q2_K *)(src0s + expert*args.nb02 + first_row*args.nb01);
        device const float * y = (device const float *)(token_src1 + expert_slot*args.nb11);
        device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
                yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
                yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
                yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
            }

            device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
            device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                        acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                        acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                        acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                        acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                        acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                        acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                        acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f/16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                         (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                         (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                         (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01/2;
                sc += args.nb01;
                dh += args.nb01/2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_slots6_q2_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        device const char *src0_cur = src00;
        switch (expert_slot) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        default: break;
        }
        device const block_q2_K *x =
            (device const block_q2_K *)(src0_cur + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
            }

            device const uint8_t  *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
            device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                        acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                        acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                        acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                        acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                        acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                        acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                        acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f / 16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                         (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                         (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                         (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01 / 2;
                sc += args.nb01;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_addr_q2_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const uint64_t * addrs,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;
    device const int32_t *token_ids =
        (device const int32_t *)(ids + (uint64_t)token * args.nbi1);

    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0 || expert >= args.ne02 || expert >= 384) {
            continue;
        }
        const uint64_t addr = addrs[(uint)expert];
        if (addr == 0) {
            continue;
        }
        device const char *src0_cur =
            reinterpret_cast<device const char *>(addr);
        device const block_q2_K *x =
            (device const block_q2_K *)(src0_cur + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
            }

            device const uint8_t  *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
            device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                        acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                        acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                        acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                        acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                        acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                        acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                        acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f / 16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                         (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                         (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                         (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }
                qs += args.nb01 / 2;
                sc += args.nb01;
                dh += args.nb01 / 2;
            }
            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_addr_q2_K_sum6_masked_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_stream_expert_split_args & split,
        device const uint64_t * addrs,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;
    device const int32_t *token_ids =
        (device const int32_t *)(ids + (uint64_t)token * args.nbi1);

    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        if ((split.active_mask & (1u << (uint)expert_slot)) == 0) {
            continue;
        }
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0 || expert >= args.ne02 || expert >= 384) {
            continue;
        }
        const uint64_t addr = addrs[(uint)expert];
        if (addr == 0) {
            continue;
        }
        device const char *src0_cur =
            reinterpret_cast<device const char *>(addr);
        device const block_q2_K *x =
            (device const block_q2_K *)(src0_cur + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
            }

            device const uint8_t  *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
            device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                        acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                        acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                        acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                        acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                        acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                        acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                        acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f / 16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                         (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                         (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                         (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }
                qs += args.nb01 / 2;
                sc += args.nb01;
                dh += args.nb01 / 2;
            }
            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            if (split.accumulate) {
                dst_f32[first_row + row] += sum_all;
            } else {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_id_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        device const block_q4_K *x =
            (device const block_q4_K *)(src0s + expert * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_group_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_moe_expert_group_args & group,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0) {
            continue;
        }
        const uint32_t expert_u = (uint32_t)expert;
        if (expert_u < group.expert_base ||
            expert_u >= group.expert_base + group.expert_count) {
            continue;
        }
        const uint32_t expert_local = expert_u - group.expert_base;

        device const block_q4_K *x =
            (device const block_q4_K *)(src0s + (uint64_t)expert_local * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            if (group.accumulate) {
                dst_f32[first_row + row] += sum_all;
            } else {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_table_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const ds4_metal_q4_expert_table & table,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0 || expert >= args.ne02 || expert >= 384) {
            return;
        }
        device const block_q4_K *x =
            (device const block_q4_K *)(table.experts[(uint)expert] + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_addr_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const ulong * addrs,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0 || expert >= args.ne02 || expert >= 384) {
            return;
        }
        device const char *expert_base =
            reinterpret_cast<device const char *>(addrs[(uint)expert]);
        device const block_q4_K *x =
            (device const block_q4_K *)(expert_base + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_slots6_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        device const char *src0_cur = src00;
        switch (expert_slot) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        default: break;
        }
        device const block_q4_K *x =
            (device const block_q4_K *)(src0_cur + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_group6_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 64;
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0) {
            continue;
        }
        const uint32_t expert_u = (uint32_t)expert;
        const uint32_t group_id = expert_u / expert_group_size;
        if (group_id >= 6) {
            continue;
        }
        const uint32_t expert_local = expert_u - group_id * expert_group_size;

        device const char *src0_cur = src00;
        switch (group_id) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        default: break;
        }

        device const block_q4_K *x =
            (device const block_q4_K *)(src0_cur + (uint64_t)expert_local * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_group8_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src06,
        device const char * src07,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 48;
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0) {
            continue;
        }
        const uint32_t expert_u = (uint32_t)expert;
        const uint32_t group_id = expert_u / expert_group_size;
        if (group_id >= 8) {
            continue;
        }
        const uint32_t expert_local = expert_u - group_id * expert_group_size;

        device const char *src0_cur = src00;
        switch (group_id) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        case 6: src0_cur = src06; break;
        case 7: src0_cur = src07; break;
        default: break;
        }

        device const block_q4_K *x =
            (device const block_q4_K *)(src0_cur + (uint64_t)expert_local * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_group24_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src06,
        device const char * src07,
        device const char * src08,
        device const char * src09,
        device const char * src10,
        device const char * src11,
        device const char * src12,
        device const char * src13,
        device const char * src14,
        device const char * src15,
        device const char * src16,
        device const char * src17,
        device const char * src18,
        device const char * src19,
        device const char * src20,
        device const char * src21,
        device const char * src22,
        device const char * src23,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 16;
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0) {
            continue;
        }
        const uint32_t expert_u = (uint32_t)expert;
        const uint32_t group_id = expert_u / expert_group_size;
        if (group_id >= 24) {
            continue;
        }
        const uint32_t expert_local = expert_u - group_id * expert_group_size;

        device const char *src0_cur = ds4_q4_group24_select(group_id,
                                                            src00, src01, src02, src03,
                                                            src04, src05, src06, src07,
                                                            src08, src09, src10, src11,
                                                            src12, src13, src14, src15,
                                                            src16, src17, src18, src19,
                                                            src20, src21, src22, src23);
        device const block_q4_K *x =
            (device const block_q4_K *)(src0_cur + (uint64_t)expert_local * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

#define QK_NL 16

// Builds the compact per-expert work map used by batched MoE matmul. DS4 routes
// each token to a small fixed top-k list, so this turns token-major ids into
// expert-major slices that the tiled matmul can consume.
template<short ne20>
kernel void kernel_mul_mm_id_map0(
        constant ds4_metal_args_mul_mm_id_map0 & args,
        device  const char * src2,
        device        char * htpe,
        device        char * hids,
        threadgroup   char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const short ide = tpitg;

    uint32_t n_all = 0;

    device int32_t * ids_i32 = (device int32_t *) hids + ide*args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) {
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            threadgroup uint16_t * sids = (threadgroup uint16_t *) shmem + tpitg*ne20;

            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sids[i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids = (threadgroup const uint16_t *) shmem + t*ne20;

            short sel = 0;
            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sel += (sids[i20] == ide)*(i20 + 1);
            }

            ids_i32[n_all] = (i21 + t)*ne20 + sel - 1;

            n_all += sel > 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * tpe_u32 = (device uint32_t *) (htpe);
    tpe_u32[ide] = n_all;
}

typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;

// Host-visible map builders for the routed-expert counts used by DS4 graph
// shapes. Some arities are generic leftovers retained for nearby batch sizes.
template [[host_name("kernel_mul_mm_id_map0_ne20_1" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<1>;
template [[host_name("kernel_mul_mm_id_map0_ne20_2" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<2>;
template [[host_name("kernel_mul_mm_id_map0_ne20_4" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<4>;
template [[host_name("kernel_mul_mm_id_map0_ne20_5" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<5>;
template [[host_name("kernel_mul_mm_id_map0_ne20_6" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<6>;
template [[host_name("kernel_mul_mm_id_map0_ne20_8" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<8>;
template [[host_name("kernel_mul_mm_id_map0_ne20_10")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<10>;
template [[host_name("kernel_mul_mm_id_map0_ne20_16")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<16>;
template [[host_name("kernel_mul_mm_id_map0_ne20_22")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<22>;

// Batched routed-expert matmul. It reads the expert-major map produced above,
// loads selected expert weights, and writes results back to token-major slots
// so the DS4 FFN can apply SwiGLU, weighting, and the down projection.
template<short NR1, typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm_id(
        constant ds4_metal_args_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    static_assert(NR1 == 32, "kernel_mul_mm_id accumulator layout supports only 32 routed rows");

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int idj = ids_i32[im*args.ne21 + r1 + j];

        const short ide = idj % args.ne20;
        const short idt = idj / args.ne20;

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}

kernel void kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16(
        constant ds4_metal_args_mul_mm_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst_mid,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/QK_NL;

    device const block_iq2_xxs * xg =
        (device const block_iq2_xxs *)(src0_gate + args.nb01*(r0 + lr0) + offset0) + offset1;
    device const block_iq2_xxs * xu =
        (device const block_iq2_xxs *)(src0_up + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];

    simdgroup_float8x8 mc_gate[8];
    simdgroup_float8x8 mc_up[8];

    for (short i = 0; i < 8; i++) {
        mc_gate[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        mc_up[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        const short sx_b = (tiitg%NL1);
        const short sy_b = (tiitg/NL1)/8;
        const short ly_b = (tiitg/NL1)%8;
        const short ib_b = 4*sx_b + sy_b;
        *(threadgroup half2x4 *)(sb + 64*ib_b + 8*ly_b) =
            (half2x4)(*((device float2x4 *) y));

        half4x4 temp_gate;
        dequantize_iq2_xxs(xg, il, temp_gate);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_gate[i/4][i%4];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_gate = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_gate + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_gate[i], mb[i/4], ma[i%4], mc_gate[i]);
            }

            lsma_gate += 8*64;
            lsmb += 4*64;
        }

        half4x4 temp_up;
        dequantize_iq2_xxs(xu, il, temp_up);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_up[i/4][i%4];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_up = (sa + 4*64*(sgitg%2));
        lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_up + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_up[i], mb[i/4], ma[i%4], mc_up[i]);
            }

            lsma_up += 8*64;
            lsmb += 4*64;
        }

        il = (il + 2 < QK_NL) ? il + 2 : il % 2;
        xg = (il < 2) ? xg + (2 + QK_NL - 1)/QK_NL : xg;
        xu = (il < 2) ? xu + (2 + QK_NL - 1)/QK_NL : xu;
        y += NK;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_gate = (threadgroup float *) shmem;
    threadgroup float * temp_up = temp_gate + NR0*NR1;
    threadgroup float * temp_gate_str =
        temp_gate + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
    threadgroup float * temp_up_str =
        temp_up + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc_gate[i], temp_gate_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        simdgroup_store(mc_up[i],   temp_up_str   + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float c = act.clamp_value;
    for (short j = sgitg; j < nr1; j += 4) {
        const int idj = ids_i32[im*args.ne21 + r1 + j];

        const short ide = idj % args.ne20;
        const short idt = idj / args.ne20;

        device half *D = (device half *)(dst_mid +
            ((uint64_t)idt*args.ne1 + (uint64_t)ide)*act.mid_row_stride) + r0;
        device const float *w = (device const float *)(weights + (uint64_t)idj*act.weight_stride);
        const float route_weight = w[0];

        threadgroup float *Cg = temp_gate + j*NR0;
        threadgroup float *Cu = temp_up   + j*NR0;

        int i = tiisg;
        for (; i < nr0; i += 32) {
            float g = Cg[i];
            float u = Cu[i];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            D[i] = (half)(silu * u * route_weight);
        }
    }
}

typedef decltype(kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>) mul_mm_id;
typedef decltype(kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, half, half4x4, half, half2x4>) mul_mm_id_f16_rhs;

// Host-visible batched MoE matmul variants for the DS4 quant formats.
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]         kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0,    2,     dequantize_q8_0,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]         kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]         kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]]      kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f16")]]         kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0,    2,     dequantize_q8_0,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16")]]         kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16")]]         kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]]      kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, half, half4x4, half, half2x4>;

#ifdef DS4_METAL_HAS_TENSOR
// Attention-output low-rank projection retained for Metal4 prefill.  It uses
// the same direct-RHS idea as dense matmul: dequantize the Q8_0 low projection
// weights to a half tile, then let TensorOps read the dense head activations
// directly.  Only the 64-token direct-RHS instantiation is exported because the
// staged-RHS and 32-token variants were benchmark-only experiments.
template<short NR1>
kernel void kernel_attn_out_low_q8_0_mpp_direct_rhs(
        constant ds4_metal_args_mul_mm_id & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NR0 = 64;
    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne21;
    const int G = args.ne1;
    const int group = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;
    const bool full_tile = r0 + NR0 <= M && r1 + NR1 <= N && (K % NK) == 0;

    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));

    device float *ptrB = (device float *)(srcB + args.nb11*group);
    const int strideB = args.nb12/sizeof(float);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (full_tile || r0 + row < M) {
                const int block_idx = k_pos/32;
                const short il = (k_pos/16)%2;
                device const block_q8_0 *row_ptr =
                    (device const block_q8_0 *)(srcA + args.nb01*(r0 + row) + group*args.nb02);

                half4x4 temp_a;
                dequantize_q8_0(row_ptr + block_idx, il, temp_a);
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (full_tile || k_pos + i < K) ? temp_a[i/4][i%4] : (half)0;
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (half)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_group = (device float *)dst + group*M;
    if (full_tile) {
        device float *dst_tile = dst_group + r0 + (uint64_t)r1*G*M;
        auto tD = tensor(dst_tile, dextents<int32_t, 2>(NR0, NR1), array<int, 2>({1, G*M}));
        cT.store(tD);
    } else {
        auto tD = tensor(dst_group, dextents<int32_t, 2>(M, N), array<int, 2>({1, G*M}));
        auto mD = tD.slice(r0, r1);
        cT.store(mD);
    }
}

typedef decltype(kernel_attn_out_low_q8_0_mpp_direct_rhs<64>) attn_out_low_q8_0_mpp_direct_rhs_n64_t;

template [[host_name("kernel_attn_out_low_q8_0_mpp_direct_rhs_n64")]] kernel attn_out_low_q8_0_mpp_direct_rhs_n64_t kernel_attn_out_low_q8_0_mpp_direct_rhs<64>;

#endif

#undef QK_NL
#undef kmask_iq2xs
#undef ksigns_iq2xs
#undef iq2xxs_grid
#undef QK_K
#undef N_R0_Q2_K
#undef N_R0_Q4_K
#undef N_R0_IQ2_XXS
"""###,
        "dsv4_hc": ###"""
struct ds4_metal_args_dsv4_hc_split_sinkhorn {
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb01;
    uint64_t nb1;
    float    eps;
};

struct ds4_metal_args_dsv4_hc_weighted_sum {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_hc_split_weighted_sum {
    int64_t  n_embd;
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    float    eps;
};

struct ds4_metal_args_dsv4_hc_split_weighted_sum_norm {
    int64_t  n_embd;
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb_norm1;
    float    eps;
    float    norm_eps;
};

struct ds4_metal_args_dsv4_hc_expand {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_block0;
    uint64_t nb_block1;
    uint64_t nb_add0;
    uint64_t nb_add1;
    uint64_t nb_res0;
    uint64_t nb_res1;
    uint64_t nb_res2;
    uint64_t nb_post0;
    uint64_t nb_post1;
    uint64_t nb_comb0;
    uint64_t nb_comb1;
    uint64_t nb_comb2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    int32_t  has_add;
};

// Numerically stable sigmoid for the standalone split/sinkhorn path. The naive
// form 1/(1+exp(-z)) overflows for large negative z (exp(-z) blows up);
// replacing it with the 0.5*(tanh(z/2)+1) identity keeps the value bounded in
// [0, 1] across the entire float range. Gated by DS4_METAL_HC_STABLE so we can
// A/B vs the historical form on M5 Max where the faster ALU is more likely to
// push HC mixer inputs into the unstable regime.
//
// Do not automatically use these helpers in the fused HC decode kernels below:
// routing the fused vector sites through the tanh form produced non-finite
// logits on M5 Max, while the historical inline exp form remains finite and is
// the decode throughput baseline.
#ifdef DS4_METAL_HC_STABLE
static inline float  ds4_hc_sigmoid(float  z)  { return 0.5f * tanh(0.5f * z) + 0.5f; }
static inline float4 ds4_hc_sigmoid(float4 z)  { return 0.5f * tanh(0.5f * z) + 0.5f; }
// 2 * sigmoid(z) == 1 + tanh(z/2).
static inline float  ds4_hc_twice_sigmoid(float  z) { return 1.0f + tanh(0.5f * z); }
static inline float4 ds4_hc_twice_sigmoid(float4 z) { return 1.0f + tanh(0.5f * z); }
#else
static inline float  ds4_hc_sigmoid(float  z)  { return 1.0f / (1.0f + exp(-z)); }
static inline float4 ds4_hc_sigmoid(float4 z)  { return 1.0f / (1.0f + exp(-z)); }
static inline float  ds4_hc_twice_sigmoid(float  z) { return 2.0f / (1.0f + exp(-z)); }
static inline float4 ds4_hc_twice_sigmoid(float4 z) { return 2.0f / (1.0f + exp(-z)); }
#endif

// Splits an HC mixer row into pre weights, post gates, and the HC-to-HC
// combination matrix. The 4-channel path is specialized because DS4 Flash uses
// HC=4 in normal inference, while the scalar fallback keeps diagnostics usable.
kernel void kernel_dsv4_hc_split_sinkhorn(
        constant ds4_metal_args_dsv4_hc_split_sinkhorn & args,
        device  const float * mixes,
        device  const float * scale,
        device  const float * base,
        device        float * dst,
        uint tid [[thread_position_in_grid]]) {
    if ((int64_t) tid >= args.n_rows) {
        return;
    }

    constexpr int HC_MAX = 16;
    const int HC = args.n_hc;
    if (HC <= 0 || HC > HC_MAX) {
        return;
    }

    device const float * mix = mixes + ((int64_t) tid)*args.mix_hc;
    device       float * out = dst    + ((int64_t) tid)*args.mix_hc;

    const float epsv       = args.eps;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    if (HC == 4) {
        const float4 pre_z =
            *((device const float4 *) mix) * pre_scale +
            *((device const float4 *) base);
        *((device float4 *) out) = ds4_hc_sigmoid(pre_z) + epsv;

        const float4 post_z =
            *((device const float4 *) (mix  + 4)) * post_scale +
            *((device const float4 *) (base + 4));
        *((device float4 *) (out + 4)) = ds4_hc_twice_sigmoid(post_z);

        float4 r0 =
            *((device const float4 *) (mix  +  8)) * comb_scale +
            *((device const float4 *) (base +  8));
        float4 r1 =
            *((device const float4 *) (mix  + 12)) * comb_scale +
            *((device const float4 *) (base + 12));
        float4 r2 =
            *((device const float4 *) (mix  + 16)) * comb_scale +
            *((device const float4 *) (base + 16));
        float4 r3 =
            *((device const float4 *) (mix  + 20)) * comb_scale +
            *((device const float4 *) (base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *) (out +  8)) = r0;
        *((device float4 *) (out + 12)) = r1;
        *((device float4 *) (out + 16)) = r2;
        *((device float4 *) (out + 20)) = r3;
        return;
    }

    for (int i = 0; i < HC; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = ds4_hc_sigmoid(z) + epsv;
    }

    for (int i = 0; i < HC; ++i) {
        const int off = HC + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = ds4_hc_twice_sigmoid(z);
    }

    float c[HC_MAX*HC_MAX];

    for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            const int off = 2*HC + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = max(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            const float v = exp(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            c[idx] = c[idx] * inv_sum + epsv;
        }
    }

    for (int src_hc = 0; src_hc < HC; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            sum += c[src_hc + dst_hc*HC];
        }

        const float inv_denom = 1.0f / (sum + epsv);
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            c[src_hc + dst_hc*HC] *= inv_denom;
        }
    }

    for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                sum += c[src_hc + dst_hc*HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                c[src_hc + dst_hc*HC] *= inv_denom;
            }
        }

        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                sum += c[src_hc + dst_hc*HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                c[src_hc + dst_hc*HC] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < HC*HC; ++i) {
        out[2*HC + i] = c[i];
    }
}

// Decode-side fusion of HC split and pre-weighted HC reduction. One threadgroup
// handles one token row: lane 0 computes the HC=4 mixer split once, stores the
// post/comb data for the following HC expand, and all lanes reuse the pre
// weights from threadgroup memory to produce the embedding row.
kernel void kernel_dsv4_hc_split_weighted_sum(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum & args,
        device  const char  * mixes,
        device  const float * scale,
        device  const float * base,
        device  const char  * x,
        device        char  * split,
        device        char  * dst,
        threadgroup   float * pre_shmem [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if ((int64_t) row >= args.n_rows || args.n_hc != 4) {
        return;
    }

    device const float * mix = (device const float *) (mixes + (uint64_t)row*args.nb_mix1);
    device       float * out = (device       float *) (split + (uint64_t)row*args.nb_split1);

    if (tid == 0) {
        const float epsv       = args.eps;
        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *) mix) * pre_scale +
            *((device const float4 *) base);
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *) out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((device const float4 *) (mix  + 4)) * post_scale +
            *((device const float4 *) (base + 4));
        *((device float4 *) (out + 4)) = 2.0f / (1.0f + exp(-post_z));

        float4 r0 =
            *((device const float4 *) (mix  +  8)) * comb_scale +
            *((device const float4 *) (base +  8));
        float4 r1 =
            *((device const float4 *) (mix  + 12)) * comb_scale +
            *((device const float4 *) (base + 12));
        float4 r2 =
            *((device const float4 *) (mix  + 16)) * comb_scale +
            *((device const float4 *) (base + 16));
        float4 r3 =
            *((device const float4 *) (mix  + 20)) * comb_scale +
            *((device const float4 *) (base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *) (out +  8)) = r0;
        *((device float4 *) (out + 12)) = r1;
        *((device float4 *) (out + 16)) = r2;
        *((device float4 *) (out + 20)) = r3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int64_t d = tid; d < args.n_embd; d += ntg) {
        float acc = 0.0f;
        acc += *((device const float *) (x + d*args.nb_x0 + 0*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[0];
        acc += *((device const float *) (x + d*args.nb_x0 + 1*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[1];
        acc += *((device const float *) (x + d*args.nb_x0 + 2*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[2];
        acc += *((device const float *) (x + d*args.nb_x0 + 3*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[3];
        *((device float *) (dst + d*args.nb0 + (uint64_t)row*args.nb1)) = acc;
    }
}

// Decode HC-pre plus the following RMSNorm.  DS4 uses HC=4 here.  The normal
// release path computes HC coefficients, collapses four residual streams into
// the model row, then immediately launches a weighted RMSNorm over the row.
// This kernel keeps the HC split math identical to
// kernel_dsv4_hc_split_weighted_sum, stores the HC-pre row for diagnostics, and
// reuses the just-collapsed values from threadgroup memory for the RMSNorm
// reduction.
kernel void kernel_dsv4_hc_split_weighted_sum_norm4(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & args,
        device  const char  * mixes,
        device  const float * scale,
        device  const float * base,
        device  const char  * x,
        device        char  * split,
        device        char  * dst,
        device  const char  * norm_weight,
        device        char  * norm_dst,
        threadgroup   float * shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    if ((int64_t)row >= args.n_rows || args.n_hc != 4 || (args.n_embd & 3) != 0) {
        return;
    }

    const uint n_embd = uint(args.n_embd);
    const uint n4 = n_embd >> 2;
    threadgroup float4 *row_shmem = (threadgroup float4 *)shared;
    threadgroup float *pre_shmem = shared + n_embd;
    threadgroup float *sum_shmem = pre_shmem + 4;

    device const float *mix = (device const float *)(mixes + (uint64_t)row * args.nb_mix1);
    device float *out = (device float *)(split + (uint64_t)row * args.nb_split1);

    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    if (tid == 0) {
        const float epsv = args.eps;
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *)mix) * pre_scale +
            *((device const float4 *)base);
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *)out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((device const float4 *)(mix + 4)) * post_scale +
            *((device const float4 *)(base + 4));
        *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));

        float4 r0 =
            *((device const float4 *)(mix + 8)) * comb_scale +
            *((device const float4 *)(base + 8));
        float4 r1 =
            *((device const float4 *)(mix + 12)) * comb_scale +
            *((device const float4 *)(base + 12));
        float4 r2 =
            *((device const float4 *)(mix + 16)) * comb_scale +
            *((device const float4 *)(base + 16));
        float4 r3 =
            *((device const float4 *)(mix + 20)) * comb_scale +
            *((device const float4 *)(base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *)(out + 8)) = r0;
        *((device float4 *)(out + 12)) = r1;
        *((device float4 *)(out + 16)) = r2;
        *((device float4 *)(out + 20)) = r3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        device const float4 *x0 = (device const float4 *)(x + 0 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x1 = (device const float4 *)(x + 1 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x2 = (device const float4 *)(x + 2 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x3 = (device const float4 *)(x + 3 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        const float4 v = x0[i] * pre_shmem[0] +
                         x1[i] * pre_shmem[1] +
                         x2[i] * pre_shmem[2] +
                         x3[i] * pre_shmem[3];
        row_shmem[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);
    const float norm_scale = rsqrt(sumf / float(n_embd) + args.norm_eps);

    device float4 *dst4 = (device float4 *)(dst + (uint64_t)row * args.nb1);
    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)(norm_dst + (uint64_t)row * args.nb_norm1);
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = row_shmem[i];
        dst4[i] = v;
        norm4[i] = (v * norm_scale) * w4[i];
    }
}

// Expands an embedding-sized block back into HC channels after attention/FFN.
// The post gate scales the current block, while the Sinkhorn combination matrix
// mixes residual HC channels from the previous state.
kernel void kernel_dsv4_hc_expand(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n_elem = args.n_embd * args.n_hc * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d      = ((int64_t) gid) % args.n_embd;
    const int64_t tmp    = ((int64_t) gid) / args.n_embd;
    const int64_t dst_hc = tmp % args.n_hc;
    const int64_t t      = tmp / args.n_hc;

    float block_v = *((device const float *) (block_out + d*args.nb_block0 + t*args.nb_block1));
    if (args.has_add) {
        block_v += *((device const float *) (block_add + d*args.nb_add0 + t*args.nb_add1));
    }
    const float post_v  = *((device const float *) (post      + dst_hc*args.nb_post0 + t*args.nb_post1));

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < args.n_hc; ++src_hc) {
        const float comb_v = *((device const float *) (comb     + dst_hc*args.nb_comb0 + src_hc*args.nb_comb1 + t*args.nb_comb2));
        const float res_v  = *((device const float *) (residual + d*args.nb_res0 + src_hc*args.nb_res1 + t*args.nb_res2));
        acc += comb_v * res_v;
    }

    *((device float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) = acc;
}

// HC=4 specialization of the post/expand step. One thread computes all four
// destination HC streams for one token/dimension, reusing the same block output
// and residual HC values while preserving the per-stream accumulation order.
kernel void kernel_dsv4_hc_expand4(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    if (args.n_hc != 4) {
        return;
    }

    const int64_t n_elem = args.n_embd * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float block_v = *((device const float *) (block_out + d*args.nb_block0 + t*args.nb_block1));
    if (args.has_add) {
        block_v += *((device const float *) (block_add + d*args.nb_add0 + t*args.nb_add1));
    }

    const float r0 = *((device const float *) (residual + d*args.nb_res0 + 0*args.nb_res1 + t*args.nb_res2));
    const float r1 = *((device const float *) (residual + d*args.nb_res0 + 1*args.nb_res1 + t*args.nb_res2));
    const float r2 = *((device const float *) (residual + d*args.nb_res0 + 2*args.nb_res1 + t*args.nb_res2));
    const float r3 = *((device const float *) (residual + d*args.nb_res0 + 3*args.nb_res1 + t*args.nb_res2));

    for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
        float acc = block_v * *((device const float *) (post + dst_hc*args.nb_post0 + t*args.nb_post1));

        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 0*args.nb_comb1 + t*args.nb_comb2)) * r0;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 1*args.nb_comb1 + t*args.nb_comb2)) * r1;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 2*args.nb_comb1 + t*args.nb_comb2)) * r2;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 3*args.nb_comb1 + t*args.nb_comb2)) * r3;

        *((device float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) = acc;
    }
}

// Decode-time FFN tail fusion:
//
//     shared_out = shared_mid @ Wshared_down
//     after_ffn_hc = HCPost(routed_out + shared_out, residual_hc, split)
//
// The Q8_0 dot reduction is intentionally copied from the normal matvec shape
// so the shared expert result is bit-identical.  The only specialization is
// that DS4 decode has one token and HC=4, so the thread that finishes each
// shared-down output row can immediately expand it into the four HC streams.
kernel void kernel_dsv4_shared_down_hc_expand4_q8_0(
        constant ds4_metal_args_mul_mv        & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device  const char * weight,
        device  const char * shared_mid,
        device        char * shared_out,
        device  const char * routed_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device        char * dst,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = tgpig.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const float *y = (device const float *)(shared_mid);
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    device const block_q8_0 *ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float shared_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(shared_out + (uint64_t)d * sizeof(float))) = shared_v;

            float block_v = *((device const float *)(routed_out + (uint64_t)d * hc.nb_block0));
            block_v += shared_v;

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc * hc.nb_post0));

                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

                *((device float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
            }
        }
    }
}

// Decode-time attention output tail fusion:
//
//     attn_out = attn_low @ Wob
//     after_attn_hc = HCPost(attn_out, residual_hc, split)
//
// This is the no-add sibling of the shared-down/FFN fusion above.  It preserves
// the exact Q8_0 matvec reduction, stores `attn_out` for diagnostics, and then
// writes the four HC streams for the same embedding dimension.
kernel void kernel_dsv4_q8_hc_expand4_q8_0(
        constant ds4_metal_args_mul_mv        & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device  const char * weight,
        device  const char * input,
        device        char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device        char * dst,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = tgpig.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const float *y = (device const float *)(input);
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    device const block_q8_0 *ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float block_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(block_out + (uint64_t)d * sizeof(float))) = block_v;

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc * hc.nb_post0));

                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

                *((device float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
            }
        }
    }
}

// Reduces HC channels to a normal embedding row with the learned pre weights.
// This is the input adapter before the attention block and before the FFN block.
kernel void kernel_dsv4_hc_weighted_sum(
        constant ds4_metal_args_dsv4_hc_weighted_sum & args,
        device  const char * x,
        device  const char * weights,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n_elem = args.n_embd * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < args.n_hc; ++h) {
        const float xv = *((device const float *) (x       + d*args.nb_x0 + h*args.nb_x1 + t*args.nb_x2));
        const float wv = *((device const float *) (weights + h*args.nb_w0 + t*args.nb_w1));
        acc += xv * wv;
    }

    *((device float *) (dst + d*args.nb0 + t*args.nb1)) = acc;
}
"""###,
        "unary": ###"""
#define FC_UNARY 1200

#define OP_UNARY_NUM_SCALE      10
#define OP_UNARY_NUM_FILL       11
#define OP_UNARY_NUM_CLAMP      12
#define OP_UNARY_NUM_SQR        13
#define OP_UNARY_NUM_SQRT       14
#define OP_UNARY_NUM_SIN        15
#define OP_UNARY_NUM_COS        16
#define OP_UNARY_NUM_LOG        17
#define OP_UNARY_NUM_LEAKY_RELU 18

#define OP_UNARY_NUM_TANH        100
#define OP_UNARY_NUM_RELU        101
#define OP_UNARY_NUM_SIGMOID     102
#define OP_UNARY_NUM_GELU        103
#define OP_UNARY_NUM_GELU_ERF    104
#define OP_UNARY_NUM_GELU_QUICK  105
#define OP_UNARY_NUM_SILU        106
#define OP_UNARY_NUM_ELU         107
#define OP_UNARY_NUM_NEG         108
#define OP_UNARY_NUM_ABS         109
#define OP_UNARY_NUM_SGN         110
#define OP_UNARY_NUM_STEP        111
#define OP_UNARY_NUM_HARDSWISH   112
#define OP_UNARY_NUM_HARDSIGMOID 113
#define OP_UNARY_NUM_EXP         114
#define OP_UNARY_NUM_SOFTPLUS    115
#define OP_UNARY_NUM_EXPM1       116
#define OP_UNARY_NUM_FLOOR       117
#define OP_UNARY_NUM_CEIL        118
#define OP_UNARY_NUM_ROUND       119
#define OP_UNARY_NUM_TRUNC       120
#define OP_UNARY_NUM_XIELU       121

struct ds4_metal_args_unary {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    slope;
    float    scale;
    float    bias;
    float    val;
    float    min;
    float    max;
};

constant float GELU_COEF_A     = 0.044715f;
constant float GELU_QUICK_COEF = -1.702f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant float SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant float p_erf  = 0.3275911f;
constant float a1_erf = 0.254829592f;
constant float a2_erf = -0.284496736f;
constant float a3_erf = 1.421413741f;
constant float a4_erf = -1.453152027f;
constant float a5_erf = 1.061405429f;

template<typename T>
inline T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x);
    return sign_x * y;
}

template<typename T> T elu_approx(T x);

template<> inline float elu_approx<float>(float x) {
    return (x > 0.f) ? x : (exp(x) - 1);
}

template<> inline float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (exp(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (exp(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (exp(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (exp(x[3]) - 1.0f);

    return res;
}

constant short FC_unary_op [[function_constant(FC_UNARY + 0)]];
constant bool  FC_unary_cnt[[function_constant(FC_UNARY + 1)]];

// Generic unary elementwise op selected by function constant. DS4 only uses a
// small subset in inference, mainly sigmoid, SiLU, softplus, sqrt, clamp,
// scale, and fill.
template <typename T0, typename T, typename TC>
kernel void kernel_unary_impl(
        constant ds4_metal_args_unary & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
#define FC_OP  FC_unary_op
#define FC_CNT FC_unary_cnt

    device const T0 * src0_ptr;
    device       T  * dst_ptr;

    int i0;

    if (FC_CNT) {
        i0 = tgpig.x;

        src0_ptr = (device const T0 *) (src0);
        dst_ptr  = (device       T  *) (dst);
    } else {
        const int i03 = tgpig.z;
        const int i02 = tgpig.y;
        const int k0  = tgpig.x/args.ne01;
        const int i01 = tgpig.x - k0*args.ne01;

        i0 = k0*ntg.x + tpitg.x;

        src0_ptr = (device const T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
        dst_ptr  = (device       T  *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1 );
    }

    {
        if (!FC_CNT) {
            if (i0 >= args.ne0) {
                return;
            }
        }

        const TC x = (TC) src0_ptr[i0];

        if (FC_OP == OP_UNARY_NUM_SCALE) {
            dst_ptr[i0] = (T) (args.scale * x + args.bias);
        }

        if (FC_OP == OP_UNARY_NUM_FILL) {
            dst_ptr[i0] = (T) args.val;
        }

        if (FC_OP == OP_UNARY_NUM_CLAMP) {
            dst_ptr[i0] = (T) clamp(x, args.min, args.max);
        }

        if (FC_OP == OP_UNARY_NUM_SQR) {
            dst_ptr[i0] = (T) (x * x);
        }

        if (FC_OP == OP_UNARY_NUM_SQRT) {
            dst_ptr[i0] = (T) sqrt(x);
        }

        if (FC_OP == OP_UNARY_NUM_SIN) {
            dst_ptr[i0] = (T) sin(x);
        }

        if (FC_OP == OP_UNARY_NUM_COS) {
            dst_ptr[i0] = (T) cos(x);
        }

        if (FC_OP == OP_UNARY_NUM_LOG) {
            dst_ptr[i0] = (T) log(x);
        }

        if (FC_OP == OP_UNARY_NUM_LEAKY_RELU) {
            dst_ptr[i0] = (T) (TC(x > 0)*x + TC(x <= 0)*(x * args.slope));
        }

        if (FC_OP == OP_UNARY_NUM_TANH) {
            dst_ptr[i0] = (T) precise::tanh(x);
        }

        if (FC_OP == OP_UNARY_NUM_RELU) {
            dst_ptr[i0] = (T) fmax(0, x);
        }

        if (FC_OP == OP_UNARY_NUM_SIGMOID) {
            dst_ptr[i0] = (T) (1 / (1 + exp(-x)));
        }

        if (FC_OP == OP_UNARY_NUM_GELU) {
            dst_ptr[i0] = (T) (0.5*x*(1 + precise::tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x))));
        }

        if (FC_OP == OP_UNARY_NUM_GELU_ERF) {
            dst_ptr[i0] = (T) (0.5*x*(1 + erf_approx(SQRT_2_INV*x)));
        }

        if (FC_OP == OP_UNARY_NUM_GELU_QUICK) {
            dst_ptr[i0] = (T) (x * (1/(1 + exp(GELU_QUICK_COEF*x))));
        }

        if (FC_OP == OP_UNARY_NUM_SILU) {
            dst_ptr[i0] = (T) (x / (1 + exp(-x)));
        }

        if (FC_OP == OP_UNARY_NUM_ELU) {
            dst_ptr[i0] = (T) elu_approx(x);
        }

        if (FC_OP == OP_UNARY_NUM_NEG) {
            dst_ptr[i0] = (T) -x;
        }

        if (FC_OP == OP_UNARY_NUM_ABS) {
            dst_ptr[i0] = (T) fabs(x);
        }

        if (FC_OP == OP_UNARY_NUM_SGN) {
            dst_ptr[i0] = T(x > 0) - T(x < 0);
        }

        if (FC_OP == OP_UNARY_NUM_STEP) {
            dst_ptr[i0] = T(x > 0);
        }

        if (FC_OP == OP_UNARY_NUM_HARDSWISH) {
            dst_ptr[i0] = (T) (x * fmax(0, fmin(1, x/6 + 0.5)));
        }

        if (FC_OP == OP_UNARY_NUM_HARDSIGMOID) {
            dst_ptr[i0] = (T) fmax(0, fmin(1, x/6 + 0.5));
        }

        if (FC_OP == OP_UNARY_NUM_EXP) {
            dst_ptr[i0] = (T) exp(x);
        }

        if (FC_OP == OP_UNARY_NUM_SOFTPLUS) {
            dst_ptr[i0] = (T) select(log(1 + exp(x)), x, x > 20);
        }

        if (FC_OP == OP_UNARY_NUM_EXPM1) {
            // Metal target profiles used here do not all expose expm1(); this
            // generic unary branch is not used by the DS4 inference graph.
            dst_ptr[i0] = (T) (exp(x) - 1);
        }

        if (FC_OP == OP_UNARY_NUM_FLOOR) {
            dst_ptr[i0] = (T) floor(x);
        }

        if (FC_OP == OP_UNARY_NUM_CEIL) {
            dst_ptr[i0] = (T) ceil(x);
        }

        if (FC_OP == OP_UNARY_NUM_ROUND) {
            dst_ptr[i0] = (T) round(x);
        }

        if (FC_OP == OP_UNARY_NUM_TRUNC) {
            dst_ptr[i0] = (T) trunc(x);
        }

        if (FC_OP == OP_UNARY_NUM_XIELU) {
            const TC xi      = x;
            const TC gate    = TC(xi > TC(0.0f));
            const TC clamped = fmin(xi, TC(args.val));
            const TC y_pos   = TC(args.scale) * xi * xi + TC(args.bias) * xi;
            const TC y_neg   = (exp(clamped) - TC(1.0f) - xi) * TC(args.slope) + TC(args.bias) * xi;
            dst_ptr[i0] = (T) (gate * y_pos + (TC(1.0f) - gate) * y_neg);
        }
    }

#undef FC_OP
#undef FC_CNT
}

typedef decltype(kernel_unary_impl<float, float, float>) kernel_unary_t;

// Decode router probability transform. The generic path applies softplus and
// sqrt as two elementwise kernels; DS4 decode always transforms one 256-wide
// expert-logit row, so this vectorized kernel does both in one pass.
kernel void kernel_dsv4_softplus_sqrt_f32_4(
        constant ds4_metal_args_unary & args,
        device const char *src,
        device       char *dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const int k0 = tgpig.x/args.ne01;
    const int i01 = tgpig.x - k0*args.ne01;
    const int i0 = k0*ntg.x + tpitg.x;
    if (i0 >= args.ne0) return;

    device const float4 *s = (device const float4 *)(src + i01*args.nb01);
    device       float4 *d = (device       float4 *)(dst + i01*args.nb1);
    const float4 x = s[i0];
    const float4 sp = select(log(1.0f + exp(x)), x, x > 20.0f);
    d[i0] = sqrt(sp);
}

// Host-visible unary variants. Function constants select the actual DS4 op.
template [[host_name("kernel_unary_f32_f32")]]   kernel kernel_unary_t kernel_unary_impl<float,  float,  float>;
template [[host_name("kernel_unary_f32_f32_4")]] kernel kernel_unary_t kernel_unary_impl<float4, float4, float4>;
template [[host_name("kernel_unary_f16_f16")]]   kernel kernel_unary_t kernel_unary_impl<half,   half,   float>;
"""###,
        "dsv4_kv": ###"""
constant float dsv4_e4m3fn_exp_scale[16] = {
    0.0f, 0.015625f, 0.03125f, 0.0625f,
    0.125f, 0.25f, 0.5f, 1.0f,
    2.0f, 4.0f, 8.0f, 16.0f,
    32.0f, 64.0f, 128.0f, 256.0f,
};

constant float dsv4_e2m1fn_values[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
};

struct ds4_metal_args_dsv4_fp8_kv_quantize {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    ulong nb0;
    ulong nb1;
    ulong nb2;
    ulong nb3;
    int n_rot;
};

struct ds4_metal_args_dsv4_kv_fp8_store {
    int32_t head_dim;
    int32_t n_rot;
    int32_t raw_row;
};

struct ds4_metal_args_dsv4_indexer_qat {
    uint32_t n_rows;
    uint32_t head_dim;
    uint64_t row_stride;
};

struct ds4_metal_args_dsv4_ratio4_shift {
    uint32_t width;
};

struct ds4_metal_args_dsv4_compressor_store_one {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos;
    uint32_t ape_type;
};

static inline float dsv4_e4m3fn_value(int i) {
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? float(mant) * 0.001953125f
        : (1.0f + float(mant) * 0.125f) * dsv4_e4m3fn_exp_scale[exp];
}

static inline float dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = min(abs(x), 448.0f);

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = abs(ax - dsv4_e4m3fn_value(best));
        const float next_diff = abs(ax - dsv4_e4m3fn_value(best + 1));
        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best = best + 1;
        }
    }

    return sign * dsv4_e4m3fn_value(best);
}

static inline float dsv4_e2m1fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = min(abs(x), 6.0f);
    int best = 0;
    float best_diff = abs(ax - dsv4_e2m1fn_values[0]);
    for (int i = 1; i < 8; i++) {
        const float diff = abs(ax - dsv4_e2m1fn_values[i]);
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1fn_values[best];
}

// Quantizes the non-RoPE part of a KV row through E4M3FN and writes the
// dequantized value back as float. DS4 uses this to match the FP8 KV-cache
// semantics while keeping the Metal graph's cache buffers float-addressable.
kernel void kernel_dsv4_fp8_kv_quantize_f32(
        constant ds4_metal_args_dsv4_fp8_kv_quantize & args,
        device  const char * src0,
        device        char * dst,
        threadgroup  float * scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    if ((int64_t) row >= n_rows) {
        return;
    }

    const int64_t i1 = row % args.ne01;
    const int64_t i2 = (row / args.ne01) % args.ne02;
    const int64_t i3 = row / (args.ne01 * args.ne02);

    device const char * src_base = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device       char * dst_base = dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3;

    const int64_t n_nope = args.ne00 - args.n_rot;

    for (int64_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (tid < 64) {
            v = *((device const float *) (src_base + (off + tid)*args.nb00));
            scratch[tid] = abs(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax = max(scratch[0], 1.0e-4f);
        const float scale = exp2(ceil(log2(amax / 448.0f)));
        if (tid < 64) {
            const float q = dsv4_e4m3fn_dequant(clamp(v / scale, -448.0f, 448.0f)) * scale;
            *((device float *) (dst_base + (off + tid)*args.nb0)) = q;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int64_t i = n_nope + tid; i < args.ne00; i += 64) {
        *((device float *) (dst_base + i*args.nb0)) = *((device const float *) (src_base + i*args.nb00));
    }
}

// The official DS4 indexer applies a 128-wide Hadamard rotation and then an
// inplace FP4 activation-simulation pass to both indexer Q and indexer KV.
kernel void kernel_dsv4_indexer_hadamard_fp4_f32(
        constant ds4_metal_args_dsv4_indexer_qat & args,
        device   char  * x,
        threadgroup float * scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.n_rows || args.head_dim != 128u || tid >= 128u) {
        return;
    }

    threadgroup float *vals = scratch;
    threadgroup float *absbuf = scratch + 128;
    device float *xr = (device float *)(x + (uint64_t)row * args.row_stride);

    vals[tid] = xr[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            const uint base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            const float a = vals[base];
            const float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float v = vals[tid] * 0.08838834764831845f;
    const uint block = tid >> 5u;
    const uint lane = tid & 31u;
    const uint block_base = block * 32u;
    absbuf[tid] = abs(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = max(absbuf[block_base + lane],
                                            absbuf[block_base + lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float amax = max(absbuf[block_base], 7.052966104933725e-38f);
    const float scale = exp2(ceil(log2(amax / 6.0f)));
    xr[tid] = dsv4_e2m1fn_dequant(clamp(v / scale, -6.0f, 6.0f)) * scale;
}

// Decode-side KV finalizer after RoPE. The normal RoPE kernel intentionally
// remains separate because tiny trigonometric codegen changes can flip later
// sampled tokens. This kernel only fuses the FP8 round-trip for the non-RoPE
// prefix with the F16-rounded raw-cache row used by FlashAttention.
kernel void kernel_dsv4_kv_fp8_store_f32(
        constant ds4_metal_args_dsv4_kv_fp8_store & args,
        device        float * kv,
        device        float * raw_cache,
        threadgroup   float * scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    const int head_dim = args.head_dim;
    const int n_rot = args.n_rot;
    const int n_nope = head_dim - n_rot;
    if (head_dim <= 0 || n_rot < 0 || n_nope < 0 || tid >= 64) {
        return;
    }

    device float * raw = raw_cache + (int64_t)args.raw_row * head_dim;

    for (int off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + (int)tid < n_nope) {
            v = kv[off + tid];
            scratch[tid] = abs(v);
        } else {
            scratch[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax = max(scratch[0], 1.0e-4f);
        const float fp8_scale = exp2(ceil(log2(amax / 448.0f)));
        if (off + (int)tid < n_nope) {
            const float q = dsv4_e4m3fn_dequant(clamp(v / fp8_scale, -448.0f, 448.0f)) * fp8_scale;
            kv[off + tid] = q;
            // Diagnostic only: skip the FP16 round-trip that normally matches the
            // half-typed FlashAttention KV buffer's precision. With this enabled the
            // indexer will see higher-precision raw values than FlashAttention does,
            // which is informative but not a production-ready setting.
#ifdef DS4_METAL_KV_RAW_F32
            raw[off + tid] = q;
#else
            raw[off + tid] = (float)((half)q);
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = n_nope + tid; i < head_dim; i += 64) {
#ifdef DS4_METAL_KV_RAW_F32
        raw[i] = kv[i];
#else
        raw[i] = (float)((half)kv[i]);
#endif
    }
}

// Ratio-4 compression keeps two 4-row halves of recurrent state. After an
// emitted compressed row, the second half becomes the next window's previous
// half. The old encoder expressed this as four generic copies; this DS4-specific
// kernel performs the KV and score copies together.
kernel void kernel_dsv4_ratio4_shift_f32(
        constant ds4_metal_args_dsv4_ratio4_shift & args,
        device float * state_kv,
        device float * state_score,
        uint gid [[thread_position_in_grid]]) {
    const uint n = 4u * args.width;
    if (gid >= n) return;

    state_kv[gid] = state_kv[n + gid];
    state_score[gid] = state_score[n + gid];
}

// One-token compressor frontier update. Decode appends exactly one projected KV
// row and one score row into a small recurrent state. The generic batch helper
// expresses this as APE copy, score add, and two set_rows operations; this
// kernel writes both state tensors directly while preserving the same
// score + APE arithmetic.
kernel void kernel_dsv4_compressor_store_one(
        constant ds4_metal_args_dsv4_compressor_store_one & args,
        device const float * kv,
        device const float * score,
        device const char  * ape,
        device       float * state_kv,
        device       float * state_score,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.width || args.width == 0 || args.ratio == 0) {
        return;
    }

    const uint pos_mod = args.pos % args.ratio;
    const uint dst_row = args.ratio == 4u ? args.ratio + pos_mod : pos_mod;
    const uint dst = dst_row * args.width + gid;
    const uint ape_i = pos_mod * args.width + gid;

    float ape_v;
    if (args.ape_type == 1u) {
        ape_v = (float)(((device const half *)ape)[ape_i]);
    } else {
        ape_v = ((device const float *)ape)[ape_i];
    }

    state_kv[dst] = kv[gid];
    state_score[dst] = score[gid] + ape_v;
}
"""###,
        "dsv4_rope": ###"""
struct ds4_metal_args_dsv4_rope_tail {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  n_dims;
    int32_t  mode;
    int32_t  n_ctx_orig;
    int32_t  inverse;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    bool     src2;
};

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    dims[0] = max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

// Applies DeepSeek V4's partial RoPE: the no-position prefix is copied and only
// the rotated tail is transformed. This is used for Q/K after their projections
// and before writing/reading the attention KV state.
kernel void kernel_dsv4_rope_tail_f32(
        constant ds4_metal_args_dsv4_rope_tail & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        uint  tid   [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const int i1 = tgpig[0];
    const int i2 = tgpig[1];
    const int i3 = tgpig[2];

    const int n_nope = args.ne00 - args.n_dims;
    if (n_nope < 0) {
        return;
    }

    device const int32_t * pos = (device const int32_t *) src1;

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;
    const bool is_neox = args.mode == 2;

    for (int i0 = tid; i0 < args.ne00; i0 += ntg.x) {
        device const char * src_base = src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01;
        device       char * dst_base = dst  + i3*args.nb3  + i2*args.nb2  + i1*args.nb1;

        if (i0 < n_nope) {
            *((device float *) (dst_base + i0*args.nb0)) = *((device const float *) (src_base + i0*args.nb00));
            continue;
        }

        const int r = i0 - n_nope;
        if (is_neox) {
            const int n_half = args.n_dims/2;
            if (r >= n_half) {
                continue;
            }

            const int ic = r;
            const int rel_i0 = 2*ic;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
            // Equivalent to pow(freq_base, k) but expressed through IEEE-754
            // primitives that have tighter precision guarantees than Metal's pow().
            const float theta = theta_base * exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
            const float theta = theta_base * pow(args.freq_base, inv_ndims*rel_i0);
#endif
            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, rel_i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
            if (args.inverse) {
                sin_theta = -sin_theta;
            }

            const int j0 = n_nope + ic;
            const int j1 = n_nope + ic + n_half;
            const float x0 = *((device const float *) (src_base + j0*args.nb00));
            const float x1 = *((device const float *) (src_base + j1*args.nb00));

            *((device float *) (dst_base + j0*args.nb0)) = x0*cos_theta - x1*sin_theta;
            *((device float *) (dst_base + j1*args.nb0)) = x0*sin_theta + x1*cos_theta;
        } else {
            if ((r & 1) != 0) {
                continue;
            }

            const int ic = r/2;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
            const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
            const float theta = theta_base * pow(args.freq_base, inv_ndims*r);
#endif
            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, r, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
            if (args.inverse) {
                sin_theta = -sin_theta;
            }

            const int j0 = n_nope + r;
            const int j1 = j0 + 1;
            const float x0 = *((device const float *) (src_base + j0*args.nb00));
            const float x1 = *((device const float *) (src_base + j1*args.nb00));

            *((device float *) (dst_base + j0*args.nb0)) = x0*cos_theta - x1*sin_theta;
            *((device float *) (dst_base + j1*args.nb0)) = x0*sin_theta + x1*cos_theta;
        }
    }
}
"""###,
        "dsv4_misc": ###"""
struct ds4_metal_args_dsv4_topk_mask {
    int64_t  ne00;
    int64_t  ne01;
    uint64_t nb00;
    uint64_t nb01;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_indexer_weighted_sum {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int64_t  ne10;
    int64_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
    float    scale;
};

struct ds4_metal_args_dsv4_softmax_pool {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_indexed_attention {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t pos0;
    uint32_t window;
    uint32_t ratio;
    uint32_t comp_kv_f16;
    uint32_t pad0;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t raw_row_stride;
    uint64_t comp_row_stride;
    uint64_t topk_token_stride;
    uint64_t dst_token_stride;
    uint64_t dst_head_stride;
    float    scale;
};

struct ds4_metal_args_dsv4_indexer_scores_fused {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t ratio;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t index_row_stride;
    uint64_t score_token_stride;
    float    scale;
};

struct ds4_metal_args_dsv4_router_select_one {
    uint32_t has_bias;
    uint32_t hash_mode;
    uint32_t use_token_buffer;
    uint32_t token;
    uint32_t hash_rows;
};

struct ds4_metal_args_dsv4_directional_steering_project {
    uint32_t width;
    uint32_t rows;
    uint32_t layer;
    uint32_t n_threads;
    float    scale;
};

// Optional directional steering projection.
//
// Each threadgroup owns one 4096-wide token row, computes
// dot(row, direction[layer]), then subtracts scale * direction * dot in-place.
// Positive scales remove a concept direction; negative scales amplify it.  The
// kernel is not used unless a steering file and nonzero scale are provided.
kernel void kernel_dsv4_directional_steering_project_f32(
        constant ds4_metal_args_dsv4_directional_steering_project & args,
        device float *x,
        device const float *directions,
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.rows || args.width == 0) return;

    device float *xr = x + (uint64_t)row * args.width;
    device const float *dir = directions + (uint64_t)args.layer * args.width;
    const uint nth = args.n_threads;

    float sum = 0.0f;
    for (uint i = tid; i < args.width; i += nth) {
        sum += xr[i] * dir[i];
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float coeff = args.scale * scratch[0];
    for (uint i = tid; i < args.width; i += nth) {
        xr[i] -= coeff * dir[i];
    }
}

// Decode-only DS4 ratio-4 indexer score builder.  One threadgroup owns one
// compressed row for the current token, stages that 128-wide row once, then
// walks the 64 indexer heads in four-head groups.  This avoids materializing the
// intermediate [compressed rows x heads] score matrix used by the generic
// matvec + weighted-sum path.
kernel void kernel_dsv4_indexer_score_one_direct(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (row >= args.n_comp || args.n_head != 64u || args.head_dim != 128u) {
        return;
    }

    threadgroup float *ktg = shared;        // [128]
    threadgroup float *psum = ktg + 128u;   // [4]

    if (tid < 128u) {
        device const float *krow = (device const float *)(index_comp +
            (uint64_t)row * args.index_row_stride);
        ktg[tid] = krow[tid];
    }

    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head0 = 0; head0 < 64u; head0 += 4u) {
        const uint head = head0 + (uint)sg;
        device const float4 *q4 = (device const float4 *)(q +
            (uint64_t)head * args.q_head_stride);
        threadgroup const float4 *k4 = (threadgroup const float4 *)ktg;

        float s = dot(q4[lane], k4[lane]);
        s = simd_sum(s);
        if (lane == 0) {
            device const float *w = (device const float *)weights;
            psum[sg] = max(s, 0.0f) * (w[head] * args.scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            acc += psum[0];
            acc += psum[1];
            acc += psum[2];
            acc += psum[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        device float *dst = (device float *)scores;
        dst[row] = acc;
    }
}

// Decode router post-processing for one token. The selected expert ids are
// already known; this gathers their probabilities, normalizes by the selected
// sum, clamps the denominator like the reference path, and applies DS4's 1.5
// expert-weight scale in one tiny dispatch.
kernel void kernel_dsv4_router_weights_one(
        device const char *probs,
        device const char *selected,
        device       char *weights,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= 6) return;

    device const float *p = (device const float *)probs;
    device const int   *s = (device const int *)selected;

    float sum = 0.0f;
    for (uint i = 0; i < 6; i++) {
        sum += p[s[i]];
    }
    sum = max(sum, 6.103515625e-5f);

    device float *w = (device float *)weights;
    w[tid] = p[s[tid]] / sum * 1.5f;
}

// Decode router selection for one token after the existing
// sqrt(softplus(logit)) probability kernel has run. Bias affects only top-k
// selection. Route-weight normalization deliberately stays in the old one-token
// kernel: even tiny denominator-order changes here are amplified by 43 MoE
// layers, so this kernel only replaces the selection work.
kernel void kernel_dsv4_router_finalize_one(
        constant ds4_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256) return;

    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + 256);
    const float p = probs[tid];
    sel_scores[tid] = args.has_bias ? p + bias[tid] : p;
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (args.hash_mode) {
        if (tid == 0) {
            const uint token = args.use_token_buffer ? (uint)tokens[0] : args.token;
            const uint row = min(token, args.hash_rows - 1u);
            device const int32_t *src = hash + row * 6u;
            for (uint i = 0; i < 6; i++) {
                selected[i] = src[i];
            }
        }
    } else {
        for (uint k = 2; k <= 256; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                const uint other = tid ^ j;
                if (other > tid) {
                    if ((tid & k) == 0) {
                        if (sel_scores[(uint)idx[tid]] < sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    } else {
                        if (sel_scores[(uint)idx[tid]] > sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (tid < 6) {
            selected[tid] = idx[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Fills the dense compressed-attention mask with -inf. The selected top-k rows
// are enabled by kernel_dsv4_topk_mask_scatter in a second ordered dispatch.
kernel void kernel_dsv4_topk_mask(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * topk,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    (void)topk;
    *((device float *) (dst + ic*args.nb0 + it*args.nb1)) = -INFINITY;
}

// Enables the selected compressed rows in the dense mask. This replaces the
// old O(n_comp * n_tokens * top_k) membership test with O(top_k * n_tokens)
// writes while preserving exactly the same 0/-inf mask consumed by attention.
kernel void kernel_dsv4_topk_mask_scatter(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * topk,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne00 * args.ne01;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ik = gid % args.ne00;
    const int64_t it = gid / args.ne00;
    const int32_t idx = *((device const int32_t *) (topk + ik*args.nb00 + it*args.nb01));
    if (idx >= 0 && (int64_t)idx < args.ne0) {
        *((device float *) (dst + (int64_t)idx*args.nb0 + it*args.nb1)) = 0.0f;
    }
}

// Sorts each token's selected compressed rows by row id. The indexer selects by
// score, but attention scans compressed K/V in cache order in the dense graph.
// Sorting preserves that order while still letting the indexed attention kernel
// touch only the selected rows.
kernel void kernel_dsv4_sort_i32_rows_asc(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * src,
        device       char * dst,
        threadgroup int32_t * row_tmp [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const uint top_k = (uint)args.ne00;
    if (row >= (uint)args.ne01 || tid >= top_k) {
        return;
    }

    row_tmp[tid] = *((device const int32_t *) (src + (uint64_t)tid*args.nb00 + (uint64_t)row*args.nb01));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= top_k; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint other = tid ^ j;
            if (other > tid && other < top_k) {
                const int32_t a = row_tmp[tid];
                const int32_t b = row_tmp[other];
                const bool up = (tid & k) == 0;
                if ((up && a > b) || (!up && a < b)) {
                    row_tmp[tid] = b;
                    row_tmp[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    *((device int32_t *) (dst + (uint64_t)tid*args.nb00 + (uint64_t)row*args.nb01)) = row_tmp[tid];
}

static inline void dsv4_attend_f32_row_as_f16(
        device const char *kv,
        uint64_t row_stride,
        uint row,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    device const float4 *kv4 = (device const float4 *)(kv + (uint64_t)row * row_stride);
    const half4 k0 = (half4)kv4[lane +  0];
    const half4 k1 = (half4)kv4[lane + 32];
    const half4 k2 = (half4)kv4[lane + 64];
    const half4 k3 = (half4)kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_f32_row_as_f16(
        threadgroup const float4 *kv4,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const half4 k0 = (half4)kv4[lane +  0];
    const half4 k1 = (half4)kv4[lane + 32];
    const half4 k2 = (half4)kv4[lane + 64];
    const half4 k3 = (half4)kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_f32_row_as_f16_at(
        threadgroup const float4 *kv4,
        uint row_in_tg,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    dsv4_attend_shared_f32_row_as_f16(kv4 + row_in_tg * 128u,
                                      q0, q1, q2, q3,
                                      scale,
                                      lane,
                                      M, S,
                                      o0, o1, o2, o3);
}

static inline void dsv4_attend_shared_h4_row(
        threadgroup const half4 *kv4,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const half4 k0 = kv4[lane +  0];
    const half4 k1 = kv4[lane + 32];
    const half4 k2 = kv4[lane + 64];
    const half4 k3 = kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_h4_row_at(
        threadgroup const half4 *kv4,
        uint row_in_tg,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    dsv4_attend_shared_h4_row(kv4 + row_in_tg * 128u,
                              q0, q1, q2, q3,
                              scale,
                              lane,
                              M, S,
                              o0, o1, o2, o3);
}

static inline half4 dsv4_load_cache_h4(
        device const char *kv,
        uint64_t row_stride,
        uint row,
        uint col,
        bool f16_rows) {
    device const char *base = kv + (uint64_t)row * row_stride;
    if (f16_rows) {
        return ((device const half4 *)base)[col];
    }
    return (half4)((device const float4 *)base)[col];
}

static inline void dsv4_attend_sink(
        float score,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;
    M = new_m;
}

// DS4 ratio-4 indexed mixed attention. It replaces the dense top-k mask path:
// the threadgroup covers one token and eight heads. Top-k rows and local raw
// rows are the same for all heads of a token, so K/V is staged once in
// threadgroup memory and reused by the eight simdgroups. It keeps the DS4 F16
// attention rounding by casting Q/K/V to half before the dot/value update.
kernel void kernel_dsv4_indexed_mixed_attention_heads8(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos = first; pos <= last; pos++) {
            const uint logical = pos - first_raw_pos;
            const uint row = (args.raw_start + logical) % args.raw_cap;
            device const float4 *src = (device const float4 *)(raw_kv +
                (uint64_t)row * args.raw_row_stride);
            if (tid < 128) kv_shared[tid] = (half4)src[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dsv4_attend_shared_h4_row(kv_shared,
                                      q0, q1, q2, q3,
                                      args.scale,
                                      lane,
                                      M, S,
                                      o0, o1, o2, o3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    for (uint i = 0; i < args.top_k; i++) {
        const int32_t idx = row_topk[i];
        if (idx < 0) {
            continue;
        }
        if ((uint)idx >= visible) {
            break;
        }
        if (tid < 128) {
            kv_shared[tid] = dsv4_load_cache_h4(comp_kv,
                                                args.comp_row_stride,
                                                (uint)idx,
                                                tid,
                                                args.comp_kv_f16 != 0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dsv4_attend_shared_h4_row(kv_shared,
                                  q0, q1, q2, q3,
                                  args.scale,
                                  lane,
                                  M, S,
                                  o0, o1, o2, o3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dsv4_attend_sink(((device const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    device float4 *dst4 = (device float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
    dst4[lane +  0] = o0 * inv_s;
    dst4[lane + 32] = o1 * inv_s;
    dst4[lane + 64] = o2 * inv_s;
    dst4[lane + 96] = o3 * inv_s;
}

// Decode specialization of kernel_dsv4_indexed_mixed_attention_heads8.
// Generation attends one token at a time, so the ratio-4 indexed path spends a
// visible amount of time repeatedly staging the same K/V row for the eight
// heads in a group. This variant stages sixteen selected rows at once and then
// consumes them sequentially, preserving the row order and online softmax math
// while cutting threadgroup barriers in the long top-k scan.
kernel void kernel_dsv4_indexed_mixed_attention_heads8_rb16(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos0 = first; pos0 <= last; pos0 += 16u) {
            const uint n_rows = min(16u, last - pos0 + 1u);
            for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
                const uint r = off >> 7;
                const uint c = off & 127u;
                const uint logical = pos0 + r - first_raw_pos;
                const uint row = (args.raw_start + logical) % args.raw_cap;
                device const float4 *src = (device const float4 *)(raw_kv +
                    (uint64_t)row * args.raw_row_stride);
                kv_shared[off] = (half4)src[c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < n_rows; r++) {
                dsv4_attend_shared_h4_row_at(kv_shared,
                                             r,
                                             q0, q1, q2, q3,
                                             args.scale,
                                             lane,
                                             M, S,
                                             o0, o1, o2, o3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    bool stop = false;
    for (uint i = 0; i < args.top_k && !stop; i += 16u) {
        uint rows[16];
        uint n_rows = 0;
        for (uint j = 0; j < 16u && i + j < args.top_k; j++) {
            const int32_t idx = row_topk[i + j];
            if (idx < 0) {
                continue;
            }
            if ((uint)idx >= visible) {
                stop = true;
                break;
            }
            rows[n_rows++] = (uint)idx;
        }
        if (n_rows == 0) {
            continue;
        }
        for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
            const uint r = off >> 7;
            const uint c = off & 127u;
            kv_shared[off] = dsv4_load_cache_h4(comp_kv,
                                                args.comp_row_stride,
                                                rows[r],
                                                c,
                                                args.comp_kv_f16 != 0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < n_rows; r++) {
            dsv4_attend_shared_h4_row_at(kv_shared,
                                         r,
                                         q0, q1, q2, q3,
                                         args.scale,
                                         lane,
                                         M, S,
                                         o0, o1, o2, o3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dsv4_attend_sink(((device const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    device float4 *dst4 = (device float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
    dst4[lane +  0] = o0 * inv_s;
    dst4[lane + 32] = o1 * inv_s;
    dst4[lane + 64] = o2 * inv_s;
    dst4[lane + 96] = o3 * inv_s;
}

static inline float dsv4_indexer_dot128_shared_q(
        float4 c0,
        float4 c1,
        float4 c2,
        float4 c3,
        threadgroup const float4 *q4,
        ushort lane) {
    float sum = 0.0f;
    if (lane < 8) {
        const ushort ib = lane >> 1;
        const ushort il = lane & 1;
        const ushort base = ib*8 + il*4;
        sum += dot(c0, q4[base + 0]);
        sum += dot(c1, q4[base + 1]);
        sum += dot(c2, q4[base + 2]);
        sum += dot(c3, q4[base + 3]);
    }
    return simd_sum(sum);
}

// Tiled prefill score builder for the sparse-compressed attention indexer.
//
// The kernel covers an 8-token by 32-compressed-row rectangle: K is copied into
// threadgroup memory once, then reused for all 64 indexer heads, while simdgroup
// matrix multiply computes each 8x8 score subtile.
//
// It still writes the exact score matrix consumed by top-k:
//
//     score[t,c] = sum_h relu(dot(Q[t,h], K[c])) * W[t,h] * scale
//
// Causal masking is applied on store so invisible compressed rows become -inf.
kernel void kernel_dsv4_indexer_scores_tiled_f32(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    threadgroup float *qtg = shared;             // [8][128]
    threadgroup float *ktg = qtg + TM*D;         // [32][128]
    threadgroup float *dot = ktg + TN*D;         // [8][32]

    const uint last_token = min(t0 + TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint cc = i / D;
        const uint d = i - cc*D;
        const uint comp = c0 + cc;
        float v = 0.0f;
        if (comp < args.n_comp) {
            device const float *row = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = row[d];
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint row0 = cell0 >> 3;
    const uint row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = t0 + row0;
    const uint token1 = t0 + row1;
    const uint comp0 = c0 + col0;
    const uint comp1 = c0 + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        for (uint i = tid; i < TM*D; i += 128) {
            const uint r = i / D;
            const uint d = i - r*D;
            const uint token = t0 + r;
            float v = 0.0f;
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = qrow[d];
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_float8x8 mq;
            simdgroup_float8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && comp0 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[row0*TN + col0];
            acc0 += max(s, 0.0f) * (w[head] * args.scale);
        }
        if (token1 < args.n_tokens && comp1 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[row1*TN + col1];
            acc1 += max(s, 0.0f) * (w[head] * args.scale);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && comp0 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp0;
        *dst = comp0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && comp1 < args.n_comp) {
        const uint visible = min((args.pos0 + token1 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + comp1;
        *dst = comp1 < visible ? acc1 : -INFINITY;
    }
}

kernel void kernel_dsv4_indexer_scores_tiled(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    // Q/K are staged as half but the dot accumulator and final score remain
    // float. This is the one intentional precision tradeoff in the indexer:
    // the indexer only ranks compressed rows for top-k selection, and long
    // context profiling shows this score matrix dominates the prefill slope.
    threadgroup half *qtg = (threadgroup half *)shared; // [8][128]
    threadgroup half *ktg = qtg + TM*D;                 // [32][128]
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D); // [8][32]

    const uint last_token = min(t0 + TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    // Stage compressed index rows once. Edge columns are zeroed so the matrix
    // loads below can stay regular; guarded stores discard them.
    for (uint i = tid; i < TN*D; i += 128) {
        const uint cc = i / D;
        const uint d = i - cc*D;
        const uint comp = c0 + cc;
        half v = half(0.0f);
        if (comp < args.n_comp) {
            device const float *row = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = half(row[d]);
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint row0 = cell0 >> 3;
    const uint row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = t0 + row0;
    const uint token1 = t0 + row1;
    const uint comp0 = c0 + col0;
    const uint comp1 = c0 + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        // Stage Q for the eight-token tile. Each 8x8 matrix load below reads a
        // contiguous depth block from this layout.
        for (uint i = tid; i < TM*D; i += 128) {
            const uint r = i / D;
            const uint d = i - r*D;
            const uint token = t0 + r;
            half v = half(0.0f);
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && comp0 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[row0*TN + col0];
            acc0 += max(s, 0.0f) * (w[head] * args.scale);
        }
        if (token1 < args.n_tokens && comp1 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[row1*TN + col1];
            acc1 += max(s, 0.0f) * (w[head] * args.scale);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && comp0 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp0;
        *dst = comp0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && comp1 < args.n_comp) {
        const uint visible = min((args.pos0 + token1 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + comp1;
        *dst = comp1 < visible ? acc1 : -INFINITY;
    }
}

#ifdef DS4_METAL_HAS_TENSOR
// Retained full-512 prefill indexer score path.  This is the part of sparse
// compressed attention that maps cleanly to TensorOps: a regular token by
// compressed-row dot tile.  The kernel intentionally leaves top-k selection and
// indexed attention semantics unchanged; all 512 selected rows remain available
// to the later attention kernel.
kernel void kernel_dsv4_indexer_scores_nax(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup half *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]]) {
    constexpr int TM = 16;
    constexpr int TN = 32;
    constexpr int NK = 32;
    constexpr int D  = 128;
    constexpr int NUM_THREADS = 128;

    // The 16-token x 32-row tile was the winning NAX shape in local sweeps.  A
    // wider 64-row compressed tile increased setup/cache pressure and was
    // slower despite doing more work per dispatch.
    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    threadgroup half  *qtg = shared;               // [16][32]
    threadgroup half  *ktg = qtg + TM*NK;          // [32][128]
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D); // [16][32], column-major

    const uint last_token = min(t0 + (uint)TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += NUM_THREADS) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint work = tid; work < TN*D; work += NUM_THREADS) {
        const uint cc = work / D;
        const uint d = work - cc*D;
        const uint comp = c0 + cc;
        half v = half(0.0f);
        if (comp < args.n_comp) {
            device const float *krow = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = half(krow[d]);
        }
        ktg[cc*D + d] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[4];
    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        acc[j] = 0.0f;
    }

    auto tq = tensor(qtg, dextents<int32_t, 2>(NK, TM));
    auto tk = tensor(ktg, dextents<int32_t, 2>(D, TN));
    auto td = tensor(dot, dextents<int32_t, 2>(TM, TN), array<int, 2>({1, TM}));

    matmul2d<
        matmul2d_descriptor(TN, TM, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    for (uint head = 0; head < args.n_head; head++) {
        auto ct = mm.template get_destination_cooperative_tensor<decltype(tk), decltype(tq), float>();
        #pragma unroll
        for (uint16_t i = 0; i < ct.get_capacity(); i++) {
            if (ct.is_valid_element(i)) {
                ct[i] = 0.0f;
            }
        }

        for (uint loop_k = 0; loop_k < D; loop_k += NK) {
            for (uint work = tid; work < TM*NK; work += NUM_THREADS) {
                const uint r = work / NK;
                const uint k = work - r*NK;
                const uint token = t0 + r;
                half v = half(0.0f);
                if (token < args.n_tokens) {
                    device const float *qrow = (device const float *)(q +
                        (uint64_t)token * args.q_token_stride +
                        (uint64_t)head  * args.q_head_stride);
                    v = half(qrow[loop_k + k]);
                }
                qtg[r*NK + k] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            auto mq = tq.slice(0, 0);
            auto mk = tk.slice(loop_k, 0);
            mm.run(mk, mq, ct);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        ct.store(td);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            const uint linear = (uint)tid + j*NUM_THREADS;
            if (linear < TM*TN) {
                const uint r = linear / TN;
                const uint cc = linear - r*TN;
                const uint token = t0 + r;
                if (token < args.n_tokens) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token * args.weights_token_stride);
                    acc[j] += max(dot[cc*TM + r], 0.0f) * (w[head] * args.scale);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        const uint linear = (uint)tid + j*NUM_THREADS;
        if (linear >= TM*TN) {
            continue;
        }
        const uint r = linear / TN;
        const uint cc = linear - r*TN;
        const uint token = t0 + r;
        const uint comp = c0 + cc;
        if (token < args.n_tokens && comp < args.n_comp) {
            const uint visible = min((args.pos0 + token + 1u) / args.ratio, args.n_comp);
            device float *dst = (device float *)(scores +
                (uint64_t)token * args.score_token_stride) + comp;
            *dst = comp < visible ? acc[j] : -INFINITY;
        }
    }
}
#endif

// Collapses per-head indexer scores into one score per compressed row using the
// learned head weights. Negative head scores are clipped exactly as DS4 expects.
kernel void kernel_dsv4_indexer_weighted_sum(
        constant ds4_metal_args_dsv4_indexer_weighted_sum & args,
        device const char * scores,
        device const char * weights,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    float acc = 0.0f;
    for (int64_t ih = 0; ih < args.ne02; ++ih) {
        const float s = *((device const float *) (scores  + ic*args.nb00 + it*args.nb01 + ih*args.nb02));
        const float w = *((device const float *) (weights + ih*args.nb10 + it*args.nb11));
        acc += max(s, 0.0f) * (w * args.scale);
    }

    *((device float *) (dst + ic*args.nb0 + it*args.nb1)) = acc;
}

// Fused softmax-weighted pooling of compressed KV rows. It is used when several
// compressor rows are present; the one-row case deliberately follows the
// unfused softmax/mul/sum graph in Objective-C to keep identical reductions.
kernel void kernel_dsv4_softmax_pool(
        constant ds4_metal_args_dsv4_softmax_pool & args,
        device const char * kv,
        device const char * score,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t id = gid % args.ne0;
    const int64_t ic = gid / args.ne0;

    float max_s = -INFINITY;
    for (int64_t ir = 0; ir < args.ne00; ++ir) {
        const float s = *((device const float *) (score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));
        max_s = max(max_s, s);
    }

    float sum = 0.0f;
    float acc = 0.0f;
    for (int64_t ir = 0; ir < args.ne00; ++ir) {
        const float s = *((device const float *) (score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));
        const float w = exp(s - max_s);
        const float v = *((device const float *) (kv + ir*args.nb00 + id*args.nb01 + ic*args.nb02));
        sum += w;
        acc += v*w;
    }

    *((device float *) (dst + id*args.nb0 + ic*args.nb1)) = acc/sum;
}
"""###,
        "argsort": ###"""
struct ds4_metal_args_argsort {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
};

struct ds4_metal_args_argsort_merge {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
};

typedef void (argsort_t)(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Sort one float row into an index row. DS4 only exports the descending
// instance because router and indexer selection both need top-k order.
template<ds4_sort_order order>
kernel void kernel_argsort_f32_i32(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= args.ne00 ||
                       (shmem_i32[ixj] <  args.ne00 && (order == DS4_SORT_ORDER_ASC ?
                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]] :
                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= args.ne00 ||
                       (shmem_i32[col] <  args.ne00 && (order == DS4_SORT_ORDER_ASC ?
                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]] :
                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < args.ne0 && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shmem_i32[col];
    }
}

// Host-visible sort variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC>;

typedef void (argsort_merge_t)(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Merges sorted index runs produced by kernel_argsort_f32_i32. In the DS4 graph
// this finishes top-k over router or compressed-attention score rows.
template<ds4_sort_order order>
kernel void kernel_argsort_merge_f32_i32(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    const int im  = tgpig[0] / args.ne01;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    const int start = im * (2 * args.len);

    const int len0 = MIN(args.len, MAX(0, args.ne0 - (int)(start)));
    const int len1 = MIN(args.len, MAX(0, args.ne0 - (int)(start + args.len)));

    const int total = len0 + len1;

    device const int32_t * tmp0 = tmp + start
        + i01*args.ne0
        + i02*args.ne0*args.ne01
        + i03*args.ne0*args.ne01*args.ne02;

    device const int32_t * tmp1 = tmp0 + args.len;

    dst += start
        + i01*args.top_k
        + i02*args.top_k*args.ne01
        + i03*args.top_k*args.ne01*args.ne02;

    device const float * src0_row = (device const float *)(src0
        + args.nb01*i01
        + args.nb02*i02
        + args.nb03*i03);

    if (total == 0) {
        return;
    }

    const int chunk = (total + ntg.x - 1) / ntg.x;

    const int k0 = tpitg.x * chunk;
    const int k1 = MIN(MIN(k0 + chunk, total), args.top_k);

    if (k0 >= args.top_k) {
        return;
    }

    if (k0 >= total) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = MIN(k0, len0);

    // binary-search partition (i, j) such that i + j = k
    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == DS4_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    // keep the merge fronts into registers
    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == DS4_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
}

// Host-visible merge variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_merge_f32_i32_desc")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC>;
"""###,
        "cpy": ###"""
struct ds4_metal_args_cpy {
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Typed copy/conversion between graph tensors. DS4 uses this for layout
// materialization and F32/F16 conversions at graph boundaries such as KV/cache
// packing and compressor pooling.
template<typename T0, typename T1>
kernel void kernel_cpy_t_t(
        constant ds4_metal_args_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tiitg/ntg[0];
    const int iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device T1 * dst_data = (device T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = iw0*ntg[0] + tiitg%ntg[0]; i00 < args.ne00; ) {
        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        dst_data[i00] = (T1) src[0];
        break;
    }
}

typedef decltype(kernel_cpy_t_t<float, float>) kernel_cpy_t;
// Host-visible copy/conversion variants used by the DS4 graph.
template [[host_name("kernel_cpy_f32_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<float, float>;
template [[host_name("kernel_cpy_f32_f16")]] kernel kernel_cpy_t kernel_cpy_t_t<float, half>;
template [[host_name("kernel_cpy_f16_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<half, float>;
"""###,
        "concat": ###"""
// DS4 Metal concat kernel used by the graph.

struct ds4_metal_args_concat {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  dim;
};

// Concatenates two float tensors along one dimension. In DS4 this is a graph
// utility for assembling attention inputs with exactly the same tensor layout
// expected by the downstream kernels.
kernel void kernel_concat(
        constant ds4_metal_args_concat & args,
        device  const char * src0,
        device  const char * src1,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    int o[4] = {0, 0, 0, 0};
    o[args.dim] = args.dim == 0 ? args.ne00 : (args.dim == 1 ? args.ne01 : (args.dim == 2 ? args.ne02 : args.ne03));

    device const float * x;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            x = (device const float *)(src0 + (i3       )*args.nb03 + (i2       )*args.nb02 + (i1       )*args.nb01 + (i0       )*args.nb00);
        } else {
            x = (device const float *)(src1 + (i3 - o[3])*args.nb13 + (i2 - o[2])*args.nb12 + (i1 - o[1])*args.nb11 + (i0 - o[0])*args.nb10);
        }

        device float * y = (device float *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

        *y = *x;
    }
}
"""###,
        "get_rows": ###"""
// DS4 Metal get-rows kernel.

struct ds4_metal_args_get_rows {
    int32_t  ne00t;
    int32_t  ne00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Gathers embedding/table rows by integer ids. DS4 uses this for token
// embeddings and small indexed tables such as router/hash lookup outputs.
template<typename T0, typename T>
kernel void kernel_get_rows_f(
        constant ds4_metal_args_get_rows & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3               tgpig[[threadgroup_position_in_grid]],
        ushort              tiitg[[thread_index_in_threadgroup]],
        ushort3             ntg [[threads_per_threadgroup]]) {
    const int32_t iw0 = tgpig.x/args.ne10;
    const int32_t i10 = tgpig.x%args.ne10;
    const int32_t i11 = tgpig.y;
    const int32_t i12 = tgpig.z;

    const int32_t r = ((const device int32_t *) (src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];

    const int32_t i02 = i11;
    const int32_t i03 = i12;

    auto psrc = (const device T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + r*args.nb01);
    auto pdst = (      device T  *) (dst  + i12*args.nb3  + i11*args.nb2  + i10*args.nb1);

    for (int ind = iw0*ntg.x + tiitg; ind < args.ne00t;) {
        pdst[ind] = psrc[ind];

        break;
    }
}

typedef decltype(kernel_get_rows_f<float, float>) get_rows_f_t;

// Host-visible gather variants for F32, F16, and I32 tables.
template [[host_name("kernel_get_rows_f32")]] kernel get_rows_f_t kernel_get_rows_f<float, float>;
template [[host_name("kernel_get_rows_f16")]] kernel get_rows_f_t kernel_get_rows_f<half, float>;
template [[host_name("kernel_get_rows_i32")]] kernel get_rows_f_t kernel_get_rows_f<int32_t, int32_t>;
"""###,
        "sum_rows": ###"""
// DS4 Metal row-sum kernel.

#define FC_SUM_ROWS 1400

#define OP_SUM_ROWS_NUM_SUM_ROWS 10
#define OP_SUM_ROWS_NUM_MEAN     11

struct ds4_metal_args_sum_rows {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

static inline float sum(float x) {
    return x;
}

static inline float sum(float4 x) {
    return x[0] + x[1] + x[2] + x[3];
}

constant short FC_sum_rows_op [[function_constant(FC_SUM_ROWS + 0)]];

// Reduces each row to a sum or mean. DS4 mainly uses the sum form to preserve
// the compressor-pooling graph boundary in the single-compressor-row case.
template <typename T0, typename T>
kernel void kernel_sum_rows_impl(
        constant ds4_metal_args_sum_rows & args,
        device const char * src0,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
#define FC_OP  FC_sum_rows_op

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    threadgroup T0 * shmem_t = (threadgroup T0 *) shmem;

    if (sgitg == 0) {
        shmem_t[tiisg] = 0.0f;
    }

    device const T0 * src_row = (device const T0 *) (src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);
    device       T  * dst_row = (device       T  *) (dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3);

    T0 sumf = T0(0.0f);

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        sumf += src_row[i0];
    }

    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_t[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_t[tiisg];
    sumf = simd_sum(sumf);

    if (tpitg.x == 0) {
        if (FC_OP == OP_SUM_ROWS_NUM_MEAN) {
            if (is_same<float4, T0>::value) {
                dst_row[0] = sum(sumf) / (4*args.ne00);
            } else {
                dst_row[0] = sum(sumf) / args.ne00;
            }
        } else {
            dst_row[0] = sum(sumf);
        }
    }

#undef FC_OP
}

typedef decltype(kernel_sum_rows_impl<float, float>) kernel_sum_rows_t;

// Host-visible F32 row reduction used by compressor pooling.
template [[host_name("kernel_sum_rows_f32_f32")]] kernel kernel_sum_rows_t kernel_sum_rows_impl<float, float>;
"""###,
        "softmax": ###"""
// DS4 Metal softmax kernel used by the compressor pooling compatibility path.
// The single-compressed-row path is intentionally left as soft_max -> mul ->
// sum_rows instead of using the fused dsv4_softmax_pool kernel.

struct ds4_metal_args_soft_max {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
};

// Row softmax for score matrices. DS4 uses it in the literal one-compressor-row
// path where preserving the original graph operation boundary avoids drift.
template<typename T>
kernel void kernel_soft_max(
        constant ds4_metal_args_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float * psrc0 =                (device const float *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const     T * pmask = src1 != src0 ? (device const T *    ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float * psrc2 = src2 != src0 ? (device const float *) (src2)                                                 : nullptr;
    device       float * pdst  =                (device       float *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    float lmax = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        lmax = MAX(lmax, psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        const float exp_psrc0 = exp((psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}

// Vectorized float4 row softmax for contiguous score rows whose length is a
// multiple of four; used by the same DS4 compressor/indexer graph path.
template<typename T>
kernel void kernel_soft_max_4(
        constant ds4_metal_args_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float4 * psrc4 =                (device const float4 *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const      T * pmask = src1 != src0 ? (device const T *     ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float *  psrc2 = src2 != src0 ? (device const float * ) (src2)                                                 : nullptr;
    device       float4 * pdst4 =                (device       float4 *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        lmax4 = fmax(lmax4, psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    float4 lsum4 = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        const float4 exp_psrc4 = exp((psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        pdst4[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>)    kernel_soft_max_t;
typedef decltype(kernel_soft_max_4<float4>) kernel_soft_max_4_t;

// Host-visible F32 softmax variants used by compressor pooling.
template [[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t   kernel_soft_max<float>;
template [[host_name("kernel_soft_max_f32_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<float4>;
"""###,
        "repeat": ###"""
// DS4 Metal repeat kernel used for HC embedding expansion.

struct ds4_metal_args_repeat {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Repeats a source row into the HC channel dimension. DS4 uses this when the
// token embedding has to become an HC activation block before layer 0.
template<typename T>
kernel void kernel_repeat(
        constant ds4_metal_args_repeat & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    const int i03 = i3%args.ne03;
    const int i02 = i2%args.ne02;
    const int i01 = i1%args.ne01;

    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;
    device       char * dst_ptr  = dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i00 = i0%args.ne00;
        *((device T *)(dst_ptr + i0*args.nb0)) = *((device T *)(src0_ptr + i00*args.nb00));
    }
}

typedef decltype(kernel_repeat<float>) kernel_repeat_t;

// Host-visible F32 repeat used for HC expansion of embeddings.
template [[host_name("kernel_repeat_f32")]] kernel kernel_repeat_t kernel_repeat<float>;
"""###,
        "glu": ###"""
struct ds4_metal_args_glu {
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    float    alpha;
    float    limit;
};

// SwiGLU activation for the FFN inner state. DS4 clamps the shared expert with
// the same swiglu_limit used by routed experts.
kernel void kernel_swiglu_f32(
        constant ds4_metal_args_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * src0_row = (device const float *) ((device const char *) src0 + tgpig*args.nb01) + args.i00;
    device const float * src1_row = (device const float *) ((device const char *) src1 + tgpig*args.nb11) + args.i10;
    device       float * dst_row  = (device       float *) ((device       char *) dst  + tgpig*args.nb1);

    for (int i0 = tpitg; i0 < args.ne0; i0 += ntg) {
        float x0 = src0_row[i0];
        float x1 = src1_row[i0];
        if (args.limit > 1.0e-6f) {
            x0 = min(x0, args.limit);
            x1 = clamp(x1, -args.limit, args.limit);
        }

        const float silu = x0 / (1.0f + exp(-x0));

        dst_row[i0] = silu*x1*args.alpha;
    }
}
"""###,
        "norm": ###"""
struct ds4_metal_args_norm {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
};

// RMSNorm over one activation row, optionally fusing the learned weight
// multiply. DS4 calls this before attention, before the FFN, and for plain
// diagnostics that need normalized but unweighted rows.
template <typename T, short F>
kernel void kernel_rms_norm_fuse_impl(
        constant ds4_metal_args_norm & args,
        device const char * src0,
        device const char * src1_0,
        device const char * src1_1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const T * x = (device const T *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);

    device const T * f0 = (device const T *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device const T * f1 = (device const T *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    float sumf = 0.0f;

    // parallel sum
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/args.ne00;
    const float scale = 1.0f/sqrt(mean + args.eps);

    device T * y = (device T *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        if (F == 1) {
            y[i00] = (x[i00]*scale);
        }
        if (F == 2) {
            y[i00] = (x[i00]*scale)*f0[i00];
        }
        if (F == 3) {
            y[i00] = (x[i00]*scale)*f0[i00] + f1[i00];
        }
    }
}

typedef decltype(kernel_rms_norm_fuse_impl<float4, 1>) kernel_rms_norm_fuse_t;

// Host-visible RMSNorm variants: plain norm and norm multiplied by weight.
template [[host_name("kernel_rms_norm_f32_4")]]     kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 1>;
template [[host_name("kernel_rms_norm_mul_f32_4")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float4, 2>;

struct ds4_metal_args_qkv_rms_norm {
    int32_t  q_n;
    int32_t  q_n4;
    int32_t  kv_n;
    int32_t  kv_n4;
    uint64_t q_row_stride;
    uint64_t kv_row_stride;
    float    eps;
};

// Normalizes DS4's q-lora row and KV row in one dispatch.  The two reductions
// deliberately mirror kernel_rms_norm_mul_f32_4: Q uses the full 256-thread
// row shape for 1024 floats, while KV only has work in the first 128 lanes for
// its 512 floats.  This keeps the q/kv normalization math aligned with the
// standalone kernels while removing one tiny launch from the attention setup.
kernel void kernel_dsv4_qkv_rms_norm_f32_4(
        constant ds4_metal_args_qkv_rms_norm & args,
        device const float4 * q_src,
        device const float4 * q_weight,
        device       float4 * q_dst,
        device const float4 * kv_src,
        device const float4 * kv_weight,
        device       float4 * kv_dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const uint row = tgpig.x;
    const bool kv_task = tgpig.y != 0;
    const int n = kv_task ? args.kv_n : args.q_n;
    const int n4 = kv_task ? args.kv_n4 : args.q_n4;
    const uint64_t row_stride4 = (kv_task ? args.kv_row_stride : args.q_row_stride) / sizeof(float4);

    device const float4 * x = kv_task ? kv_src + row * row_stride4 : q_src + row * row_stride4;
    device const float4 * w = kv_task ? kv_weight : q_weight;
    device       float4 * y = kv_task ? kv_dst + row * row_stride4 : q_dst + row * row_stride4;

    float sumf = 0.0f;
    for (int i = tpitg.x; i < n4; i += ntg.x) {
        const float4 v = x[i];
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

#ifdef DS4_METAL_NORM_RSQRT_DISABLE
    // Match the formula used by kernel_rms_norm_fuse_impl above so both RMSNorm
    // entry points produce bit-identical scales. Hardware rsqrt() and 1.0f/sqrt()
    // can differ by ~1 ULP and that difference compounds across 43 layers.
    const float scale = 1.0f / sqrt(sumf / float(n) + args.eps);
#else
    const float scale = rsqrt(sumf / float(n) + args.eps);
#endif

    for (int i = tpitg.x; i < n4; i += ntg.x) {
        y[i] = (x[i] * scale) * w[i];
    }
}
"""###,
        "bin": ###"""
struct ds4_metal_args_bin {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    uint64_t offs;
    uint64_t o1[8];
};

constant short FC_bin_op [[function_constant(FC_BIN + 0)]];
constant short FC_bin_f  [[function_constant(FC_BIN + 1)]];
constant bool  FC_bin_rb [[function_constant(FC_BIN + 2)]];
constant bool  FC_bin_cb [[function_constant(FC_BIN + 3)]];

// Generic binary elementwise op with compile-time operation and broadcast
// modes. DS4 currently instantiates this as add, multiply, scalar multiply, and
// row division in the static graph.
template <typename T0, typename T1, typename T>
kernel void kernel_bin_fuse_impl(
        constant ds4_metal_args_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
#define FC_OP FC_bin_op
#define FC_F  FC_bin_f
#define FC_RB FC_bin_rb
#define FC_CB FC_bin_cb

    if (FC_RB) {
        const uint i0 = tgpig.y*args.ne00 + tgpig.x;
        const uint i1 = FC_CB ? tgpig.x%args.ne10 : tgpig.x;

        device const T0 * src0_row = (device const T0 *) (src0);
        device       T  * dst_row  = (device       T  *) (dst);

        if (FC_F == 1) {
            device const T1 * src1_row = (device const T1 *) (src1 + args.o1[0]);

            if (FC_OP == 0) {
                dst_row[i0] = src0_row[i0] + src1_row[i1];
            }

            if (FC_OP == 1) {
                dst_row[i0] = src0_row[i0] - src1_row[i1];
            }

            if (FC_OP == 2) {
                dst_row[i0] = src0_row[i0] * src1_row[i1];
            }

            if (FC_OP == 3) {
                dst_row[i0] = src0_row[i0] / src1_row[i1];
            }
        } else {
            T0 res = src0_row[i0];

            if (FC_OP == 0) {
                FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                    res += ((device const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            if (FC_OP == 1) {
                FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                    res -= ((device const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            if (FC_OP == 2) {
                FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                    res *= ((device const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            if (FC_OP == 3) {
                FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                    res /= ((device const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            dst_row[i0] = res;
        }
    } else {
        const int i03 = tgpig.z;
        const int i02 = tgpig.y;
        const int i01 = tgpig.x;

        if (i01 >= args.ne01) {
            return;
        }

        const int i13 = i03%args.ne13;
        const int i12 = i02%args.ne12;
        const int i11 = i01%args.ne11;

        device const T0 * src0_ptr = (device const T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + args.offs);
        device       T  * dst_ptr  = (device       T  *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1  + args.offs);

        if (FC_F == 1) {
            device const T1 * src1_ptr = (device const T1 *) (src1 + args.o1[0] + i13*args.nb13 + i12*args.nb12 + i11*args.nb11);

            for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
                const int i10 = FC_CB ? i0%args.ne10 : i0;

                if (FC_OP == 0) {
                    dst_ptr[i0] = src0_ptr[i0] + src1_ptr[i10];
                }

                if (FC_OP == 1) {
                    dst_ptr[i0] = src0_ptr[i0] - src1_ptr[i10];
                }

                if (FC_OP == 2) {
                    dst_ptr[i0] = src0_ptr[i0] * src1_ptr[i10];
                }

                if (FC_OP == 3) {
                    dst_ptr[i0] = src0_ptr[i0] / src1_ptr[i10];
                }
            }
        } else {
            device const T1 * src1_ptr[8];
            FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                src1_ptr[j] = (device const T1 *) (src1 + args.o1[j] + i13*args.nb13 + i12*args.nb12 + i11*args.nb11);
            }

            for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
                const int i10 = FC_CB ? i0%args.ne10 : i0;

                T res = src0_ptr[i0];

                if (FC_OP == 0) {
                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                        res += src1_ptr[j][i10];
                    }
                }

                if (FC_OP == 1) {
                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                        res -= src1_ptr[j][i10];
                    }
                }

                if (FC_OP == 2) {
                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                        res *= src1_ptr[j][i10];
                    }
                }

                if (FC_OP == 3) {
                    FOR_UNROLL (short j = 0; j < FC_F; ++j) {
                        res /= src1_ptr[j][i10];
                    }
                }

                dst_ptr[i0] = res;
            }
        }
    }

#undef FC_OP
#undef FC_F
#undef FC_RB
#undef FC_CB
}

typedef decltype(kernel_bin_fuse_impl<float, float, float>) kernel_bin_fuse_t;
// Host-visible F32 binary op; function constants specialize it per use site.
template [[host_name("kernel_bin_fuse_f32_f32_f32")]] kernel kernel_bin_fuse_t kernel_bin_fuse_impl<float, float, float>;
"""###,
        "set_rows": ###"""
// DS4 Metal set-rows kernel used for KV writes.

struct ds4_metal_args_set_rows {
    int32_t  nk0;
    int32_t  ne01;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Scatters rows into the KV cache by token position. DS4 uses this after Q/K/V
// preparation so decode and later prefill chunks can attend to previous tokens.
template<typename T, typename TI>
kernel void kernel_set_rows_f(
        constant ds4_metal_args_set_rows & args,
        device const char  * src0,
        device const char  * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) (src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device T     * dst_row = (      device T     *) ((device char *) dst  + i1*args.nb1   + i02*args.nb2  + i03*args.nb3);
    const device float * src_row = (const device float *) (                src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        dst_row[ind] = (T) src_row[ind];
    }
}

typedef decltype(kernel_set_rows_f<float, int64_t>) set_rows_f_t;

// Host-visible F32/I32 scatter variant used by KV-cache writes.
template [[host_name("kernel_set_rows_f32_i32")]] kernel set_rows_f_t kernel_set_rows_f<float, int32_t>;
"""###,
    ]
}
