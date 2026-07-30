// Laguna S 2.1 — dense Q6_K projection for the legacy recipe (ported from
// the reference laguna-s2.1 branch, head 448d569). Self-contained.
// Part of the concatenated library: see MetalRuntime.kernelFiles for the order.

#include <metal_stdlib>
using namespace metal;

// llama.cpp Q6_K block (256 elements / 210 bytes), spelled with literal sizes
// because the port does not define a shared QK_K macro. Used only by
// kernel_laguna_q6_K_matmul_f32 at the end of this file.
struct block_q6_K {
    uchar ql[128];
    uchar qh[64];
    char scales[16];
    half d;
};

struct ds4_metal_args_laguna_q6_matmul {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_tokens;
    uint32_t pad0;
    uint64_t row_bytes;
};

// Dense Q6_K projection used by Laguna's down projections and output head.
// The quantized arithmetic follows DwarfStar's existing Q6_K routed-down
// implementation, but addresses a single dense matrix directly.
kernel void kernel_laguna_q6_K_matmul_f32(
        constant ds4_metal_args_laguna_q6_matmul &args,
        device const char  *weight,
        device const float *x,
        device float       *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint rows_per_simd = 2u;
    constexpr uint simd_groups = 2u;
    constexpr uint kmask1 = 0x03u;
    constexpr uint kmask2 = 0x0Cu;
    constexpr uint kmask3 = 0x30u;
    constexpr uint kmask4 = 0xC0u;
    constexpr uint qk_k = 256u;

    const uint row0 = (tgpig.x * simd_groups + simd_group) * rows_per_simd;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const int n_blocks = (int)(args.in_dim / qk_k);
    const short tid = (short)(lane / 2u);
    const short ix = (short)(lane & 1u);
    const short ip = (short)(tid / 8);
    const short il = (short)(tid % 8);
    const short l0 = (short)(4 * il);
    const short is = (short)(8 * ip + l0 / 16);
    const short y_offset = (short)(128 * ip + l0);
    const short q_offset_l = (short)(64 * ip + l0);
    const short q_offset_h = (short)(32 * ip + l0);
    device const float *input = x + (uint64_t)token * args.in_dim;
    float sums[rows_per_simd] = {0.0f, 0.0f};
    float yl[16];

    for (int ib = ix; ib < n_blocks; ib += 2) {
        device const float *y = input + (uint64_t)ib * qk_k + y_offset;
        for (short l = 0; l < 4; l++) {
            yl[4 * l + 0] = y[l + 0];
            yl[4 * l + 1] = y[l + 32];
            yl[4 * l + 2] = y[l + 64];
            yl[4 * l + 3] = y[l + 96];
        }

        for (uint r = 0u; r < rows_per_simd && row0 + r < args.out_dim; r++) {
            device const block_q6_K *block =
                (device const block_q6_K *)(weight +
                    (uint64_t)(row0 + r) * args.row_bytes) + ib;
            device const uchar *q1 = block->ql + q_offset_l;
            device const uchar *q2 = q1 + 32;
            device const uchar *qh = block->qh + q_offset_h;
            device const char *sc = block->scales + is;
            float4 part = float4(0.0f);
            for (short l = 0; l < 4; l++) {
                const uint h = (uint)qh[l];
                part[0] += yl[4 * l + 0] *
                    (float)((int)((q1[l] & 0x0Fu) | ((h & kmask1) << 4u)) - 32);
                part[1] += yl[4 * l + 1] *
                    (float)((int)((q2[l] & 0x0Fu) | ((h & kmask2) << 2u)) - 32);
                part[2] += yl[4 * l + 2] *
                    (float)((int)((q1[l] >> 4u) | (h & kmask3)) - 32);
                part[3] += yl[4 * l + 3] *
                    (float)((int)((q2[l] >> 4u) | ((h & kmask4) >> 2u)) - 32);
            }
            sums[r] += (float)block->d *
                (part[0] * (float)sc[0] + part[1] * (float)sc[2] +
                 part[2] * (float)sc[4] + part[3] * (float)sc[6]);
        }
    }

    for (uint r = 0u; r < rows_per_simd && row0 + r < args.out_dim; r++) {
        const float sum = simd_sum(sums[r]);
        if (lane == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + r] = sum;
        }
    }
}

// MARK: - Multi-token routed MoE prefill

// One selected expert is applied to every token that routed to it. Keeping a
// four-token tile inside the kernel lets each quantized weight group serve up
// to four activations before it leaves cache. `app_start` indexes the shared
// token/weight and scratch planes, so separate expert dispatches write
// disjoint rows and may share one command buffer.
struct ds4_metal_args_laguna_moe_prefill {
    uint32_t weight_type;
    uint32_t hidden_width;
    uint32_t input_width;
    uint32_t app_start;
    uint32_t app_count;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

kernel void kernel_laguna_moe_prefill_swiglu_sg(
        constant ds4_metal_args_laguna_moe_prefill &args,
        device const uint  *app_tokens,
        device const float *app_weights,
        device const float *x_all,
        device const uchar *gate_rows,
        device const uchar *up_rows,
        device float       *mids,
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntg   [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x * ntg.y + (uint)sgitg;
    if (row >= args.hidden_width) return;
    const uint row_bytes = glm52_kquant_row_bytes(args.weight_type,
                                                  args.input_width);
    if (row_bytes == 0u) return;
    device const uchar *gate_row =
        gate_rows + (uint64_t)row * row_bytes;
    device const uchar *up_row =
        up_rows + (uint64_t)row * row_bytes;
    const uint groups = args.input_width / 32u;
    for (uint t0 = 0u; t0 < args.app_count; t0 += 4u) {
        const uint tile = min(4u, args.app_count - t0);
        float acc_gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float acc_up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint g = (uint)tiisg; g < groups; g += 32u) {
            for (uint i = 0u; i < tile; i++) {
                const uint app = args.app_start + t0 + i;
                device const float *x = x_all
                    + (uint64_t)app_tokens[app] * args.input_width;
                acc_gate[i] += glm52_dot_kquant_group(
                    args.weight_type, gate_row, x, g);
                acc_up[i] += glm52_dot_kquant_group(
                    args.weight_type, up_row, x, g);
            }
        }
        for (uint i = 0u; i < tile; i++) {
            const float gate = simd_sum(acc_gate[i]);
            const float up = simd_sum(acc_up[i]);
            if (tiisg == 0u) {
                const uint app = args.app_start + t0 + i;
                mids[(uint64_t)app * args.hidden_width + row] =
                    glm52_silu(gate) * up * app_weights[app];
            }
        }
    }
}

kernel void kernel_laguna_moe_prefill_down_sg(
        constant ds4_metal_args_laguna_moe_prefill &args,
        device const float *mids,
        device const uchar *down_rows,
        device float       *contribs,
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntg   [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x * ntg.y + (uint)sgitg;
    if (row >= args.input_width) return;
    const uint row_bytes = glm52_kquant_row_bytes(args.weight_type,
                                                  args.hidden_width);
    if (row_bytes == 0u) return;
    device const uchar *down_row =
        down_rows + (uint64_t)row * row_bytes;
    const uint groups = args.hidden_width / 32u;
    for (uint t0 = 0u; t0 < args.app_count; t0 += 4u) {
        const uint tile = min(4u, args.app_count - t0);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint g = (uint)tiisg; g < groups; g += 32u) {
            for (uint i = 0u; i < tile; i++) {
                const uint app = args.app_start + t0 + i;
                device const float *mid =
                    mids + (uint64_t)app * args.hidden_width;
                acc[i] += glm52_dot_kquant_group(
                    args.weight_type, down_row, mid, g);
            }
        }
        for (uint i = 0u; i < tile; i++) {
            const float total = simd_sum(acc[i]);
            if (tiisg == 0u) {
                const uint app = args.app_start + t0 + i;
                contribs[(uint64_t)app * args.input_width + row] = total;
            }
        }
    }
}
