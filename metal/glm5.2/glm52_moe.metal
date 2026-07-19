// GLM 5.2 — FFN esperti: kernel scalari di riferimento, varianti simdgroup e MoE batched
// Parte della libreria concatenata: vedi MetalRuntime.kernelFiles per l'ordine.

#include <metal_stdlib>
using namespace metal;

// MARK: - Routed expert FFN (K-quant weights)

// Validation kernels for the quantized FFN matvec stages: one thread owns one
// output row and dequantizes its row serially with the exact reference
// pairing (same element order as the CPU dequant references), so the GPU path
// is comparable to the FFN oracle on dequantized weights. Q8_0 covers the
// dense blocks, the shared expert and the output head; the K-quants cover the
// routed experts. Correctness only — the tuned per-quant families come later,
// beside these fixtures.

struct ds4_metal_args_glm52_moe {
    uint32_t weight_type;    // GGUF id: 10=q2_K, 12=q4_K, 13=q5_K, 14=q6_K
    uint32_t row_count;
    uint32_t input_width;    // dot width, multiple of 256
    float    route_weight;   // pair kernel: multiplies the SwiGLU mid
};

// mid[r] = silu(gate_r · x) * (up_r · x) * route_weight — GLM SwiGLU has no
// clamp and the route weight multiplies the mid BEFORE the down projection.
kernel void kernel_glm52_moe_pair_swiglu(
        constant ds4_metal_args_glm52_moe &args,
        device const float *x,
        device const uchar *gate_rows,
        device const uchar *up_rows,
        device float       *mid,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.row_count) return;
    const uint row_bytes = glm52_kquant_row_bytes(args.weight_type,
                                                  args.input_width);
    if (row_bytes == 0u) return;
    const uint64_t offset = (uint64_t)tid * row_bytes;
    const float g = glm52_dot_kquant_row(args.weight_type, gate_rows + offset,
                                         x, args.input_width);
    const float u = glm52_dot_kquant_row(args.weight_type, up_rows + offset,
                                         x, args.input_width);
    mid[tid] = glm52_silu(g) * u * args.route_weight;
}

// out[r] = down_r · mid for ONE expert; the chained validation wrapper sums
// expert contributions host-side.
kernel void kernel_glm52_moe_down(
        constant ds4_metal_args_glm52_moe &args,
        device const float *mid,
        device const uchar *down_rows,
        device float       *out,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.row_count) return;
    const uint row_bytes = glm52_kquant_row_bytes(args.weight_type,
                                                  args.input_width);
    if (row_bytes == 0u) return;
    out[tid] = glm52_dot_kquant_row(args.weight_type,
                                    down_rows + (uint64_t)tid * row_bytes,
                                    mid, args.input_width);
}

kernel void kernel_glm52_moe_pair_swiglu_sg(
        constant ds4_metal_args_glm52_moe &args,
        device const float *x,
        device const uchar *gate_rows,
        device const uchar *up_rows,
        device float       *mid,
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntg   [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x * ntg.y + (uint)sgitg;
    if (row >= args.row_count) return;
    const uint row_bytes = glm52_kquant_row_bytes(args.weight_type,
                                                  args.input_width);
    if (row_bytes == 0u) return;
    device const uchar *gate_row = gate_rows + (uint64_t)row * row_bytes;
    device const uchar *up_row = up_rows + (uint64_t)row * row_bytes;
    const uint groups = args.input_width / 32u;
    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    for (uint g = (uint)tiisg; g < groups; g += 32u) {
        acc_gate += glm52_dot_kquant_group(args.weight_type, gate_row, x, g);
        acc_up += glm52_dot_kquant_group(args.weight_type, up_row, x, g);
    }
    const float gate = simd_sum(acc_gate);
    const float up = simd_sum(acc_up);
    if (tiisg == 0u) {
        mid[row] = glm52_silu(gate) * up * args.route_weight;
    }
}

kernel void kernel_glm52_moe_down_sg(
        constant ds4_metal_args_glm52_moe &args,
        device const float *mid,
        device const uchar *down_rows,
        device float       *out,
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntg   [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x * ntg.y + (uint)sgitg;
    if (row >= args.row_count) return;
    const uint row_bytes = glm52_kquant_row_bytes(args.weight_type,
                                                  args.input_width);
    if (row_bytes == 0u) return;
    device const uchar *row_ptr = down_rows + (uint64_t)row * row_bytes;
    const uint groups = args.input_width / 32u;
    float acc = 0.0f;
    for (uint g = (uint)tiisg; g < groups; g += 32u) {
        acc += glm52_dot_kquant_group(args.weight_type, row_ptr, mid, g);
    }
    const float total = simd_sum(acc);
    if (tiisg == 0u) {
        out[row] = total;
    }
}

// Varianti ACCUMULANTI (out[row] += dot): fondono il residual add nel
// matvec — un dispatch e una passata su out in meno per ogni proiezione
// seguita da un add (attn output, down dense/shared/esperti).
kernel void kernel_glm52_moe_down_acc(
        constant ds4_metal_args_glm52_moe &args,
        device const float *mid,
        device const uchar *down_rows,
        device float       *out,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.row_count) return;
    const uint row_bytes = glm52_kquant_row_bytes(args.weight_type,
                                                  args.input_width);
    if (row_bytes == 0u) return;
    out[tid] += glm52_dot_kquant_row(args.weight_type,
                                     down_rows + (uint64_t)tid * row_bytes,
                                     mid, args.input_width);
}

kernel void kernel_glm52_moe_down_acc_sg(
        constant ds4_metal_args_glm52_moe &args,
        device const float *mid,
        device const uchar *down_rows,
        device float       *out,
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntg   [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x * ntg.y + (uint)sgitg;
    if (row >= args.row_count) return;
    const uint row_bytes = glm52_kquant_row_bytes(args.weight_type,
                                                  args.input_width);
    if (row_bytes == 0u) return;
    device const uchar *row_ptr = down_rows + (uint64_t)row * row_bytes;
    const uint groups = args.input_width / 32u;
    float acc = 0.0f;
    for (uint g = (uint)tiisg; g < groups; g += 32u) {
        acc += glm52_dot_kquant_group(args.weight_type, row_ptr, mid, g);
    }
    const float total = simd_sum(acc);
    if (tiisg == 0u) {
        out[row] += total;
    }
}

// MARK: - MoE batched

// Tutti gli esperti routed di un layer in DUE dispatch (prima: swiglu +
// down + add PER esperto, 18 dispatch per 6 esperti — matvec piccole dove
// l'overhead di lancio e i vuoti fra kernel dominano). Gli offset dei
// record (contigui nell'arena/staging) e i pesi del router viaggiano nella
// struct di argomenti: max 8 esperti, nessun buffer aggiuntivo.
struct ds4_metal_args_glm52_moe_batch {
    uint32_t gate_up_type;
    uint32_t hidden_width;        // righe di mid per esperto
    uint32_t input_width;         // embedding
    uint32_t expert_count;        // esperti routed attivi (<= 8)
    uint32_t up_offset;           // byte: inizio righe up nel record
    uint32_t down_offset;         // byte: inizio righe down nel record
    uint32_t down_type;
    uint32_t pad0;
    uint32_t record_offsets[8];   // base byte di ogni record nel buffer
    uint32_t route_weights[8];    // float bits (peso del router)
};

// Dispatch: tptg (32, NSG, 1); threadgroups (ceil(hidden/NSG), 1, E).
kernel void kernel_glm52_moe_batch_swiglu_sg(
        constant ds4_metal_args_glm52_moe_batch &args,
        device const float *x,
        device const uchar *records,
        device float       *mids,
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntg   [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x * ntg.y + (uint)sgitg;
    const uint expert = tgpig.z;
    if (row >= args.hidden_width || expert >= args.expert_count) return;
    const uint row_bytes = glm52_kquant_row_bytes(args.gate_up_type,
                                                  args.input_width);
    if (row_bytes == 0u) return;
    device const uchar *record = records + args.record_offsets[expert];
    device const uchar *gate_row = record + (uint64_t)row * row_bytes;
    device const uchar *up_row = record + args.up_offset
        + (uint64_t)row * row_bytes;
    const uint groups = args.input_width / 32u;
    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    for (uint g = (uint)tiisg; g < groups; g += 32u) {
        acc_gate += glm52_dot_kquant_group(args.gate_up_type, gate_row,
                                           x, g);
        acc_up += glm52_dot_kquant_group(args.gate_up_type, up_row, x, g);
    }
    const float gate = simd_sum(acc_gate);
    const float up = simd_sum(acc_up);
    if (tiisg == 0u) {
        const float w = as_type<float>(args.route_weights[expert]);
        mids[expert * args.hidden_width + row] =
            glm52_silu(gate) * up * w;
    }
}

// hidden[row] += somma dei contributi down di TUTTI gli esperti: la
// riduzione per corsia attraversa gli esperti prima della simd_sum —
// stessa matematica della sequenza matvec+add per esperto a meno
// dell'ordine delle somme (entro la tolleranza dei test di parità).
// Dispatch: tptg (32, NSG, 1); threadgroups (ceil(embedding/NSG), 1, 1).
kernel void kernel_glm52_moe_batch_down_sg(
        constant ds4_metal_args_glm52_moe_batch &args,
        device const float *mids,
        device const uchar *records,
        device float       *hidden,
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntg   [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x * ntg.y + (uint)sgitg;
    if (row >= args.input_width) return;
    const uint row_bytes = glm52_kquant_row_bytes(args.down_type,
                                                  args.hidden_width);
    if (row_bytes == 0u) return;
    const uint groups = args.hidden_width / 32u;
    float acc = 0.0f;
    for (uint e = 0u; e < args.expert_count; e++) {
        device const uchar *down_row = records + args.record_offsets[e]
            + args.down_offset + (uint64_t)row * row_bytes;
        device const float *mid = mids + e * args.hidden_width;
        for (uint g = (uint)tiisg; g < groups; g += 32u) {
            acc += glm52_dot_kquant_group(args.down_type, down_row, mid, g);
        }
    }
    const float total = simd_sum(acc);
    if (tiisg == 0u) {
        hidden[row] += total;
    }
}

