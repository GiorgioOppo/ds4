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


// MARK: - Prefill fase B multi-token: esperti staged su TUTTI i loro token

// Il percorso per-applicazione paga la rilettura dei pesi dell'esperto una
// volta PER TOKEN e serializza ogni tripla di dispatch sugli scratch
// condivisi. Qui il loop sui token sta DENTRO il kernel (tile da 4): i byte
// del gruppo pesi attraversano la DRAM una volta per tile, e un'unica
// coppia di dispatch copre tutti gli esperti della wave (griglia z).
// La matematica per token è ESATTAMENTE quella dei kernel per-applicazione
// (stesso stride di gruppo per corsia, stessa simd_sum), e la riduzione
// finale somma i contributi in ordine di applicazione ascendente partendo
// dal valore corrente di hidden: il risultato è bit-identico alla sequenza
// matvec+add del percorso legacy.

struct ds4_metal_args_glm52_moe_prefill {
    uint32_t gate_up_type;      // GGUF id dei record gate/up
    uint32_t hidden_width;      // expert hidden (2048)
    uint32_t input_width;       // embedding (6144)
    uint32_t expert_count;      // esperti nella wave
    uint32_t up_offset;         // byte: inizio del blocco up nel record
    uint32_t down_offset;       // byte: inizio del blocco down nel record
    uint32_t down_type;         // GGUF id dei record down
    uint32_t token_entry_count; // voci token della riduzione
};

// mids[app][hidden]: SwiGLU pesato per ogni (esperto, token applicato).
// expert_meta: triple u32 per esperto {record_offset, app_start, app_count};
// app_tokens/app_weights: per applicazione, indice token e peso di route.
// Dispatch: tptg (32, NSG, 1); threadgroups (ceil(hidden/NSG), 1, esperti).
kernel void kernel_glm52_moe_prefill_swiglu_sg(
        constant ds4_metal_args_glm52_moe_prefill &args,
        device const uint  *expert_meta,
        device const uint  *app_tokens,
        device const float *app_weights,
        device const float *x_all,
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
    device const uchar *record = records + expert_meta[3u * expert];
    device const uchar *gate_row = record + (uint64_t)row * row_bytes;
    device const uchar *up_row = record + args.up_offset
        + (uint64_t)row * row_bytes;
    const uint start = expert_meta[3u * expert + 1u];
    const uint count = expert_meta[3u * expert + 2u];
    const uint groups = args.input_width / 32u;
    for (uint t0 = 0u; t0 < count; t0 += 4u) {
        const uint tile = min(4u, count - t0);
        float acc_gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float acc_up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint g = (uint)tiisg; g < groups; g += 32u) {
            for (uint i = 0u; i < tile; i++) {
                device const float *x = x_all
                    + (uint64_t)app_tokens[start + t0 + i]
                        * args.input_width;
                acc_gate[i] += glm52_dot_kquant_group(args.gate_up_type,
                                                      gate_row, x, g);
                acc_up[i] += glm52_dot_kquant_group(args.gate_up_type,
                                                    up_row, x, g);
            }
        }
        for (uint i = 0u; i < tile; i++) {
            const float gate = simd_sum(acc_gate[i]);
            const float up = simd_sum(acc_up[i]);
            if (tiisg == 0u) {
                mids[(uint64_t)(start + t0 + i) * args.hidden_width + row] =
                    glm52_silu(gate) * up * app_weights[start + t0 + i];
            }
        }
    }
}

// contribs[app][embedding]: proiezione down di ogni applicazione, stessa
// forma multi-token del kernel qui sopra (righe su input_width).
// Dispatch: tptg (32, NSG, 1); threadgroups (ceil(embedding/NSG), 1, esperti).
kernel void kernel_glm52_moe_prefill_down_sg(
        constant ds4_metal_args_glm52_moe_prefill &args,
        device const uint  *expert_meta,
        device const float *mids,
        device const uchar *records,
        device float       *contribs,
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntg   [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x * ntg.y + (uint)sgitg;
    const uint expert = tgpig.z;
    if (row >= args.input_width || expert >= args.expert_count) return;
    const uint row_bytes = glm52_kquant_row_bytes(args.down_type,
                                                  args.hidden_width);
    if (row_bytes == 0u) return;
    device const uchar *down_row = records + expert_meta[3u * expert]
        + args.down_offset + (uint64_t)row * row_bytes;
    const uint start = expert_meta[3u * expert + 1u];
    const uint count = expert_meta[3u * expert + 2u];
    const uint groups = args.hidden_width / 32u;
    for (uint t0 = 0u; t0 < count; t0 += 4u) {
        const uint tile = min(4u, count - t0);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint g = (uint)tiisg; g < groups; g += 32u) {
            for (uint i = 0u; i < tile; i++) {
                device const float *mid = mids
                    + (uint64_t)(start + t0 + i) * args.hidden_width;
                acc[i] += glm52_dot_kquant_group(args.down_type, down_row,
                                                 mid, g);
            }
        }
        for (uint i = 0u; i < tile; i++) {
            const float total = simd_sum(acc[i]);
            if (tiisg == 0u) {
                contribs[(uint64_t)(start + t0 + i) * args.input_width
                         + row] = total;
            }
        }
    }
}

// hidden[token] += contributi in ordine di applicazione ASCENDENTE partendo
// dal valore corrente: ((h + c1) + c2)…, la stessa associatività della
// sequenza add del percorso legacy. token_meta: triple u32
// {token_index, list_start, list_count}; token_apps: indici applicazione.
// Dispatch: tptg (256, 1, 1); threadgroups (ceil(embedding/256), voci, 1).
kernel void kernel_glm52_moe_prefill_reduce(
        constant ds4_metal_args_glm52_moe_prefill &args,
        device const uint  *token_meta,
        device const uint  *token_apps,
        device const float *contribs,
        device float       *hidden_all,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg   [[threads_per_threadgroup]]) {
    const uint r = tgpig.x * ntg.x + tpitg.x;
    const uint entry = tgpig.y;
    if (r >= args.input_width || entry >= args.token_entry_count) return;
    const uint token = token_meta[3u * entry];
    const uint start = token_meta[3u * entry + 1u];
    const uint count = token_meta[3u * entry + 2u];
    const uint64_t base = (uint64_t)token * args.input_width + r;
    float acc = hidden_all[base];
    for (uint j = 0u; j < count; j++) {
        acc += contribs[(uint64_t)token_apps[start + j] * args.input_width
                        + r];
    }
    hidden_all[base] = acc;
}
