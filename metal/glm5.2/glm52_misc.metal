// GLM 5.2 — primitive generiche del grafo di decode (RMSNorm, matvec F32, add)
// Parte della libreria concatenata: vedi MetalRuntime.kernelFiles per l'ordine.

#include <metal_stdlib>
using namespace metal;

// MARK: - Resident decode graph primitives

struct ds4_metal_args_glm52_rms_norm {
    uint32_t width;
    float    epsilon;
    uint32_t pad0;
    uint32_t pad1;
};

// Generic-width RMSNorm with weight: one 256-thread threadgroup per row,
// float parallel reduction. Serves the resident decode graph's attn_norm and
// q_a_norm stages so activations never leave the GPU; the 512-wide KV-LoRA
// prefix keeps its dedicated cache-ready kernel.
kernel void kernel_glm52_rms_norm_f32(
        constant ds4_metal_args_glm52_rms_norm &args,
        device const float *input,
        device const float *weight,
        device float       *output,
        threadgroup float  *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 group [[threadgroup_position_in_grid]],
        uint3 threads [[threads_per_threadgroup]]) {
    const uint nth = threads.x;
    if (nth != 256u) return;
    const uint64_t row_offset = (uint64_t)group.x * args.width;
    device const float *row_input = input + row_offset;
    device float *row_output = output + row_offset;
    float sum_squares = 0.0f;
    for (uint i = tid; i < args.width; i += nth) {
        const float value = row_input[i];
        sum_squares += value * value;
    }
    scratch[tid] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = nth >> 1u; step > 0u; step >>= 1u) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // 1/sqrt, NOT rsqrt: the hardware reciprocal estimate differs by ~1 ULP
    // and the reference engine documents that drift compounding across the
    // layer stack (upstream metal/norm.metal). The CPU oracles on both
    // families divide by sqrt as well.
    const float inverse_rms =
        1.0f / sqrt(scratch[0] / (float)args.width + args.epsilon);
    for (uint i = tid; i < args.width; i += nth) {
        row_output[i] = (row_input[i] * inverse_rms) * weight[i];
    }
}

struct ds4_metal_args_glm52_matvec_f32 {
    uint32_t row_count;
    uint32_t input_width;
    uint32_t pad0;
    uint32_t pad1;
};

// Plain F32 matvec (one thread per output row) for the small F32 tensors the
// chained graph keeps on GPU — the 32-row indexer.proj. Validation-grade on
// purpose, like the quantized row kernels beside it.
kernel void kernel_glm52_matvec_f32(
        constant ds4_metal_args_glm52_matvec_f32 &args,
        device const float *rows,
        device const float *x,
        device float       *out,
        uint tid [[thread_position_in_grid]]) {
    const uint token_count = max(args.pad0, 1u);
    const uint total_rows = args.row_count * token_count;
    if (tid >= total_rows) return;
    const uint token = tid / args.row_count;
    const uint row_index = tid - token * args.row_count;
    device const float *row =
        rows + (uint64_t)row_index * args.input_width;
    device const float *token_x =
        x + (uint64_t)token * args.input_width;
    float acc = 0.0f;
    for (uint i = 0u; i < args.input_width; i++) {
        acc += row[i] * token_x[i];
    }
    out[(uint64_t)token * args.row_count + row_index] = acc;
}

struct ds4_metal_args_glm52_add {
    uint32_t count;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

// Elementwise residual add for the resident graph. Aliasing out with a is
// safe: each thread reads and writes only its own index.
kernel void kernel_glm52_add_f32(
        constant ds4_metal_args_glm52_add &args,
        device const float *a,
        device const float *b,
        device float       *out,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.count) return;
    out[tid] = a[tid] + b[tid];
}

// Interleaved compact-cache row store: cache-ready 576-wide F32 rows
// (normalized KV-LoRA prefix + RAW K-RoPE tail) converted to F16 at their
// absolute positions — the exact layout kernel_glm52_attention_indexed_f16
// reads. The two-plane store kernel remains for the upstream-shaped caches.

// MARK: - Matvec Q8/K-quant a COPPIA (qA + kvA condividono l'input normato)

// Le due proiezioni dell'attenzione che leggono lo stesso vettore normato
// in un solo dispatch: righe [0, rows_a) → out_a, [rows_a, rows_a+rows_b)
// → out_b. Un lancio in meno per layer; stessa matematica dei due matvec.
struct ds4_metal_args_glm52_matvec_pair {
    uint32_t type_a;
    uint32_t rows_a;
    uint32_t type_b;
    uint32_t rows_b;
    uint32_t input_width;
    uint32_t pad0, pad1, pad2;
};

kernel void kernel_glm52_matvec_pair_sg(
        constant ds4_metal_args_glm52_matvec_pair &args,
        device const float *x,
        device const uchar *weights_a,
        device const uchar *weights_b,
        device float       *out_a,
        device float       *out_b,
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  ntg   [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x * ntg.y + (uint)sgitg;
    if (row >= args.rows_a + args.rows_b) return;
    const bool second = row >= args.rows_a;
    const uint type = second ? args.type_b : args.type_a;
    const uint local_row = second ? row - args.rows_a : row;
    const uint row_bytes = glm52_kquant_row_bytes(type, args.input_width);
    if (row_bytes == 0u) return;
    device const uchar *weight_row = (second ? weights_b : weights_a)
        + (uint64_t)local_row * row_bytes;
    const uint groups = args.input_width / 32u;
    float acc = 0.0f;
    for (uint g = (uint)tiisg; g < groups; g += 32u) {
        acc += glm52_dot_kquant_group(type, weight_row, x, g);
    }
    const float total = simd_sum(acc);
    if (tiisg == 0u) {
        (second ? out_b : out_a)[local_row] = total;
    }
}

// MARK: - Argmax del head (decode greedy)

// Due stadi: riduzione parziale per threadgroup, poi riduzione finale in un
// threadgroup singolo. Pareggi: vince l'indice più basso — la stessa regola
// dell'argmax CPU, per un decode greedy deterministico e identico.
struct ds4_metal_args_glm52_argmax {
    uint32_t count;
    uint32_t chunk;      // elementi per threadgroup (stadio parziale)
    uint32_t pad0, pad1;
};

static inline void glm52_argmax_reduce(threadgroup float *values,
                                       threadgroup int32_t *indices,
                                       uint tid, uint width) {
    for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < stride) {
            const float a = values[tid];
            const float b = values[tid + stride];
            if (b > a || (b == a && indices[tid + stride] < indices[tid])) {
                values[tid] = b;
                indices[tid] = indices[tid + stride];
            }
        }
    }
}

kernel void kernel_glm52_argmax_partial_f32(
        constant ds4_metal_args_glm52_argmax &args,
        device const float   *values,
        device float         *partial_values,
        device int32_t       *partial_indices,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint tg  [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tptg [[threads_per_threadgroup]]) {
    threadgroup int32_t *indices = (threadgroup int32_t *)(scratch + tptg);
    const uint begin = tg * args.chunk;
    const uint end = min(begin + args.chunk, args.count);
    float best = -INFINITY;
    int32_t bestIndex = 0;
    for (uint i = begin + tid; i < end; i += tptg) {
        const float v = values[i];
        if (v > best) {
            best = v;
            bestIndex = (int32_t)i;
        }
    }
    scratch[tid] = best;
    indices[tid] = bestIndex;
    glm52_argmax_reduce(scratch, indices, tid, tptg);
    if (tid == 0u) {
        partial_values[tg] = scratch[0];
        partial_indices[tg] = indices[0];
    }
}

kernel void kernel_glm52_argmax_final_f32(
        constant ds4_metal_args_glm52_argmax &args,
        device const float   *partial_values,
        device const int32_t *partial_indices,
        device int32_t       *result,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]],
        uint tptg [[threads_per_threadgroup]]) {
    threadgroup int32_t *indices = (threadgroup int32_t *)(scratch + tptg);
    float best = -INFINITY;
    int32_t bestIndex = 0;
    for (uint i = tid; i < args.count; i += tptg) {
        const float v = partial_values[i];
        const int32_t index = partial_indices[i];
        if (v > best || (v == best && index < bestIndex)) {
            best = v;
            bestIndex = index;
        }
    }
    scratch[tid] = best;
    indices[tid] = bestIndex;
    glm52_argmax_reduce(scratch, indices, tid, tptg);
    if (tid == 0u) {
        result[0] = indices[0];
    }
}
