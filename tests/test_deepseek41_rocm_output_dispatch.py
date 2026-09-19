"""Exercise the actual V4.1 ROCm graph output dispatch without a GPU.

Extract the scalar/batch wrappers and their low projection helper. Recording
backend stubs check weight types, row views, BF16 ordering and failure stops;
this establishes graph control flow, not HIP compilation or numerical parity.
"""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from kernel_source import extract_function


ROOT = Path(__file__).resolve().parents[1]

SHIM = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#undef __APPLE__
#define DS4_ROCM_BUILD 1
#define DS4_N_OUT_GROUP 8u
#define DS4_N_LORA_O 1024u
enum { DS4_TENSOR_Q8_0 = 8, DS4_TENSOR_Q4_K = 12, MAX_ROWS = 65 };
typedef struct { unsigned id; } ds4_gpu_tensor;
typedef struct { uint32_t type; uint64_t abs_offset; } ds4_tensor;
typedef struct { const void *map; uint64_t size; } ds4_model;
typedef struct { const ds4_tensor *attn_output_a, *attn_output_b; } ds4_layer_weights;
typedef struct { ds4_gpu_tensor *heads, *low, *block; } ds41_prefill_row;
typedef struct { ds4_gpu_tensor *attention_input[4]; } ds41_imatrix;
typedef struct {
    uint32_t tp_world, tp_rank;
    ds4_gpu_tensor *heads, *low, *block;
    ds41_prefill_row batch, rows_view[MAX_ROWS];
    ds41_imatrix *imatrix;
} ds41_gpu_graph;

static ds4_gpu_tensor heads = {1}, low = {2}, block = {3};
static ds4_gpu_tensor batch_heads = {4}, batch_low = {5}, batch_block = {6};
static ds4_gpu_tensor row_heads[MAX_ROWS], row_low[MAX_ROWS];
static ds4_tensor a, b;
static ds4_model model = {&a, UINT64_C(200000000)};
static ds4_layer_weights layer = {&a, &b};
static ds41_gpu_graph graph;
static char trace[2 * MAX_ROWS + 2];
static unsigned events, expected_rows, projected, rounded, cases;
static bool batched;
static char failure;
static unsigned failure_row;

static void record(char op) {
    assert(events + 1u < sizeof(trace));
    trace[events++] = op;
    trace[events] = '\0';
}
static const ds4_gpu_tensor *expected_heads(unsigned row) {
    return batched ? &row_heads[row] : &heads;
}
static ds4_gpu_tensor *expected_low(unsigned row) {
    return batched ? &row_low[row] : &low;
}
static bool project_a(ds4_gpu_tensor *out, const ds4_gpu_tensor *in) {
    assert(projected < expected_rows && projected == rounded);
    assert(out == expected_low(projected) && in == expected_heads(projected));
    record('A');
    const unsigned row = projected++;
    return !(failure == 'A' && row == failure_row);
}
static bool tensor_nbytes(uint32_t type, uint64_t width, uint64_t *bytes) {
    assert(type == DS4_TENSOR_Q8_0 && width == 4096u);
    *bytes = width / 32u * 34u;
    return true;
}
static bool metal_graph_attention_output_dense_quant_low(
        ds4_gpu_tensor *out, ds4_gpu_tensor *scratch, const ds4_model *m,
        const ds4_tensor *weight, uint64_t width, uint64_t rank,
        uint32_t group0, uint32_t groups, const ds4_gpu_tensor *in, bool final) {
    assert(!scratch && m == &model && weight == &a && a.type == DS4_TENSOR_Q4_K);
    assert(width == 4096u && rank == 1024u && group0 == 0u && groups == 8u && !final);
    return project_a(out, in);
}
static int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor *out, const void *map, uint64_t size, uint64_t offset,
        uint64_t width, uint64_t rank, uint32_t groups, const ds4_gpu_tensor *in) {
    assert(a.type == DS4_TENSOR_Q8_0 && map == model.map && size == model.size);
    assert(offset == a.abs_offset && width == 4096u && rank == 1024u && groups == 8u);
    return project_a(out, in);
}
static bool ds41_bf16(ds4_gpu_tensor *out, uint32_t width) {
    assert(rounded < expected_rows && projected == rounded + 1u);
    assert(out == expected_low(rounded) && width == 8192u);
    assert(events && trace[events - 1u] == 'A');
    record('R');
    const unsigned row = rounded++;
    return !(failure == 'R' && row == failure_row);
}
static int ds4_gpu_tensor_copy(ds4_gpu_tensor *out, uint64_t out_offset,
        const ds4_gpu_tensor *in, uint64_t in_offset, uint64_t bytes) {
    (void)out; (void)out_offset; (void)in; (void)in_offset; (void)bytes;
    assert(!"unexpected imatrix copy");
    return 0;
}
static bool project_b(ds4_gpu_tensor *out, const ds4_model *m,
        const ds4_tensor *weight, const ds4_gpu_tensor *in, uint32_t rows, bool round) {
    assert(m == &model && weight == &b && rows == expected_rows && !round);
    assert(projected == rows && rounded == rows);
    assert(out == (batched ? &batch_block : &block));
    assert(in == (batched ? &batch_low : &low));
    record('B');
    return failure != 'B';
}
static bool ds41_matmul(ds4_gpu_tensor *out, const ds4_model *m,
        const ds4_tensor *weight, const ds4_gpu_tensor *in, bool round) {
    assert(!batched);
    return project_b(out, m, weight, in, 1u, round);
}
static bool ds41_matmul_batch(ds4_gpu_tensor *out, const ds4_model *m,
        const ds4_tensor *weight, const ds4_gpu_tensor *in, uint32_t rows, bool round) {
    assert(batched);
    return project_b(out, m, weight, in, rows, round);
}
static int q8_output(ds4_gpu_tensor *out, ds4_gpu_tensor *lo,
        const void *map, uint64_t size, uint64_t offset_a, uint64_t offset_b,
        const ds4_gpu_tensor *in, uint32_t rows, uint32_t world, uint32_t rank) {
    assert(a.type == DS4_TENSOR_Q8_0 && b.type == DS4_TENSOR_Q8_0);
    assert(map == model.map && size == model.size);
    assert(offset_a == a.abs_offset && offset_b == b.abs_offset);
    assert(out == (batched ? &batch_block : &block));
    assert(lo == (batched ? &batch_low : &low));
    assert(in == (batched ? &batch_heads : &heads));
    assert(rows == expected_rows && world == graph.tp_world && rank == graph.tp_rank);
    assert(events == 0u);
    record(world == 1u ? 'Q' : 'T');
    return failure != 'Q';
}
static int ds4_gpu_dsv41_attention_output_batch(ds4_gpu_tensor *out,
        ds4_gpu_tensor *lo, const void *map, uint64_t size,
        uint64_t offset_a, uint64_t offset_b, const ds4_gpu_tensor *in, uint32_t rows) {
    return q8_output(out, lo, map, size, offset_a, offset_b, in, rows, 1u, 0u);
}
static int ds4_gpu_dsv41_attention_output_tp_batch(ds4_gpu_tensor *out,
        ds4_gpu_tensor *lo, const void *map, uint64_t size,
        uint64_t offset_a, uint64_t offset_b, const ds4_gpu_tensor *in,
        uint32_t rows, uint32_t rank) {
    return q8_output(out, lo, map, size, offset_a, offset_b, in, rows, 2u, rank);
}
'''

CASES = r'''
static void reset(uint32_t at, uint32_t bt, uint32_t world, uint32_t rank,
                  bool batch, uint32_t count) {
    a = (ds4_tensor){at, UINT64_C(4096)};
    b = (ds4_tensor){bt, UINT64_C(80000000)};
    graph = (ds41_gpu_graph){.tp_world = world, .tp_rank = rank,
        .heads = &heads, .low = &low, .block = &block,
        .batch = {&batch_heads, &batch_low, &batch_block}};
    for (unsigned row = 0; row < MAX_ROWS; ++row) {
        row_heads[row].id = 100u + row;
        row_low[row].id = 200u + row;
        graph.rows_view[row] = (ds41_prefill_row){&row_heads[row], &row_low[row], NULL};
    }
    expected_rows = count;
    batched = batch;
    events = projected = rounded = 0u;
    trace[0] = '\0';
    failure = 0;
    failure_row = 0u;
}
static bool call(void) {
    ++cases;
    const bool result = batched ?
        ds41_attention_output_batch(&graph, &model, &layer, expected_rows) :
        ds41_attention_output(&graph, &model, &layer);
    /* The temporary row graph must not change the caller's scalar handles. */
    assert(graph.heads == &heads && graph.low == &low && graph.block == &block);
    return result;
}
static void check_low_prefix(unsigned rows) {
    for (unsigned row = 0; row < rows; ++row) {
        assert(trace[2u * row] == 'A');
        assert(trace[2u * row + 1u] == 'R');
    }
}
int main(void) {
    const uint32_t counts[] = {1u, 2u, MAX_ROWS};
    const uint32_t types[] = {DS4_TENSOR_Q8_0, DS4_TENSOR_Q4_K};
    for (unsigned batch = 0; batch < 2u; ++batch)
    for (unsigned n = 0; n < (batch ? 3u : 1u); ++n)
    for (uint32_t world = 1u; world <= 2u; ++world)
    for (uint32_t rank = 0u; rank < world; ++rank)
    for (unsigned ai = 0u; ai < 2u; ++ai)
    for (unsigned bi = 0u; bi < 2u; ++bi) {
        reset(types[ai], types[bi], world, rank, batch, counts[n]);
        const bool q8 = ai == 0u && bi == 0u;
        if (q8) {
            assert(call());
            assert(strcmp(trace, world == 1u ? "Q" : "T") == 0);
            reset(types[ai], types[bi], world, rank, batch, counts[n]);
            failure = 'Q';
            assert(!call());
            assert(strcmp(trace, world == 1u ? "Q" : "T") == 0);
        } else if (world == 2u) {
            assert(!call());
            assert(events == 0u); /* Q4 is never interpreted by a Q8-only kernel. */
        } else {
            assert(call());
            check_low_prefix(counts[n]);
            assert(events == counts[n] * 2u + 1u && trace[events - 1u] == 'B');
            const char stages[] = {'A', 'R', 'B'};
            for (unsigned s = 0u; s < 3u; ++s) {
                reset(types[ai], types[bi], world, rank, batch, counts[n]);
                failure = stages[s];
                failure_row = counts[n] / 2u;
                assert(!call());
                if (failure == 'B') {
                    check_low_prefix(counts[n]);
                    assert(events == counts[n] * 2u + 1u && trace[events - 1u] == 'B');
                } else {
                    check_low_prefix(failure_row);
                    assert(events == 2u * failure_row + (failure == 'A' ? 1u : 2u));
                    assert(trace[events - 1u] == failure);
                }
            }
        }
    }
    printf("PASS V4.1 ROCm output dispatch: %u scalar/batch/type/TP/failure cases; "
           "BF16 low precedes B, final output remains unrounded. HIP unverified.\n", cases);
    return 0;
}
'''


def main():
    source = (ROOT / "ds4.c").read_text()
    bodies = "\n".join(extract_function(source, signature) for signature in (
        "static bool ds41_attention_low(",
        "static bool ds41_attention_output(",
        "static bool ds41_attention_output_batch(",
    ))
    with tempfile.TemporaryDirectory(prefix="ds4-v41-rocm-output-") as directory:
        path = Path(directory)
        unit = path / "dispatch.c"
        executable = path / "dispatch"
        unit.write_text(SHIM + bodies + CASES)
        subprocess.run(shlex.split(os.environ.get("CC", "clang")) + [
            "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
            str(unit), "-o", str(executable),
        ], check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
