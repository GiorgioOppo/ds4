"""Host oracle for actual V4.1 ROCm graph dispatch, TP admission and identity.

Recording stubs check typed arguments, failure propagation and ordered
imatrix capture. Projection math and BF16 low belong to the backend; this
test establishes neither HIP compilation nor numerical parity.
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
#define DS4_HAS_DEEPSEEK41_GPU 1
#define DS4_N_OUT_GROUP 8u
#define DS4_N_LORA_O 1024u
#define DS4_N_LAYER 40u
#define DS4_MODEL_FAMILY_DEEPSEEK41 41u
#define DS4_MODEL_VARIANT 7u
static unsigned model_family = DS4_MODEL_FAMILY_DEEPSEEK41;
#define DS4_MODEL_FAMILY model_family
enum { DS4_TENSOR_Q8_0 = 8, DS4_TENSOR_Q2_K = 10,
       DS4_TENSOR_Q4_K = 12, DS4_TENSOR_IQ2_XXS = 16 };
typedef struct { unsigned id; bool bf16; } ds4_gpu_tensor;
typedef struct { uint32_t type; uint64_t abs_offset; } ds4_tensor;
typedef struct { const void *map; uint64_t size; } ds4_model;
typedef struct {
    const ds4_tensor *attn_q_a, *attn_q_b, *attn_kv, *attn_output_a, *attn_output_b;
    const ds4_tensor *ffn_gate_exps, *ffn_up_exps, *ffn_down_exps;
} ds4_layer_weights;
typedef struct { ds4_layer_weights layer[DS4_N_LAYER]; } ds4_weights;
typedef struct { ds4_weights weights; } ds4_engine;
typedef struct { ds4_gpu_tensor *heads, *low, *block; } ds41_prefill_row;
typedef struct { ds4_gpu_tensor *attention_input[4]; } ds41_imatrix;
typedef struct {
    uint32_t tp_world, tp_rank;
    ds4_gpu_tensor *heads, *low, *block;
    ds41_prefill_row batch;
    ds41_imatrix *imatrix;
} ds41_gpu_graph;
static ds4_gpu_tensor heads = {1, false}, low = {2, false}, block = {3, false};
static ds4_gpu_tensor batch_heads = {4, false}, batch_low = {5, false}, batch_block = {6, false};
static ds4_gpu_tensor captured_low = {7, false};
static ds41_imatrix imatrix = {{NULL, NULL, NULL, &captured_low}};
static ds4_tensor a, b, gate = {DS4_TENSOR_IQ2_XXS, 0},
    up = {DS4_TENSOR_IQ2_XXS, 0}, down = {DS4_TENSOR_Q2_K, 0};
static ds4_model model = {&a, UINT64_C(200000000)};
static ds4_layer_weights layer;
static ds41_gpu_graph graph;
static char trace[4];
static unsigned events, expected_rows, dispatch_cases, admission_cases, identity_cases;
static bool batched, backend_failure, copy_failure;
static void record(char op) {
    assert(events + 1u < sizeof(trace));
    trace[events++] = op; trace[events] = '\0';
}
static int ds4_gpu_dsv41_attention_output_typed_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *lo, const void *map, uint64_t size,
        uint64_t offset_a, uint64_t offset_b, uint32_t type_a, uint32_t type_b,
        const ds4_gpu_tensor *in, uint32_t rows, uint32_t world, uint32_t rank) {
    assert(events == 0u && map == model.map && size == model.size);
    assert(offset_a == a.abs_offset && offset_b == b.abs_offset);
    assert(type_a == a.type && type_b == b.type);
    assert(out == (batched ? &batch_block : &block));
    assert(lo == (batched ? &batch_low : &low));
    assert(in == (batched ? &batch_heads : &heads));
    assert(rows == expected_rows && world == graph.tp_world && rank == graph.tp_rank);
    record('T');
    if (backend_failure || !rows || (world != 1u && world != 2u) || rank >= world ||
        (type_a != DS4_TENSOR_Q8_0 && type_a != DS4_TENSOR_Q4_K) ||
        (type_b != DS4_TENSOR_Q8_0 && type_b != DS4_TENSOR_Q4_K)) return 0;
    lo->bf16 = true;
    out->bf16 = false; /* The caller owns the rank sum and final rounding. */
    return 1;
}
static int ds4_gpu_tensor_copy(ds4_gpu_tensor *out, uint64_t out_offset,
        const ds4_gpu_tensor *in, uint64_t in_offset, uint64_t bytes) {
    assert(!batched && graph.imatrix == &imatrix && strcmp(trace, "T") == 0);
    assert(out == &captured_low && in == &low && low.bf16);
    assert(!out_offset && !in_offset);
    assert(bytes == (uint64_t)(8u / graph.tp_world) * 1024u * sizeof(float));
    record('C');
    if (copy_failure) return 0;
    out->bf16 = in->bf16;
    return 1;
}
'''

CASES = r'''
static void reset(uint32_t at, uint32_t bt, uint32_t world, uint32_t rank,
                  bool batch, uint32_t count, bool capture) {
    a = (ds4_tensor){at, UINT64_C(4096)};
    b = (ds4_tensor){bt, UINT64_C(80000000)};
    layer = (ds4_layer_weights){.attn_output_a = &a, .attn_output_b = &b,
        .ffn_gate_exps = &gate, .ffn_up_exps = &up, .ffn_down_exps = &down};
    graph = (ds41_gpu_graph){.tp_world = world, .tp_rank = rank,
        .heads = &heads, .low = &low, .block = &block,
        .batch = {&batch_heads, &batch_low, &batch_block},
        .imatrix = capture ? &imatrix : NULL};
    expected_rows = count; batched = batch; events = 0u; trace[0] = '\0';
    backend_failure = copy_failure = false;
    low.bf16 = batch_low.bf16 = captured_low.bf16 = false;
    block.bf16 = batch_block.bf16 = true;
}
static bool call(void) {
    ++dispatch_cases;
    const bool result = batched ?
        ds41_attention_output_batch(&graph, &model, &layer, expected_rows) :
        ds41_attention_output(&graph, &model, &layer);
    assert(graph.heads == &heads && graph.low == &low && graph.block == &block);
    return result;
}
static void check_dispatch(void) {
    const uint32_t counts[] = {1u, 2u, 65u}, types[] = {8u, 12u};
    for (unsigned batch = 0; batch < 2u; ++batch)
    for (unsigned n = 0; n < (batch ? 3u : 1u); ++n)
    for (uint32_t world = 1u; world <= 2u; ++world)
    for (uint32_t rank = 0u; rank < world; ++rank)
    for (unsigned ai = 0u; ai < 2u; ++ai)
    for (unsigned bi = 0u; bi < 2u; ++bi)
    for (unsigned capture = 0u; capture < (batch ? 1u : 2u); ++capture) {
        reset(types[ai], types[bi], world, rank, batch, counts[n], capture);
        assert(call() && strcmp(trace, capture ? "TC" : "T") == 0);
        assert(!(batch ? batch_block.bf16 : block.bf16));
        assert(!capture || captured_low.bf16);
        reset(types[ai], types[bi], world, rank, batch, counts[n], capture);
        backend_failure = true;
        assert(!call() && strcmp(trace, "T") == 0 && !captured_low.bf16);
        if (capture) {
            reset(types[ai], types[bi], world, rank, batch, counts[n], true);
            copy_failure = true;
            assert(!call() && strcmp(trace, "TC") == 0 && !captured_low.bf16);
        }
    }
    /* Typed admission rejects bad arguments before imatrix copying or any
     * fallback. World zero catches premature division in the graph wrapper. */
    const uint32_t invalid[][4] = {
        {8, 12, 0, 0}, {12, 8, 3, 0}, {12, 12, 1, 1}, {12, 12, 2, 2},
        {2, 8, 1, 0}, {8, 2, 2, 1}, {UINT32_MAX, 12, 1, 0},
    };
    for (unsigned batch = 0; batch < 2u; ++batch)
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        reset(invalid[i][0], invalid[i][1], invalid[i][2], invalid[i][3], batch, 1, !batch);
        assert(!call() && strcmp(trace, "T") == 0 && !captured_low.bf16);
    }
    reset(12, 12, 1, 0, true, 0, false);
    assert(!call() && strcmp(trace, "T") == 0);
}
static void check_admission(void) {
    const uint32_t types[] = {8u, 12u, 2u, 1u, UINT32_MAX};
    for (unsigned ai = 0; ai < 5u; ++ai)
    for (unsigned bi = 0; bi < 5u; ++bi) {
        reset(types[ai], types[bi], 2, 1, false, 1, false);
        assert(ds41_rocm_tp_layer_supported(&layer) == (ai < 2u && bi < 2u));
        ++admission_cases;
    }
    reset(12, 12, 2, 1, false, 1, false);
    ds4_tensor *experts[] = {&gate, &up, &down};
    for (unsigned i = 0; i < 3u; ++i) {
        const uint32_t saved = experts[i]->type;
        experts[i]->type = DS4_TENSOR_Q4_K;
        assert(!ds41_rocm_tp_layer_supported(&layer));
        experts[i]->type = saved; ++admission_cases;
    }
    assert(!ds41_rocm_tp_layer_supported(NULL));
    layer.attn_output_a = NULL;
    assert(!ds41_rocm_tp_layer_supported(&layer));
    admission_cases += 2u;
}
static void check_identity(void) {
    ds4_engine engine = {0};
    ds4_tensor projections[DS4_N_LAYER][5] = {0};
    for (unsigned il = 0; il < DS4_N_LAYER; ++il) {
        for (unsigned p = 0; p < 5u; ++p) projections[il][p].type = DS4_TENSOR_Q8_0;
        engine.weights.layer[il] = (ds4_layer_weights){
            .attn_q_a = &projections[il][0], .attn_q_b = &projections[il][1],
            .attn_kv = &projections[il][2], .attn_output_a = &projections[il][3],
            .attn_output_b = &projections[il][4]};
    }
    assert(ds41_attention_type_id(&engine.weights) == 0x413431u);
    assert(ds4_engine_model_id(&engine) == DS4_MODEL_VARIANT);
    for (unsigned il = 0; il < DS4_N_LAYER; ++il)
    for (unsigned p = 0; p < 5u; ++p) {
        projections[il][p].type = DS4_TENSOR_Q4_K;
        const int id = ds4_engine_model_id(&engine);
        assert(id > 0 && id != DS4_MODEL_VARIANT);
        assert((uint32_t)id == (ds41_attention_type_id(&engine.weights) & 0x7fffffffu));
        projections[il][p].type = DS4_TENSOR_Q8_0;
        assert(ds4_engine_model_id(&engine) == DS4_MODEL_VARIANT); ++identity_cases;
    }
    projections[0][0].type = DS4_TENSOR_Q4_K;
    model_family = 4u;
    assert(ds4_engine_model_id(&engine) == DS4_MODEL_VARIANT);
    model_family = DS4_MODEL_FAMILY_DEEPSEEK41;
    assert(ds4_engine_model_id(NULL) == DS4_MODEL_VARIANT);
}
int main(void) {
    check_dispatch(); check_admission(); check_identity();
    printf("PASS V4.1 ROCm graph: %u typed dispatch/failure/imatrix cases, "
           "%u TP admission cases, %u attention identity positions. HIP unverified.\n",
           dispatch_cases, admission_cases, identity_cases);
    return 0;
}
'''


def main():
    source = (ROOT / "ds4.c").read_text()
    bodies = "\n".join(extract_function(source, signature) for signature in (
        "static bool ds41_rocm_tp_layer_supported(",
        "static bool ds41_attention_output(",
        "static bool ds41_attention_output_batch(",
        "static uint32_t ds41_attention_type_id(",
        "int ds4_engine_model_id(",
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
