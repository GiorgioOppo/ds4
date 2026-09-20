"""Run the production V4 decode output dispatch for mixed Q8/Q4 matrices.

The unfused single-device branch is shared by Metal, CUDA and ROCm.  Extract
its actual decision and calls; replace the GPU operations with typed spies.
This verifies dispatch/error propagation, not GPU arithmetic or the older
all-Q4 prefill failure reported against a different checkout.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ds4.c").read_text()
BODY_ANCHOR = """        ds4_gpu_tensor *attn_out_dst = g->tp_world == 2 ?
                g->tp_out[il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_ATTN] : metal_graph_attn_out(g);"""
assert SOURCE.count(BODY_ANCHOR) == 2, "expected the two decode output arms"
body_start = SOURCE.index(BODY_ANCHOR)
branch_start = SOURCE.rfind("    } else if (", 0, body_start)
branch_end = SOURCE.index("    if (ok && g->tp_world == 2) {", body_start)
BRANCH = SOURCE[branch_start:branch_end]
assert BRANCH.startswith("    } else if (ok &&"), "unexpected output dispatch"

STUBS = r'''
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
enum { DS4_TENSOR_Q8_0=8, DS4_TENSOR_Q4_K=12, DS4_N_EMBD=4096,
       DS4_TP_GATES_PER_LAYER=2, DS4_TP_GATE_ATTN=0 };
typedef struct { int tag; } ds4_gpu_tensor;
typedef struct { uint32_t type; uint64_t abs_offset; } ds4_tensor;
typedef struct { const void *map; uint64_t size; } ds4_model;
typedef struct { ds4_tensor *attn_output_a,*attn_output_b; } ds4_layer_weights;
typedef struct { uint32_t tp_world; ds4_gpu_tensor **tp_out; } ds4_gpu_graph;
static ds4_gpu_tensor dst={1},low={2},heads={3},group_tmp={4},low_tmp={5};
static ds4_gpu_graph graph={1,0};
static ds4_tensor a={0,64},b={0,4096};
static ds4_model model={&a,8192};
static ds4_layer_weights layer={&a,&b};
static unsigned checks,cases,low_calls,expand_calls,q8_calls;
enum { FAIL_NONE,FAIL_PRIOR,FAIL_LOW,FAIL_EXPAND,FAIL_Q8 };
static int failure;
static void check(bool ok,const char *why) {
    ++checks;
    if (!ok) { fprintf(stderr,"FAIL V4 output types A=%u B=%u mode=%d: %s\n",
                       a.type,b.type,failure,why); exit(1); }
}
static ds4_gpu_tensor *metal_graph_attn_out(ds4_gpu_graph *g) { check(g==&graph,"output graph"); return &dst; }
static ds4_gpu_tensor *metal_graph_attn_low(ds4_gpu_graph *g) { check(g==&graph,"low graph"); return &low; }
static ds4_gpu_tensor *metal_graph_heads(ds4_gpu_graph *g) { check(g==&graph,"heads graph"); return &heads; }
static ds4_gpu_tensor *metal_graph_batch_group_tmp(ds4_gpu_graph *g) { check(g==&graph,"group scratch graph"); return &group_tmp; }
static ds4_gpu_tensor *metal_graph_batch_low_tmp(ds4_gpu_graph *g) { check(g==&graph,"low scratch graph"); return &low_tmp; }
static bool metal_graph_attention_output_dense_quant_low(
        ds4_gpu_tensor *out,ds4_gpu_graph *g,const ds4_model *m,
        const ds4_tensor *w,uint64_t group_dim,uint64_t rank,
        uint32_t group0,uint32_t group_cnt,const ds4_gpu_tensor *x,bool decode) {
    ++low_calls;
    check(out==&low && g==&graph && m==&model && w==&a && x==&heads,
          "typed A pointers preserve the original matrix and buffers");
    check(group_dim==4096 && rank==1024 && group0==0 && group_cnt==8 && decode,
          "typed A dimensions and decode mode");
    return failure!=FAIL_LOW;
}
static bool metal_graph_matmul_dense_quant_tensor(
        ds4_gpu_tensor *out,const ds4_model *m,const ds4_tensor *w,
        uint64_t in_dim,uint64_t out_dim,const ds4_gpu_tensor *x,uint64_t n) {
    ++expand_calls;
    check(out==&dst && m==&model && w==&b && x==&low,
          "typed B pointers preserve the original matrix and buffers");
    check(in_dim==8192 && out_dim==4096 && n==1,"typed B dimensions");
    return failure!=FAIL_EXPAND;
}
static int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor *out,ds4_gpu_tensor *lo,ds4_gpu_tensor *gt,ds4_gpu_tensor *lt,
        const void *map,uint64_t size,uint64_t ao,uint64_t bo,
        uint64_t group_dim,uint64_t rank,uint32_t groups,uint64_t out_dim,
        const ds4_gpu_tensor *x,uint32_t n) {
    ++q8_calls;
    check(a.type==DS4_TENSOR_Q8_0 && b.type==DS4_TENSOR_Q8_0,
          "Q8-only kernel must not reinterpret a Q4 matrix");
    check(out==&dst && lo==&low && gt==&group_tmp && lt==&low_tmp && x==&heads,
          "Q8 output and scratch buffers");
    check(map==model.map && size==model.size && ao==a.abs_offset && bo==b.abs_offset,
          "Q8 model ranges");
    check(group_dim==4096 && rank==1024 && groups==8 && out_dim==4096 && n==1,
          "Q8 dimensions");
    return failure!=FAIL_Q8;
}
static bool dispatch(bool ok) {
    ds4_gpu_graph *g=&graph;
    const ds4_model *model_p=&model;
    const ds4_layer_weights *layer_p=&layer;
    const uint32_t il=3,n_groups=8;
    const uint64_t group_dim=4096,rank=1024;
    /* Use the production parameter names without hiding fixture globals in
     * the GPU spies above. Earlier fused/TP branches are deliberately false. */
    const ds4_model *model=model_p;
    const ds4_layer_weights *layer=layer_p;
    if (false) {
'''
TEST = r'''
    return ok;
}
int main(void) {
    const uint32_t types[]={DS4_TENSOR_Q8_0,DS4_TENSOR_Q4_K};
    for (unsigned ia=0;ia<2;++ia) for (unsigned ib=0;ib<2;++ib) {
        a.type=types[ia]; b.type=types[ib];
        const bool typed=ia!=0 || ib!=0;
        for (failure=FAIL_NONE;failure<=FAIL_Q8;++failure) {
            low_calls=expand_calls=q8_calls=0;
            const bool was_ok=failure!=FAIL_PRIOR;
            const bool expected=was_ok && (typed ?
                failure!=FAIL_LOW && failure!=FAIL_EXPAND : failure!=FAIL_Q8);
            check(dispatch(was_ok)==expected,"failure propagation");
            check(low_calls==(unsigned)(was_ok && typed),"typed A call count");
            check(expand_calls==(unsigned)(was_ok && typed && failure!=FAIL_LOW),
                  "B must not run after A fails");
            check(q8_calls==(unsigned)(was_ok && !typed),"Q8 fast-path call count");
            ++cases;
        }
    }
    printf("PASS V4 decode output type dispatch: %u cases, %u checks\n",cases,checks);
    return 0;
}
'''


def main():
    compiler = shlex.split(os.environ.get("CC", "cc"))
    with tempfile.TemporaryDirectory(prefix="ds4-output-types-") as tmp:
        directory = Path(tmp)
        for name, branch in (("production", BRANCH), ("old-guard", old_guard())):
            source = directory / f"{name}.c"
            binary = directory / name
            source.write_text(STUBS + branch + TEST)
            subprocess.run(compiler + ["-std=c11", "-O1", "-g", "-Wall", "-Wextra",
                                       "-Werror", "-fsanitize=address,undefined",
                                       "-fno-omit-frame-pointer", str(source),
                                       "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            if name == "production":
                assert result.returncode == 0, result.stdout + result.stderr
                print(result.stdout, end="")
            else:
                assert result.returncode != 0 and "Q8-only kernel must not reinterpret" in result.stderr, result.stderr
                print("PASS negative control: previous A-only guard rejects Q8_A/Q4_B")


def old_guard():
    # Mutate only the condition, preserving both production branch bodies.
    branch, count = re.subn(r"else if \(ok && .*?\) \{",
                           "else if (ok && layer->attn_output_a->type != DS4_TENSOR_Q8_0) {",
                           BRANCH, count=1, flags=re.S)
    assert count == 1, "missing output type decision"
    return branch


if __name__ == "__main__":
    main()
