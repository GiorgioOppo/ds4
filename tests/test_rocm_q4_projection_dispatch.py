"""Execute the production V4/V4.1 projection dispatch with recording backends.

Includes the actual ROCm compatibility dispatcher, so Q4_K must reach its Q4
backend instead of merely a generic quantized API. This checks graph control
flow, shapes, TP row slices and BF16 ordering, not HIP compilation or GPU math.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]
SHIM = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#undef __APPLE__
#define DS4_ROCM_BUILD 1
#define DS4_TP_BATCH_MAX_ROWS 8u
#define DS4_N_VOCAB 129280u
enum { DS4_TENSOR_F32=0, DS4_TENSOR_F16=1, DS4_TENSOR_Q4_0=2,
       DS4_TENSOR_Q8_0=8, DS4_TENSOR_Q4_K=12, DS4_V41_BF16=0 };
#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "oracle failure at %d: %s\n", __LINE__, #x); \
    std::exit(42); } } while (0)
struct ds4_gpu_tensor { unsigned id; uint64_t offset, bytes; };
struct ds4_model { const void *map; uint64_t size; };
struct ds4_tensor { uint32_t type; uint64_t abs_offset, dim[2]; };
struct gguf_type_info { uint64_t block_elems, block_bytes; };
static const gguf_type_info *tensor_type(uint32_t t) {
    static const gguf_type_info f32={1,4}, f16={1,2}, q4={256,144}, q8={32,34};
    switch (t) {
    case DS4_TENSOR_F32: return &f32;
    case DS4_TENSOR_F16: return &f16;
    case DS4_TENSOR_Q4_K: return &q4;
    case DS4_TENSOR_Q8_0: return &q8;
    default: return nullptr;
    }
}
static const char *tensor_type_name(uint32_t) { return "unsupported"; }
struct event { char op; uint64_t weight, k, m, rows, xoff, yoff; };
static std::vector<event> events;
static unsigned mat_calls, round_calls, fail_mat, fail_round, cases;
static unsigned views, frees, live_views, fail_view;
static ds4_gpu_tensor input={101,0,UINT64_C(1)<<40};
static ds4_gpu_tensor output={202,0,UINT64_C(1)<<40};
static int model_anchor;
static ds4_model model={&model_anchor,UINT64_C(1)<<40};
static int record(char op, ds4_gpu_tensor *out, const void *map, uint64_t size,
        uint64_t off, uint64_t k, uint64_t m, const ds4_gpu_tensor *x, uint64_t rows) {
    CHECK(map==model.map && size==model.size && out && x);
    CHECK(out->id==output.id && x->id==input.id);
    events.push_back({op,off,k,m,rows,x->offset,out->offset});
    ++mat_calls;
    return mat_calls != fail_mat;
}
#define DENSE_STUB(name, op) \
static int name(ds4_gpu_tensor *out, const void *map, uint64_t size, uint64_t off, \
        uint64_t k, uint64_t m, const ds4_gpu_tensor *x, uint64_t rows) { \
    return record(op,out,map,size,off,k,m,x,rows); \
}
DENSE_STUB(ds4_gpu_matmul_f16_tensor,'h')
DENSE_STUB(ds4_gpu_matmul_f32_tensor,'f')
DENSE_STUB(ds4_gpu_matmul_q8_0_tensor,'q')
DENSE_STUB(ds4_rocm_matmul_q4_K_tensor,'4')
DENSE_STUB(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor,'e')
#define V41_STUB(name, op) \
static int name(ds4_gpu_tensor *out, const void *map, uint64_t size, uint64_t off, \
        uint32_t k, uint32_t m, uint32_t rows, const ds4_gpu_tensor *x) { \
    return record(op,out,map,size,off,k,m,x,rows); \
}
V41_STUB(ds4_gpu_dsv41_projection_rows,'H')
V41_STUB(ds4_gpu_dsv41_q8_projection_rows,'Q')
static int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *out, uint32_t width,
                                 uint32_t rows, unsigned format) {
    CHECK(out==&output && format==DS4_V41_BF16);
    events.push_back({'B',0,0,width,rows,0,0});
    ++round_calls;
    return round_calls != fail_round;
}
static ds4_gpu_tensor *ds4_gpu_tensor_view(ds4_gpu_tensor *p, uint64_t off, uint64_t n) {
    ++views;
    if (views==fail_view) return nullptr;
    CHECK(off<=p->bytes && n<=p->bytes-off);
    ++live_views;
    return new ds4_gpu_tensor{p->id,p->offset+off,n};
}
static void ds4_gpu_tensor_free(ds4_gpu_tensor *p) {
    if (!p) return;
    CHECK(live_views>0);
    --live_views; ++frees;
    delete p;
}
'''

CASES = r'''
static void reset(unsigned fail_projection=0, unsigned fail_bf16=0, unsigned fail_alloc=0) {
    CHECK(!live_views);
    events.clear(); mat_calls=round_calls=views=frees=0;
    fail_mat=fail_projection; fail_round=fail_bf16; fail_view=fail_alloc;
}
static ds4_tensor weight(unsigned type, uint64_t k, uint64_t m) {
    return ds4_tensor{type,UINT64_C(0x12345000),{k,m}};
}
static char expected_op(unsigned type, bool v41) {
    switch (type) {
    case DS4_TENSOR_Q4_K: return '4';
    case DS4_TENSOR_Q8_0: return v41 ? 'Q' : 'q';
    case DS4_TENSOR_F16: return v41 ? 'H' : 'h';
    case DS4_TENSOR_F32: return 'f';
    default: CHECK(false); return 0;
    }
}
static void check_projection(unsigned at, char op, uint64_t off, uint64_t k,
        uint64_t m, uint64_t rows, uint64_t xoff=0, uint64_t yoff=0) {
    CHECK(at<events.size());
    const event &e=events[at];
    CHECK(e.op==op && e.weight==off && e.k==k && e.m==m && e.rows==rows);
    CHECK(e.xoff==xoff && e.yoff==yoff);
}
static void check_bf16(unsigned at, uint64_t width, uint64_t rows) {
    CHECK(at<events.size());
    CHECK(events[at].op=='B' && events[at].m==width && events[at].rows==rows);
}
static bool call(unsigned helper, const ds4_tensor &w, unsigned rows, bool round) {
    ++cases;
    switch (helper) {
    case 0: return metal_graph_matmul_plain_tensor(&output,&model,&w,w.dim[0],w.dim[1],&input,rows);
    case 1: return metal_graph_matmul_q8_0_named_tensor("attn_q_a",7,1024,
        &output,&model,&w,w.dim[0],w.dim[1],&input,rows);
    case 2: CHECK(rows==1); return ds41_matmul(&output,&model,&w,&input,round);
    default: return ds41_matmul_batch(&output,&model,&w,&input,rows,round);
    }
}
static void check_shapes(void) {
    // V4 and V4.1 QA, QB and KV, respectively. No weight/input arrays are
    // allocated: the exact production dimensions must reach the API recorder.
    const uint64_t shape[][2]={{4096,1024},{1024,32768},{4096,512},
                               {5120,1280},{1280,32768},{5120,512}};
    const unsigned types[]={DS4_TENSOR_Q4_K,DS4_TENSOR_Q8_0,DS4_TENSOR_F16};
    const unsigned counts[]={1,2,8,9,32,128,2048,8192};
    for (const auto &s:shape) for (unsigned type:types)
    for (unsigned helper=0;helper<4;++helper) for (unsigned rows:counts) {
        if ((helper==1 && type==DS4_TENSOR_F16) || (helper==2 && rows!=1)) continue;
        const ds4_tensor w=weight(type,s[0],s[1]);
        for (unsigned round=0;round<(helper>=2 ? 2u : 1u);++round) {
            reset(); CHECK(call(helper,w,rows,round));
            CHECK(events.size()==1u+round && mat_calls==1 && round_calls==round);
            check_projection(0,expected_op(type,helper>=2),w.abs_offset,s[0],s[1],rows);
            if (round) check_bf16(1,s[1],rows);
            reset(1); CHECK(!call(helper,w,rows,round));
            CHECK(events.size()==1 && !round_calls);
            check_projection(0,expected_op(type,helper>=2),w.abs_offset,s[0],s[1],rows);
            if (round) {
                reset(0,1); CHECK(!call(helper,w,rows,true));
                CHECK(events.size()==2 && mat_calls==1 && round_calls==1);
                check_bf16(1,s[1],rows);
            }
        }
    }
    // The historical named helper is dense-quant only; F16 has a distinct
    // plain path. Unsupported Q4_0 must not silently run Q4_K or Q8 kernels.
    const unsigned rejected[]={DS4_TENSOR_F16,DS4_TENSOR_F32,DS4_TENSOR_Q4_0,999u};
    for (unsigned type:rejected) {
        const ds4_tensor w=weight(type,4096,1024);
        reset(); CHECK(!call(1,w,33,false)); CHECK(events.empty());
    }
    reset();
    CHECK(!metal_graph_matmul_dense_quant_tensor(&output,&model,nullptr,4096,1024,&input,9));
    CHECK(events.empty());
}
static void check_slices(void) {
    const unsigned types[]={DS4_TENSOR_Q4_K,DS4_TENSOR_Q8_0,DS4_TENSOR_F16};
    for (unsigned type:types) for (unsigned k:{1024u,1280u})
    for (unsigned rows:{1u,3u,128u})
    for (unsigned first:{0u,16384u,32512u}) {
        const ds4_tensor w=weight(type,k,32768);
        const unsigned count=first==32512u ? 256u : 16384u;
        const uint64_t row_bytes=type==DS4_TENSOR_Q4_K ? (uint64_t)k/256u*144u :
            (type==DS4_TENSOR_Q8_0 ? (uint64_t)k/32u*34u : (uint64_t)k*2u);
        for (unsigned failure=0;failure<3;++failure) {
            reset(failure==1, failure==2);
            const bool ok=rows==1
                ? ds41_matmul_rows(&output,&model,&w,&input,first,count)
                : ds41_matmul_rows_batch(&output,&model,&w,&input,first,count,rows);
            ++cases; CHECK(ok==(failure==0));
            CHECK(events.size()==(failure==1 ? 1u : 2u));
            check_projection(0,expected_op(type,true),w.abs_offset+first*row_bytes,
                             k,count,rows);
            if (failure!=1) check_bf16(1,count,rows);
            CHECK(w.abs_offset==UINT64_C(0x12345000) && w.dim[1]==32768);
        }
    }
    ds4_tensor w=weight(DS4_TENSOR_Q4_K,1024,32768);
    for (unsigned first:{32769u,UINT32_MAX}) {
        reset(); CHECK(!ds41_matmul_rows(&output,&model,&w,&input,first,1));
        CHECK(events.empty()); ++cases;
    }
    reset(); CHECK(!ds41_matmul_rows_batch(&output,&model,&w,&input,32768,1,32));
    CHECK(events.empty()); ++cases;
    w.type=999;
    reset(); CHECK(!ds41_matmul_rows(&output,&model,&w,&input,0,32));
    CHECK(events.empty()); ++cases;
}
static void check_f32_rows(void) {
    const ds4_tensor w=weight(DS4_TENSOR_F32,5120,1024);
    for (unsigned rows:{1u,2u,8u,9u}) {
        reset(); CHECK(call(3,w,rows,true));
        const unsigned projections=rows>=2 && rows<=8 ? rows : 1;
        CHECK(events.size()==projections+1 && mat_calls==projections);
        for (unsigned i=0;i<projections;++i)
            check_projection(i,'f',w.abs_offset,5120,1024,projections>1 ? 1 : rows,
                (uint64_t)i*5120*4,(uint64_t)i*1024*4);
        check_bf16(projections,1024,rows);
        CHECK(!live_views && views==frees);
    }
    reset(2); CHECK(!call(3,w,3,true));
    CHECK(mat_calls==2 && !round_calls && !live_views && views==frees);
    for (unsigned bad_view:{1u,2u,3u,4u}) {
        reset(0,0,bad_view); CHECK(!call(3,w,3,true));
        CHECK(!round_calls && !live_views && frees+1==views);
    }
}
int main(void) {
    check_shapes(); check_slices(); check_f32_rows();
    std::printf("PASS: %u production ROCm V4/V4.1 projection dispatch cases; Q4/Q8/F16, BF16 and TP slices\n",cases);
}
'''


def production_parts():
    src = (ROOT / "ds4.c").read_text()
    compat = (ROOT / "ds4_rocm_compat.cu").read_text()
    parts = {
        "compat": extract_function(compat, 'extern "C" int ds4_gpu_matmul_quant_tensor('),
        "types": extract_function(src, "static bool tensor_type_is_dense_quant("),
        "nbytes": extract_function(src, "static bool tensor_nbytes("),
    }
    names = ["metal_graph_matmul_plain_tensor", "metal_graph_matmul_dense_quant_abs",
             "metal_graph_matmul_dense_quant_tensor", "metal_graph_matmul_q8_0_named_tensor",
             "ds41_bf16", "ds41_matmul", "ds41_matmul_batch", "ds41_matmul_rows_batch",
             "ds41_matmul_rows"]
    for name in names:
        signature = f"static bool {name}("
        # plain_tensor has an earlier forward declaration; keep the last
        # definition, then use the shared balanced-function extractor.
        parts[name] = extract_function(src[src.rindex(signature):], signature)
    return parts


def main():
    parts = production_parts()
    mutations = {
        "compat_q8_only": ("compat", "if (weight_type == 12u)", "if (false)"),
        "dense_q8_only": ("types", "type == DS4_TENSOR_Q4_K", "false"),
        "v41_q4_as_q8": ("ds41_matmul", "weight->type == DS4_TENSOR_Q8_0) {",
                           "weight->type == DS4_TENSOR_Q8_0 || weight->type == DS4_TENSOR_Q4_K) {"),
        "slice_wrong_stride": ("ds41_matmul_rows_batch", "(uint64_t)first * row_bytes", "(uint64_t)first * 34u"),
        "drop_bf16": ("ds41_bf16", "return ds4_gpu_dsv41_quantize(x, width, 1, DS4_V41_BF16) != 0;", "return true;"),
    }
    with tempfile.TemporaryDirectory(prefix="ds4-rocm-q4-projection-") as work:
        work = Path(work)
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        for name in ["production", *mutations]:
            variant = parts.copy()
            if name != "production":
                key, old, new = mutations[name]
                if variant[key].count(old) != 1:
                    raise AssertionError(f"negative control no longer matches: {name}")
                variant[key] = variant[key].replace(old, new)
            source, binary = work / f"{name}.cpp", work / name
            source.write_text(SHIM + "\n".join(variant.values()) + CASES)
            subprocess.run(compiler + ["-std=c++17", "-O1", "-g", "-Wall", "-Wextra",
                "-Wno-unused-function", "-Wno-unused-parameter", "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all", "-fno-omit-frame-pointer", str(source), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            if name == "production":
                if result.returncode:
                    raise AssertionError(result.stdout + result.stderr)
                print(result.stdout, end="")
            elif result.returncode != 42 or "oracle failure" not in result.stderr:
                raise AssertionError(f"negative control {name} was not rejected by oracle: "
                                     f"rc={result.returncode}\n{result.stdout}{result.stderr}")
        print(f"PASS: {len(mutations)} negative controls reject Q8-only dispatch, wrong slices and missing BF16")


if __name__ == "__main__":
    main()
