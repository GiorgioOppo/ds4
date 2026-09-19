"""Check mixed-output Q8 launches against the original V4.1 Q8/Q8 path.

Execute extracted host code with HIP launch recorders. Compare every kernel,
grid, tile, TP offset and physical stride; this does not execute GPU arithmetic.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]
SHIM = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>
struct dim3 { unsigned x,y,z; dim3(unsigned x=1,unsigned y=1,unsigned z=1):x(x),y(y),z(z){} };
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; };
static bool g_quality_mode, gfx;
static constexpr int DS4_V41_BF16=0;
static std::vector<uint64_t> trace;
static void rec(std::initializer_list<uint64_t> v) { trace.insert(trace.end(),v); }
static uint64_t addr(const void *p) { return (uintptr_t)p; }
static bool ds4_rocm_is_gfx1151() { return gfx; }
static bool cuda_model_range_fits(uint64_t sz,uint64_t off,uint64_t n) { return off<=sz && n<=sz-off; }
static bool cuda_tensor_has_elems2(const ds4_gpu_tensor *p,uint64_t a,uint64_t b,uint64_t s) {
    return p && p->ptr && a*b*s<=p->bytes;
}
static const char *cuda_model_range_ptr(const void *p,uint64_t off,uint64_t,const char*) { return (const char*)p+off; }
static int cudaGetLastError() { return 0; }
static int cuda_ok(int e,const char*) { return !e; }
static int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *p,uint32_t w,uint32_t n,int f) {
    rec({9,addr(p->ptr),w,n,(unsigned)f}); return 1;
}
template<class F> static auto launch(F fn,dim3 g,unsigned block) {
    return [=](auto... args) { rec({0,g.x,g.y,g.z,block}); fn(args...); };
}
static void v41_grouped_q8_f32_blocks4_kernel(float *o,const unsigned char *w,const float *x,
        uint32_t k,uint32_t m,uint32_t g) { rec({1,addr(o),addr(w),addr(x),k,m,g}); }
template<uint32_t M,uint32_t W,uint32_t G=8>
static void v41_grouped_q8_f32_wmma_rowtile_kernel(float *o,const unsigned char *w,const float *x,
        uint32_t n,uint32_t k,uint32_t m,uint64_t s) { rec({2,M,W,G,addr(o),addr(w),addr(x),n,k,m,s}); }
static void grouped_q8_0_a_f32_batch_warp8_kernel(float *o,const unsigned char *w,const float *x,
        uint32_t k,uint32_t m,uint32_t g,uint32_t n,uint32_t b) { rec({3,addr(o),addr(w),addr(x),k,m,g,n,b}); }
static void v41_q8_f32_blocks4_kernel(float *o,const unsigned char *w,const float *x,
        uint32_t k,uint32_t m,uint64_t s) { rec({4,addr(o),addr(w),addr(x),k,m,s}); }
template<uint32_t M,uint32_t W>
static void matmul_q8_0_f32_batch_wmma_rowtile_kernel(float *o,const unsigned char *w,const float *x,
        uint32_t n,uint32_t k,uint32_t m,uint64_t s) { rec({5,M,W,addr(o),addr(w),addr(x),n,k,m,s}); }
static void matmul_q8_0_f32_batch_warp8_kernel(float *o,const unsigned char *w,const float *x,
        uint32_t k,uint32_t m,uint32_t n,uint32_t b) { rec({6,addr(o),addr(w),addr(x),k,m,n,b}); }
static void cuda_launch_grouped_q8_a_sharedx(float *o,const unsigned char *w,const float *x,
        uint32_t n,uint32_t g,uint32_t b,uint32_t r,uint64_t s,uint32_t rb,uint32_t ti,uint32_t bt) {
    rec({7,addr(o),addr(w),addr(x),n,g,b,r,s,rb,ti,bt});
}
static void cuda_launch_q8_batch_sharedx(float *o,const unsigned char *w,const float *x,
        uint32_t b,uint32_t m,uint32_t n,uint64_t s,uint32_t rb,uint32_t ti,uint32_t bt) {
    rec({8,addr(o),addr(w),addr(x),b,m,n,s,rb,ti,bt});
}
'''
CHECKS = r'''
int main() {
    constexpr uint64_t size=100000000, offa=4096, offb=40000000;
    auto *model=(unsigned char*)malloc(size); assert(model);
    float data[3]; unsigned cases=0;
    for (auto n : {1u,2u,8u,9u,31u,32u,33u,2048u,2049u,8192u,65535u})
    for (auto world : {1u,2u}) for (unsigned rank=0;rank<world;++rank)
    for (auto hw : {false,true}) for (auto quality : {false,true}) {
        gfx=hw;g_quality_mode=quality;
        ds4_gpu_tensor x{data,(uint64_t)n*32768/world*4};
        ds4_gpu_tensor low{data+1,(uint64_t)n*8192/world*4}, out{data+2,(uint64_t)n*5120*4};
        trace.clear();
        assert(world==1
            ? ds4_gpu_dsv41_attention_output_batch(&out,&low,model,size,offa,offb,&x,n)
            : ds4_gpu_dsv41_attention_output_tp_batch(&out,&low,model,size,offa,offb,&x,n,rank));
        const auto original=trace; trace.clear();
        assert(v41_attention_output_q8_a((float*)low.ptr,
            model+offa+rank*(8192/world)*128*34,(float*)x.ptr,n,8/world));
        assert(ds4_gpu_dsv41_quantize(&low,8192/world,n,DS4_V41_BF16));
        assert(v41_attention_output_q8_b((float*)out.ptr,
            model+offb+rank*(256/world)*34,(float*)low.ptr,n,8192/world));
        assert(trace==original); ++cases;
    }
    free(model);
    printf("ROCm V4.1 mixed Q8 projections: %u launch traces match original Q8/Q8 path\n",cases);
}
'''


def main():
    source = (ROOT / "rocm/ds4_rocm_v41.cuh").read_text()
    code = SHIM
    for signature in (
        'extern "C" int ds4_gpu_dsv41_attention_output_batch(',
        'extern "C" int ds4_gpu_dsv41_attention_output_tp_batch(',
        'static int v41_attention_output_q8_a(',
        'static int v41_attention_output_q8_b(',
    ):
        body = extract_function(source, signature)
        body = re.sub(r"(\w+(?:<[^<>]*>)?)<<<(.*?)>>>", r"launch(\1,\2)", body, flags=re.S)
        code += body
    code += CHECKS
    with tempfile.TemporaryDirectory(prefix="ds4-rocm-v41-q8-") as tmp:
        src, exe = Path(tmp) / "oracle.cpp", Path(tmp) / "oracle"
        src.write_text(code)
        subprocess.run(shlex.split(os.environ.get("CXX", "c++")) + [
            "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-g",
            "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
            str(src), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
