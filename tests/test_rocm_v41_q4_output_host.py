"""Run the production typed ROCm wrapper and scalar kernel address walk on CPU.

Backend stubs record projections, TP strides, BF16 boundaries and injected
failures. This tests control flow/addressing, not HIP compilation or arithmetic.
"""
import os
from pathlib import Path
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
#include <cstring>
#include <string>
#include <vector>
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; };
struct cuda_block_q4_K { uint16_t d, dm; uint8_t scales[12], q[128]; };
struct cuda_block_q8_K { float d; int8_t q[256]; int16_t sums[16]; };
enum { ROCM_Q4_PREFILL_WMMA_REQUIRED_FAILURE=-1,
       ROCM_Q4_PREFILL_WMMA_FALLBACK=0, ROCM_Q4_PREFILL_WMMA_USE=1, DS4_V41_BF16=0 };
static unsigned rows, world, rank, at, bt, stage, resolves, cases;
static int fail, wmma, tile, require_tile, require_wmma;
static uint64_t scratch;
static std::string trace;
static char *model;
static constexpr uint64_t as = 4096, bs = 80000000, size = 140000000;
static ds4_gpu_tensor x, low, out;
static uint64_t rowbytes(unsigned type, unsigned k) {
    return type == 12 ? (uint64_t)k/256*144 : (uint64_t)k/32*34;
}
static int cuda_model_range_fits(uint64_t size, uint64_t offset, uint64_t n) {
    return offset <= size && n <= size-offset;
}
static int cuda_tensor_has_bytes(const ds4_gpu_tensor *p, uint64_t n) {
    return p && p->ptr && p->bytes >= n;
}
static const char *cuda_model_range_ptr(const void *p, uint64_t offset,
                                       uint64_t n, const char *) {
    assert(p == model);
    ++resolves;
    if (resolves == 1) {
        assert(offset == as + rank * (8192/world) * rowbytes(at,4096));
        assert(n == (8192/world)*rowbytes(at,4096));
    } else {
        assert(resolves == 2 && offset == bs && n == 5120*rowbytes(bt,8192));
    }
    return fail == (int)resolves ? nullptr : model + offset;
}
static int rocm_q4_K_prefill_tile8_scope(uint64_t n) { return n > 8 && n <= 8192; }
static int rocm_q4_K_prefill_tile8_requested() { return tile; }
static int rocm_q4_K_prefill_tile8_required() { return require_tile; }
static void rocm_q4_K_prefill_tile8_note(unsigned a,unsigned b,unsigned c,unsigned d,uint32_t n) {
    assert(!a && !b && c==1 && !d && n==rows && stage==3 && fail!=6);
}
static int rocm_q4_attn_q_b_env_bool(const char *) { return require_wmma; }
static int rocm_q4_K_prefill_wmma_select(uint64_t n, uint64_t k, uint64_t m) {
    assert(n == rows && k == 4096 && m == 1024); return wmma;
}
static cuda_block_q8_K *rocm_q4_K_prequant_alloc(uint64_t n, uint64_t k, const char *) {
    assert(trace.empty() && stage == 0 && k == 1);
    scratch = n*k;
    trace += 'P';
    static cuda_block_q8_K block;
    return fail == 3 ? nullptr : &block;
}
static int project(float *y, const char *w, const float *v, unsigned n,
                   unsigned groups, unsigned k, unsigned m, uint64_t stride,
                   unsigned type, int actual_wmma, int actual_tile) {
    assert(n == rows);
    if (!stage) {
        assert(type == at && y == low.ptr && v == x.ptr);
        assert(groups == 8/world && k == 4096 && m == 1024);
        assert(w == model + as + rank*groups*m*rowbytes(at,k));
        assert(stride == rowbytes(at,k));
        assert(actual_wmma == (at == 12 && !require_tile ? wmma : 0));
        trace += 'A'; stage = 1;
    } else {
        assert(stage == 2 && type == bt && y == out.ptr && v == low.ptr);
        assert(groups == 1 && k == 8192/world && m == 5120);
        assert(w == model + bs + rank*rowbytes(bt,8192)/world);
        assert(stride == rowbytes(bt,8192) && actual_wmma == 0);
        trace += 'B'; stage = 3;
    }
    if (type == 12) {
        assert(actual_tile == (rows > 8 && rows <= 8192 && tile));
        if (!actual_wmma) assert(scratch >= (uint64_t)rows*groups*(k/256));
    }
    return fail != (stage == 1 ? 4 : 6);
}
static int v41_attention_output_q4(float *y,const char *w,const float *v,
        uint32_t n,uint32_t groups,uint32_t k,uint32_t m,uint64_t stride,int wm,int ti) {
    return project(y,w,v,n,groups,k,m,stride,12,wm,ti);
}
static int v41_attention_output_q8_a(float *y,const unsigned char *w,const float *v,
        uint32_t n,uint32_t groups) {
    return project(y,(const char*)w,v,n,groups,4096,1024,rowbytes(8,4096),8,0,0);
}
static int v41_attention_output_q8_b(float *y,const unsigned char *w,const float *v,
        uint32_t n,uint32_t k) {
    return project(y,(const char*)w,v,n,1,k,5120,rowbytes(8,8192),8,0,0);
}
static int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *p,uint32_t width,uint32_t n,int fmt) {
    assert(stage == 1 && p == &low && width == 8192/world && n == rows && fmt == DS4_V41_BF16);
    trace += 'R'; stage = 2; return fail != 5;
}
static int legacy(ds4_gpu_tensor *y,ds4_gpu_tensor *l,const void *p,uint64_t sz,
        uint64_t a,uint64_t b,const ds4_gpu_tensor *v,uint32_t n,uint32_t wr,uint32_t rk) {
    assert(y==&out && l==&low && p==model && sz==size && a==as && b==bs && v==&x);
    assert(n==rows && wr==world && rk==rank && at==8 && bt==8);
    assert(!stage && trace.empty() && !resolves);
    trace += 'Q'; return fail != 4;
}
static int ds4_gpu_dsv41_attention_output_batch(ds4_gpu_tensor *y,ds4_gpu_tensor *l,
        const void *p,uint64_t sz,uint64_t a,uint64_t b,const ds4_gpu_tensor *v,uint32_t n) {
    return legacy(y,l,p,sz,a,b,v,n,1,0);
}
static int ds4_gpu_dsv41_attention_output_tp_batch(ds4_gpu_tensor *y,ds4_gpu_tensor *l,
        const void *p,uint64_t sz,uint64_t a,uint64_t b,const ds4_gpu_tensor *v,uint32_t n,uint32_t r) {
    return legacy(y,l,p,sz,a,b,v,n,2,r);
}
'''
CHECKS = r'''
static void reset() {
    // Recording stubs never dereference these disjoint virtual ranges.
    x = {(void*)UINT64_C(0x100000000000), (uint64_t)rows*(32768/world)*4};
    low = {(void*)UINT64_C(0x200000000000), (uint64_t)rows*(8192/world)*4};
    out = {(void*)UINT64_C(0x300000000000), (uint64_t)rows*5120*4};
    trace.clear(); stage=resolves=0; scratch=0; fail=wmma=require_tile=require_wmma=0; tile=1;
}
static int call(uint64_t sz=size,uint64_t a=as,uint64_t b=bs) {
    ++cases;
    return ds4_gpu_dsv41_attention_output_typed_batch(&out,&low,model,sz,a,b,at,bt,&x,rows,world,rank);
}
static void reject() { assert(!call() && trace.empty() && resolves==0); }
static void wrapper_checks() {
    model = (char*)malloc(size); assert(model);
    for (auto n : {1u,2u,8u,9u,31u,32u,33u,255u,256u,8192u,65535u})
    for (auto a : {8u,12u}) for (auto b : {8u,12u})
    for (auto wr : {1u,2u}) for (unsigned rk=0;rk<wr;++rk) {
        rows=n;at=a;bt=b;world=wr;rank=rk;reset();
        assert(call() && trace == (a==8 && b==8 ? "Q" : "PARB"));
        if (a==8 && b==8) continue;
        reset(); tile=0; assert(call() && trace=="PARB");
        if (a==12) {
            reset(); wmma=1; assert(call());
            assert(trace==(b==8 ? "ARB" : "PARB"));
            reset(); wmma=-1; assert(!call() && trace.empty() && !resolves);
        }
        if (n>8 && n<=8192) {
            reset(); require_tile=1; tile=0;
            assert(!call() && trace.empty() && !resolves);
            reset(); require_tile=1; wmma=a==12;
            assert(call() && trace=="PARB");
            if (a==12) {
                reset();require_tile=require_wmma=wmma=1;
                assert(!call() && trace.empty() && !resolves);
            }
        }
        for (int f=1;f<=6;++f) {
            reset();fail=f;assert(!call());
            assert(trace == (f<3 ? "" : f==3 ? "P" : f==4 ? "PA" : f==5 ? "PAR" : "PARB"));
        }
        reset();--x.bytes;reject();reset();--low.bytes;reject();reset();--out.bytes;reject();
        reset();x.ptr=nullptr;reject();reset();low.ptr=nullptr;reject();reset();out.ptr=nullptr;reject();
        reset();low.ptr=x.ptr;reject();reset();out.ptr=x.ptr;reject();reset();out.ptr=low.ptr;reject();
        reset();low.ptr=(char*)x.ptr+4;reject();
        reset();x.ptr=(char*)x.ptr+1;reject();
        reset();assert(!call(bs+5120*rowbytes(bt,8192)-1) && trace.empty() && !resolves);
        reset();assert(!call(size,UINT64_MAX) && trace.empty() && !resolves);
        reset();assert(!call(size,as,UINT64_MAX) && trace.empty() && !resolves);
        reset();out.ptr=model+bs;assert(!call() && trace.empty() && resolves==2);
    }
    rows=1;world=1;rank=0;at=bt=12;
    reset();rows=0;reject();reset();rows=65536;reject();rows=1;
    reset();world=0;reject();world=1;reset();world=3;reject();world=1;
    reset();rank=1;reject();rank=0;reset();at=2;reject();at=12;reset();bt=2;reject();
    free(model);
    printf("ROCm V4.1 typed attention: %u host dispatch/failure cases PASS\n",cases);
}
'''
KERNEL_SHIM = r'''
static struct { uint32_t x=0,y=0,z=0; } blockIdx, threadIdx;
static const cuda_block_q4_K *expected_w;
static const cuda_block_q8_K *expected_x;
static uint32_t expected_b, dots;
static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *w,const cuda_block_q8_K *x) {
    assert(w==expected_w+expected_b && x==expected_x+expected_b);
    expected_b+=8; ++dots; return 0;
}
static float rocm_q4_quarter_warp_sum_f32(float x,uint32_t lane) {
    assert(lane==(threadIdx.x&7));return x;
}
'''
KERNEL_CHECKS = r'''
static void address_checks() {
    // Production shapes, distinct rank offsets and full B row stride.
    unsigned walks=0;
    for (unsigned wr : {1u,2u}) for (unsigned rk=0;rk<wr;++rk)
    for (bool a : {false,true}) {
        const unsigned groups=a?8/wr:1, k=a?4096:8192/wr, m=a?1024:5120;
        const uint64_t stride=rowbytes(12,a?4096:8192), blocks=k/256;
        std::vector<cuda_block_q4_K> weights((a?8192:5120)*stride/144);
        std::vector<cuda_block_q8_K> input(3*groups*blocks);
        std::vector<float> output(3*groups*m+2,123.f);
        const char *w=(char*)weights.data()+(a?rk*groups*m*stride:rk*stride/wr);
        for (unsigned t=0;t<3;++t) for (unsigned g=0;g<groups;++g)
        for (unsigned r : {0u,31u,32u,m-1}) for (unsigned lane=0;lane<8;++lane) {
            blockIdx={r/32,t,g};threadIdx.x=(r%32)*8+lane;
            expected_w=(const cuda_block_q4_K*)(w+((uint64_t)g*m+r)*stride);
            expected_x=input.data()+((uint64_t)t*groups+g)*blocks;
            expected_b=lane;dots=0;
            v41_grouped_q4_q8_K_kernel(output.data()+1,w,input.data(),stride,blocks,m,3,groups);
            assert(dots==blocks/8 && output.front()==123.f && output.back()==123.f);
            if (!lane) assert(output[1+((uint64_t)t*groups+g)*m+r]==0);
            ++walks;
        }
        for (auto axis : {0u,1u,2u}) {
            blockIdx={axis==0?m/32:0,axis==1?3u:0u,axis==2?groups:0u};threadIdx.x=0;
            dots=0;v41_grouped_q4_q8_K_kernel(output.data()+1,w,input.data(),stride,blocks,m,3,groups);
            assert(!dots);
        }
    }
    printf("ROCm V4.1 Q4 scalar: %u token/group/rank/lane address walks PASS\n",walks);
}
int main() { wrapper_checks(); address_checks(); }
'''


def main():
    src = (ROOT / "rocm/ds4_rocm_v41.cuh").read_text()
    q4 = (ROOT / "rocm/ds4_rocm_q4.cuh").read_text()
    code = SHIM + extract_function(q4, "static int rocm_q4_K_byte_ranges_overlap(")
    code += extract_function(src, 'extern "C" int ds4_gpu_dsv41_attention_output_typed_batch(')
    code += CHECKS + KERNEL_SHIM
    code += extract_function(src, "__global__ static void v41_grouped_q4_q8_K_kernel(").replace("__global__ ", "")
    code += KERNEL_CHECKS
    with tempfile.TemporaryDirectory(prefix="ds4-rocm-v41-q4-") as tmp:
        source, binary = Path(tmp) / "oracle.cpp", Path(tmp) / "oracle"
        source.write_text(code)
        subprocess.run(shlex.split(os.environ.get("CXX", "c++")) + [
            "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-g",
            "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
            str(source), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
