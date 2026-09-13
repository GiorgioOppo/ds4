"""Host checks for the production V4.1 Q4 MMVQ sample bridge.

Compile the extracted production function with instrumented CUDA launch
boundaries. The boundaries record arguments and inject launch errors; they do
not emulate quantization or matrix arithmetic. These checks cover admission,
packed token/group strides and error propagation. The native kernel is also
executed lane-by-lane with a dot-product recorder to check weight and activation
addresses for TP slices. No check here establishes CUDA execution or parity.
"""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from kernel_source import extract_function


ROOT = Path(__file__).resolve().parents[1]
COMMON = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <climits>
#include <algorithm>
using cudaStream_t = uintptr_t;
using cudaError_t = int;
enum ggml_type { GGML_TYPE_Q4_K = 12 };
constexpr int cudaSuccess = 0, MATRIX_ROW_PADDING = 512, QK8_1 = 32, QK_K = 256;
#define GGML_PAD(x, n) (((x)+(n)-1)/(n)*(n))
struct block_q8_1 { uint16_t d, s; int8_t qs[32]; };
static_assert(sizeof(block_q8_1) == 36, "canonical Q8_1 block size");
struct cuda_block_q4_K { uint16_t d,dmin; uint8_t scales[12],qs[128]; };
struct cuda_block_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; };
static_assert(sizeof(cuda_block_q4_K) == 144 && sizeof(cuda_block_q8_K) == 292,
              "GGUF block sizes");
static struct { uint32_t x, y; } blockIdx, threadIdx;
static const cuda_block_q4_K *expected_w;
static const cuda_block_q8_K *expected_x;
static uint32_t expected_block, expected_step, dot_calls;
static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *w,
                                    const cuda_block_q8_K *x) {
    assert(w == expected_w + expected_block);
    assert(x == expected_x + expected_block);
    expected_block += expected_step;
    ++dot_calls;
    return 0.0f;
}
static float quarter_warp_sum_f32(float acc, uint32_t lane) {
    assert(lane == (threadIdx.x & 7u));
    return acc;  // Address test only: no emulation of the CUDA reduction.
}
struct ggml_cuda_mm_fusion_args_device { const void *gate; };
struct Call {
    int quant = 0, mmvq = 0, sanitize = 0, errors = 0, fail_at = 0;
    std::array<int64_t, 8> q{};
    std::array<int, 18> m{};
    uint64_t sanitized = 0;
    cudaStream_t stream = 0;
} called;
static const char *cudaGetErrorString(cudaError_t) { return "injected launch error"; }
static cudaError_t cudaGetLastError() {
    return ++called.errors == called.fail_at ? 1 : cudaSuccess;
}
static void quantize_row_q8_1_cuda(
        const float *, const int32_t *ids, void *, ggml_type type,
        int64_t k, int64_t s1, int64_t s2, int64_t s3,
        int64_t n0, int64_t n1, int64_t n2, int64_t n3, cudaStream_t stream) {
    assert(!ids && type == GGML_TYPE_Q4_K);
    ++called.quant;
    called.q = {k, s1, s2, s3, n0, n1, n2, n3};
    called.stream = stream;
}
static void mul_mat_vec_q_switch_type(
        const void *, ggml_type type, const void *, const int32_t *ids,
        ggml_cuda_mm_fusion_args_device fusion, float *,
        int k, int m, int n, int wr, int yc, int oc,
        int wx, int yx, int ox, int wc, int yg, int og,
        int ws, int os, int wss, int yss, int oss, int id_stride,
        cudaStream_t stream) {
    assert(!ids && !fusion.gate && type == GGML_TYPE_Q4_K);
    assert(stream == called.stream);
    ++called.mmvq;
    called.m = {k, m, n, wr, yc, oc, wx, yx, ox, wc, yg, og,
                ws, os, wss, yss, oss, id_stride};
}
static void ds4_mmq_sanitize_f32(float *, uint64_t count, cudaStream_t stream) {
    assert(stream == called.stream);
    ++called.sanitize;
    called.sanitized = count;
}
'''

CHECKS = r'''
int main() {
    alignas(16) float input[16]{}, out[16]{};
    alignas(16) uint8_t weights[16]{};
    std::vector<uint64_t> arena(1024 * 1024);
    void *scratch = arena.data();
    assert(((uintptr_t)scratch & 15u) == 0);
    const int shapes[][3] = {{1024,4096,4}, {1024,4096,8}, {5120,8192,1}};
    for (const auto &shape : shapes) {
        const int m = shape[0], k = shape[1], groups = shape[2];
        for (int rows = 1; rows <= 64; ++rows) {
            // Independent layout oracle: one canonical Q8_1 block per 32
            // elements; each admitted K is already a 512-element multiple.
            const int qblocks = k / 32;
            const size_t need = size_t(rows) * groups * qblocks * 36;
            called = {};
            assert(ds4_mmq_q4_K_decode_samples(weights, input, out, scratch,
                need, m, k, rows, groups, 7) == 0);
            assert(called.quant == 1 && called.mmvq == 1 && called.sanitize == 1);
            const std::array<int64_t,8> quant = {
                k, k, int64_t(k)*rows*groups, int64_t(k)*rows*groups,
                k, rows*groups, 1, 1};
            assert(called.q == quant);
            const std::array<int,18> mmvq = {
                k, m, 1, k/256, qblocks, m, groups, groups, groups,
                m*(k/256), qblocks, m, 1, rows, 0, groups*qblocks, groups*m, 0};
            assert(called.m == mmvq);
            assert(called.sanitized == uint64_t(rows)*groups*m);
            // One byte below the canonical allocation must not enqueue work.
            called = {};
            assert(ds4_mmq_q4_K_decode_samples(weights, input, out, scratch,
                need-1, m, k, rows, groups, 7) != 0);
            assert(called.quant == 0 && called.mmvq == 0 && called.sanitize == 0);
        }
    }
    auto rejected = [&](const void *w, const float *x, float *y, void *s,
                        int m, int k, int rows, int groups) {
        called = {};
        assert(ds4_mmq_q4_K_decode_samples(w,x,y,s,arena.size()*8,
            m,k,rows,groups,7) != 0);
        assert(called.quant == 0 && called.mmvq == 0 && called.sanitize == 0);
    };
    rejected(nullptr,input,out,scratch,1024,4096,1,8);
    rejected(weights,nullptr,out,scratch,1024,4096,1,8);
    rejected(weights,input,nullptr,scratch,1024,4096,1,8);
    rejected(weights,input,out,nullptr,1024,4096,1,8);
    rejected(weights,input,out,(char *)scratch+1,1024,4096,1,8);
    for (int rows : {-1,0,65,INT_MAX})
        rejected(weights,input,out,scratch,1024,4096,rows,8);
    for (const auto &s : std::array<std::array<int,3>,7>{{
        {1023,4096,8}, {1024,4095,8}, {1024,4096,1}, {5120,8192,4},
        {1024,1280,8}, {INT_MAX,INT_MAX,INT_MAX}, {0,0,0}}})
        rejected(weights,input,out,scratch,s[0],s[1],1,s[2]);
    for (int stage = 1; stage <= 3; ++stage) {
        called = {};
        called.fail_at = stage;
        assert(ds4_mmq_q4_K_decode_samples(weights,input,out,scratch,
            arena.size()*8,1024,4096,64,8,7) == -(stage+1));
        assert(called.quant == 1);
        assert(called.mmvq == (stage >= 2));
        assert(called.sanitize == (stage >= 3));
    }
    // Compile and exercise the actual range helpers at adjacency and pointer
    // overflow boundaries. Integer-created pointers are never dereferenced.
    auto p = [](uintptr_t n) { return reinterpret_cast<void *>(n); };
    assert(!dsv41_output_range(nullptr, 1));
    assert(dsv41_output_range(p(UINTPTR_MAX-15), 15));
    assert(!dsv41_output_range(p(UINTPTR_MAX-15), 16));
    assert(dsv41_output_range(input, sizeof(input)));
    assert(!dsv41_output_overlap(p(100),20,p(120),10));
    assert(!dsv41_output_overlap(p(120),10,p(100),20));
    assert(dsv41_output_overlap(p(100),20,p(119),10));
    assert(dsv41_output_overlap(p(119),10,p(100),20));
    assert(dsv41_output_overlap(p(100),20,p(105),2));
    assert(dsv41_output_overlap(p(100),20,p(100),20));

    // Independent packed-layout oracle: full output weight rows are retained
    // across TP K slices, whereas activation rows contain only their shard.
    const uint32_t layouts[][5] = {
        {1024,16,16,0,8}, {1024,16,16,0,4},
        {5120,32,32,0,1}, {5120,16,32,0,1}, {5120,16,32,16,1}};
    for (const auto &layout : layouts) {
        const uint32_t outputs = layout[0], blocks = layout[1];
        const uint32_t full_blocks = layout[2], block0 = layout[3], groups = layout[4];
        constexpr uint32_t tokens = 3;
        std::vector<cuda_block_q4_K> w(groups*outputs*full_blocks);
        std::vector<cuda_block_q8_K> x(tokens*groups*blocks);
        std::vector<float> y(tokens*groups*outputs);
        for (uint32_t token : {0u, tokens-1})
            for (uint32_t row : {0u, outputs-1, groups*outputs-1})
                for (uint32_t lane = 0; lane < 8; ++lane) {
                    blockIdx = {row/32, token};
                    threadIdx = {(row%32)*8+lane, 0};
                    expected_w = w.data() + row*full_blocks + block0;
                    expected_x = x.data() + (token*groups + row/outputs)*blocks;
                    expected_block = lane;
                    expected_step = 8;
                    dot_calls = 0;
                    dsv41_output_q4_native_kernel(y.data(), (const char *)w.data(),
                        x.data(), outputs, blocks, full_blocks, block0, groups);
                    assert(dot_calls == blocks/8);
                }
        // The first entirely out-of-range row must not touch either input.
        blockIdx = {groups*outputs/32, 0};
        threadIdx = {0,0};
        dot_calls = 0;
        dsv41_output_q4_native_kernel(y.data(), (const char *)w.data(), x.data(),
            outputs, blocks, full_blocks, block0, groups);
        assert(dot_calls == 0);
    }
    puts("CUDA V4.1 Q4 host: admission, ranges, sample/group/TP strides and launch errors PASS");
}
'''


def main():
    production = (ROOT / "cuda/mmq/ds4_mmq.cu").read_text()
    body = extract_function(production, 'extern "C" int ds4_mmq_q4_K_decode_samples(')
    backend = (ROOT / "ds4_deepseek41_cuda.cuh").read_text()
    helpers = "\n".join(extract_function(backend, signature) for signature in (
        "static bool dsv41_output_range(",
        "static bool dsv41_output_overlap(",
        "__global__ static void dsv41_output_q4_native_kernel(",
    )).replace("__global__ ", "")
    with tempfile.TemporaryDirectory(prefix="ds4-v41-q4-host-") as directory:
        source = Path(directory) / "check.cpp"
        binary = Path(directory) / "check"
        source.write_text(COMMON + body + helpers + CHECKS)
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        subprocess.run(compiler + ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
            str(source), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
