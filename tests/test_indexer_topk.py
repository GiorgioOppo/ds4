#!/usr/bin/env python3
"""Host-check the real compact Metal merge body and shared geometry."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]


def actual_leaf_source(source):
    leaf = extract_function(source, 'kernel void kernel_argsort_f32_i32(')
    start = leaf.index('    for (int k = 2;')
    end = leaf.index('    const int64_t i0 =', start)
    network = leaf[start:end]
    # Compare pairs are disjoint within each stage. Execute all real lane
    # bodies before advancing to the next GPU barrier.
    network = network.replace('            int ixj = col ^ j;',
        '            for (int col = 0; col < int(nth); ++col) {\n'
        '            int ixj = col ^ j;')
    network = network.replace('            threadgroup_barrier(mem_flags::mem_threadgroup);',
                              '            }')
    assert 'threadgroup_barrier' not in network
    return r'''
enum { DS4_SORT_ORDER_ASC, DS4_SORT_ORDER_DESC };
#define SWAP(a,b) std::swap(a,b)
static std::vector<int32_t> actual_leaf(const float *scores, uint32_t n, uint32_t begin, uint32_t nth) {
    const struct { int ne00; } args = {int(n)};
    const struct { int x; } ntg = {int(nth)};
    constexpr auto order = DS4_SORT_ORDER_DESC;
    const int i00 = int(begin);
    std::vector<int32_t> shmem_i32(nth);
    std::vector<float> shmem_f32(nth, std::numeric_limits<float>::quiet_NaN());
    for (uint32_t i = 0; i < nth; ++i) {
        shmem_i32[i] = int32_t(begin+i);
        if (begin+i < n) shmem_f32[i] = scores[begin+i];
    }
''' + network + r'''
    shmem_i32.resize(std::min(nth,n-begin));
    return shmem_i32;
}
'''


def main():
    source = (ROOT / "metal/argsort.metal").read_text()
    name = "kernel_argsort_merge_f32_i32_desc_compact"
    kernel = extract_function(source, 'kernel void ' + name + '(')
    body = kernel[kernel.index('{'):].replace("device ", "")
    constants = extract_function(source, 'struct ds4_metal_args_argsort_compact_merge {') + ';\n'
    generated = """// Extracted production Metal body; only address spaces are removed.
static void actual_compact_merge(const ds4_indexer_topk_merge_args &args,
    const char *src0, const int32_t *tmp, int32_t *dst,
    uint3 tgpig, ushort3 tpitg, ushort3 ntg)
""" + body + "\n" + constants + actual_leaf_source(source)
    compiler = shlex.split(os.environ.get("CXX", "clang++"))
    if not compiler:
        raise SystemExit("CXX must name a C++ compiler")
    with tempfile.TemporaryDirectory(prefix="ds4-indexer-topk-") as directory:
        temp = Path(directory)
        (temp / "indexer_topk_actual.h").write_text(generated)
        for label, flags in (("strict", ["-ffp-contract=off"]),
                             ("fast", ["-ffast-math", "-fno-finite-math-only"])):
            exe = temp / label
            command = compiler + ["-std=c++17", "-O2", "-g", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer"] + flags + [
                "-I", str(temp), "-I", str(ROOT), str(ROOT / "tests/test_indexer_topk.cpp"),
                "-o", str(exe)]
            subprocess.run(command, check=True)
            subprocess.run([str(exe)], check=True)
            print(f"PASS: {label} actual-source merge + geometry, ASan/UBSan", flush=True)


if __name__ == "__main__":
    main()
