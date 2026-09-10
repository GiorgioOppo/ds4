#!/usr/bin/env python3
"""Check extracted Q-B F16 conversion, staging and dispatch; --rocm runs WMMA.

Host strict/fast ASan/UBSan validates addressing and launch/error contracts.
Native --rocm [--bench] compares the unchanged F32 K128 input against conversion
plus reused F16 input. A native run requires gfx1151 and never silently skips.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]


def build_source(native):
    q4 = (ROOT / 'rocm/ds4_rocm_q4.cuh').read_text()
    common = (ROOT / 'rocm/ds4_rocm_common.cuh').read_text()
    convert = extract_function(common, '__global__ static void f32_to_f16_kernel(')
    definitions = [convert]
    names = [
        'static int rocm_q4_K_prefill_wmma_load4_compatible(',
        'static void rocm_q4_K_prefill_wmma_k128_enqueue(',
        'static int rocm_q4_K_prefill_wmma_k128_half_enqueue(',
        'extern "C" int ds4_rocm_bench_q4_K_wmma_k128_enqueue(',
        'extern "C" int ds4_rocm_bench_q4_K_wmma_k128_half_enqueue(',
    ]
    if native:
        signature = '__global__ static void\nrocm_matmul_q4_K_prefill_wmma_k128_p144_rowtile_strided_kernel('
        body = extract_function(q4, signature)
        begin = q4.rfind('template <', 0, q4.index(signature))
        definitions.append(q4[begin:q4.index(signature)] + body)
    else:
        names += [
            'static int rocm_q4_K_prefill_wmma_load2_compatible(',
            'static int rocm_q4_K_prefill_wmma_k64_control_policy(',
            'static int rocm_q4_K_prefill_wmma_k128_policy(',
            'static int rocm_q4_K_prefill_wmma_k128_half_try(',
            'static int rocm_q4_K_prefill_wmma_launch(',
        ]
    definitions += [extract_function(q4, name) for name in names]
    if not native:
        body = extract_function(q4, '__global__ static void\nrocm_matmul_q4_K_prefill_wmma_k128_p144_rowtile_strided_kernel(')
        stage = extract_function(body, 'for (uint32_t j = tid * 4u;')
        definitions.append('''static void stage_f32_baseline(_Float16 *lds_x, const float *x,
                uint32_t n_tok, uint32_t tok0, uint32_t k0, uint32_t tid) {
            const uint32_t group=0, group32_base=k0/32u;
            const uint64_t x_token_stride=1024, x_group_stride=0;
        ''' + stage + '\n}')
    text = '\n'.join(definitions)
    if not native:
        text = text.replace('__global__', '')
        text, count = re.subn(r'f32_to_f16_kernel<<<(.*?)>>>\(',
                             r'conversion_launch(dim3(\1), ', text, flags=re.S)
        assert count == 1, count
        text, count = re.subn(
            r'rocm_matmul_q4_K_prefill_wmma_k128_p144_rowtile_strided_kernel<\s*'
            r'256u, 16u, 1u(, __half)?><<<grid, 512u>>>\(',
            lambda match: ('wmma_half_launch(' if match[1] else 'wmma_float_launch(') + 'grid, ',
            text)
        assert count == 2, count
    fixture = (ROOT / 'tests/test_rocm_q4_activation.cpp').read_text()
    return ('#define TEST_NATIVE\n' if native else '') + fixture.replace('// PRODUCTION_SOURCE', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rocm', action='store_true')
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if args.bench and not args.rocm:
        parser.error('--bench requires --rocm')
    with tempfile.TemporaryDirectory(prefix='ds4-q4-activation-') as tmp:
        source = Path(tmp) / 'activation.cpp'
        source.write_text(build_source(args.rocm))
        binary = Path(tmp) / 'activation'
        if args.rocm:
            compiler = shlex.split(os.environ.get('HIPCC', 'hipcc'))
            if not compiler or not shutil.which(compiler[0]):
                parser.error('HIPCC and a gfx1151 AMD GPU are required')
            flags = shlex.split(os.environ.get('ROCM_CFLAGS', '-O3 --offload-arch=gfx1151'))
            subprocess.run(compiler + flags + ['-x', 'hip', '-std=c++17', '-I', str(ROOT),
                str(source), '-o', str(binary)], check=True)
            subprocess.run([str(binary), str(args.device)] + (['--bench'] if args.bench else []), check=True)
        else:
            compiler = shlex.split(os.environ.get('CXX', 'clang++'))
            for flags in (['-O2'], ['-O3', '-ffast-math', '-fno-finite-math-only']):
                print('Q4 activation host ' + ' '.join(flags), flush=True)
                subprocess.run(compiler + flags + ['-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-Wno-unknown-pragmas', '-fsanitize=address,undefined', '-I', str(ROOT),
                    str(source), '-o', str(binary)], check=True)
                subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
