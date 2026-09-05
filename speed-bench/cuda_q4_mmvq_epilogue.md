# CUDA Q4 decode: fused output sanitization

## Scope and mechanism

For aligned, single-token dense Q4_K MMVQ calls on NVIDIA Turing or newer,
the canonical matvec now sanitizes its result when storing it. Previously each
call enqueued output zeroing, the matvec, and a separate sanitizer that read
the entire output again. Every eligible row is written exactly once, so the
pre-clear and standalone sanitizer are unnecessary with the new store.

This removes one memset submission and one kernel launch per eligible
projection. The shared-activation dense pair still launches the same two
matvecs; each eligible leg removes its own pre-clear and sanitizer.
Neither quantization nor dot products, block assignment, warp count, reduction
tree, stream selection, allocation, or synchronization is changed. There is
no new weight sidecar or activation format.

Admission requires `N=1`, positive `M%4=0`, positive `K%256=0`, and
`M*(K/256) <= INT_MAX`. Four-row alignment covers both the canonical small-K
four-row kernel and the ordinary one-row kernel. The persistent K1024 candidate
and its oracle retain their old epilogue. HIP/ROCm, Metal, other quant types,
grouped/MoE entry points and multi-token MMVQ/MMQ are not controlled by this
dense flag. The separate [grouped MMVQ specialization](cuda_q4_grouped_mmvq_fusion.md)
has its own rollback and additional tests. Model storage
policy is unchanged; resident/SSD runs can benefit only when they actually
reach the affected dense MMVQ entries.

## Rollback and tests

The path is enabled by default when eligible. Set
`DS4_CUDA_DISABLE_Q4_MMVQ_EPILOGUE=1` to restore the previous sequence. Any
defined value, including `0` or an empty value, disables it. This is read at
enqueue time; changing it does not rewrite a previously captured CUDA Graph.
Use fresh processes for full-model A/B measurements.

```sh
make test-q4-epilogue-host
make test-cuda-q4-epilogue CUDA_ARCH=sm_121
compute-sanitizer --tool memcheck ./tests/test_cuda_q4_epilogue
```

Use the architecture appropriate for the tester's GPU. The CUDA target is
available in the Linux/CUDA Makefile branch and requires the CUDA toolkit.

The host test directly exercises the production bit-classification helper
against `isfinite` over 4,194,320 patterns, including signed zeros, subnormals,
NaNs and infinities. It checks 112,860 shape/admission cases, models row
coverage, and rejects overflowing block indices. It does not compile CUDA
device code or establish numerical parity of the GPU matvec.

The GPU oracle invokes the production single and shared-pair APIs for 40 cases
(and now runs 20 additional grouped cases under the independent flag):
small-K and ordinary dispatch, production widths (including 32768x1024),
unequal/reversed pair widths, nonfinite weights, zero/changing activations,
mixed eligible/ineligible legs, multi-token fallback and output/input guards.
Odd row counts use fixture-only physical weight padding for the legacy loader;
the new default path never admits them. Each case compares default versus
rollback `1`, `0` and the empty string, both directly and after graph
capture/repeated replay.
It also requires graph node counts to drop by exactly one memset and one
sanitizer kernel per eligible output, preventing candidate-vs-candidate or
silent-fallback comparisons from passing as optimization coverage.

## Validation status

Local host tests pass, also under AddressSanitizer and UndefinedBehaviorSanitizer.
A source comparison confirms that the canonical kernel body differs only by
the guarded output store and its assertion. Single and pair calls share one
dense launch helper with the original strides and dispatch arguments; clearing,
sanitization and launch order remain with the callers. `git diff --check`
passes and the Linux CUDA test build recipe was checked with `make -n`.

CUDA/nvcc and HIP are not installed in the local Mac
environment, so GPU compilation, the 60-case GPU oracle, compute-sanitizer,
and performance measurements are **pending**. No remote machine was contacted.
No percentage speedup, GPU parity result, or end-to-end decode gain is claimed.

For throughput, compare fresh processes with the same binary, model, prompt,
context, warmup, cache policy and graph settings; alternate adjacent legacy
and default runs and form ratios within each paired round. Do not compare
different quantizations when isolating this flag. Validate correctness first.
