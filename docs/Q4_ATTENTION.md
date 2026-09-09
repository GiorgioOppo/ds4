# Q4_K attention support

[Model formats](MODELS.md) | [SSD streaming](SSD_STREAMING.md) | [Q4 controls](Q4_CONTROLS.md)

This branch supports DeepSeek V4 dense attention projections stored as GGUF
`Q4_K` (tensor type 12). The five projections are `attn_q_a`, `attn_q_b`,
`attn_kv`, `attn_output_a`, and `attn_output_b`. The sparse indexer's
`indexer.attn_q_b` projection also accepts Q4_K. Dispatch follows each tensor's
type, so a model may retain Q8_0 or F16 tensors where supported by its layout.

Q4_K uses 256-value blocks with packed four-bit values and scale/min metadata.
Attention quantization is independent of routed experts, the shared expert,
the output head, and the KV-cache format. An `AProjQ4` filename describes the
attention projections; it does not mean every tensor or the KV cache is Q4.

## Build and run

Build for the host using the existing platform target:

| Backend | Build | Runtime selector |
| --- | --- | --- |
| Apple Metal | `make -j4` | `--metal` |
| CUDA on DGX Spark | `make cuda-spark` | `--cuda` |
| Other CUDA devices | Follow [CUDA setup](CUDA_MULTI_GPU.md) | `--cuda` |
| ROCm on Strix Halo | `make strix-halo` | `--rocm` |
| CPU reference | `make cpu` | `--cpu` |

No Q4 enable flag is needed. For a memory-constrained Mac, start with an
explicit SSD-streaming configuration:

```sh
./ds4-server --metal \
  -m /path/to/DeepSeek-V4-Flash-AProjQ4.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 16 \
  --ctx 4096 --prefill-chunk 128 \
  --host 127.0.0.1 --port 8000
```

Choose the expert-cache and context sizes for the available memory. The
streaming path still needs memory for non-expert weights, activations, KV
state, and backend workspace. See the [SSD memory guide](SSD_STREAMING.md).
The existing upstream DSpark and serving configurations retain their own
backend restrictions.

To create an AProjQ4 model from an existing Q8_0/F16 GGUF, use
`gguf-tools/deepseek4-quantize --source-gguf ... --attention-proj q4_k`.
The [quantizer guide](../gguf-tools/README.md#requantize-a-gguf-directly) covers
the dry run, imatrix requirements, and optional `--indexer-q q4_k` conversion.
Use a distinct output file and validate it before use.

## Execution paths

| Backend | Decode and small batches | Prefill |
| --- | --- | --- |
| CPU | Q4_K × Q8_K dot products; reusable activation scratch; grouped output-A | Two tokens share packed-weight decoding; sufficiently large activation batches are prepared in parallel |
| Metal | Native Q4 matvec, shared Q-A/KV input, eligible Q-B token pairs and output-B/HC expansion | Native Q4 matrix kernels, shape-limited shared F16 input for Q-A/KV, direct grouped output-A, and eligible automatic Q-B F16 staging |
| CUDA | Q4 MMVQ/MMQ, shared Q8_1 input for projection pairs, eligible grouped output-A and fused output sanitization | Canonical Q4 MMQ, per-group strided output-A MMQ on GB10, and eligible long-batch transient F16 Q-B staging |
| ROCm | Dense and grouped Q4 paths with Q8_K input; prefill pairs retain their shape gates | Resident gfx1151 direct-Q4 WMMA for eligible projections, exact Q8_K + TILE8 for output-B, aligned/vector LDS staging, and eligible transient F16 Q-B staging |

Shape, architecture, quality mode, residency, and tensor placement determine
eligibility. A clean rejection uses the general implementation. A failure
after submitting a writer is fatal to that operation; the caller must not
retry a fallback over partially submitted work. The shared API documents
this `1` / `0` / `-1` distinction.

The Q4 CUDA output-TP split that assumes Q8_0 weights is rejected explicitly.
This port does not claim every multi-GPU or tensor-parallel configuration is
validated for AProjQ4.

### Q-B F16 staging and memory

The automatic path expands one layer's Q4_K Q-B weights into reusable F16
scratch for sufficiently long batches. Its default threshold is 4096 tokens
on all three GPU backends, and its fixed K=1024, M=32768 weight expansion uses
64 MiB, with additional activation workspace. Shorter batches retain native
Q4 kernels. Metal admits this path on eligible pre-M5 devices in resident or
SSD-streaming mode. CUDA requires a physically device-resident model image
on a single GPU; ROCm accepts a device image or device-owned weight ranges.
CUDA and ROCm exclude SSD streaming. Quality mode excludes this staging path
on all three backends.

Metal also retains a persistent F16 weight cache for eligible resident
pre-M5 runs. It is selected only by requiring that cache or disabling the
transient path; the automatic transient path does not fall through to it.
Its default cap is 3072 MiB and its default minimum batch is 512 tokens.
Allocation also accounts for the working-set limit and future session
reservations. Cache entries belong to their model mapping, and SSD streaming
always excludes this persistent cache.

Preparation occurs before prompt submission. `ds4_session_prepare_sync()`
lets a benchmark perform one-time preparation before its timed prefill;
ordinary `ds4_session_sync()` performs the same preflight automatically.
Adding sessions, changing mappings, and tearing down the final session must
invalidate or release the corresponding resources at a synchronized point.

Controls are listed in [Q4_CONTROLS.md](Q4_CONTROLS.md). Keep them unset for
normal operation. Rollback and `REQUIRE` controls are useful for targeted
parity tests; they are not prerequisites for Q4 model support.

## Validation

The host suite requires no model or GPU:

```sh
make test-cpu-q4 test-quantizer-indexer-q4
make test-q4-preflight-host
make test-q4-epilogue-host test-q4-prefill-dequant-host \
  test-q4-prefill-reduce-host test-cuda-q4-prefill-norm-host \
  test-cuda-q4-dequant-flat-host test-rocm-q4-dequant-flat-host \
  test-cuda-mmq-dense-ids-host
make test-rocm-q4-dot-host test-rocm-q4-lds-host \
  test-rocm-q4-lds-aligned-host test-rocm-q4-wmma-load-host \
  test-rocm-q4-qb-epilogue-host
```

Metal fixtures exercise the actual GPU kernels with generated weights,
guard regions, odd batch sizes, fallback shapes, and cache transitions:

```sh
make test-metal-indexer-q4 test-metal-q4-prefill-pair \
  test-metal-q4-qb-token-pair test-metal-q4-attn-out-a-direct \
  test-metal-q4-qb-f16-cache test-metal-q4-hc
```

On NVIDIA hardware, compile and run the native oracles:

```sh
make test-mmq-parity-cuda test-cuda-q4-epilogue CUDA_ARCH=sm_121
make test-cuda-q4-prefill-dequant test-cuda-q4-prefill-reduce \
  test-cuda-q4-prefill-norm CUDA_ARCH=sm_121
```

On Strix Halo, require a visible device so a skipped test cannot count as
validation:

```sh
make test-strix-rocm-q4-parity test-strix-rocm-q4-prefill
make test-rocm-q4-prefill-dequant ROCM_ARCH=gfx1151
```

The local port was validated on Apple M1 Max with CPU tests and native Metal
fixtures. Host simulations and syntax checks cover CUDA/ROCm indexing,
reductions, layout, and policy. They do not establish NVCC/HIP compilation,
GPU scheduling correctness, or performance on those devices. The Metal4
shader library was compiled locally; M5 tensor execution still requires M5
hardware. Full-model Q4/Q8 quality and throughput comparisons remain separate
from these synthetic fixtures.

## Measuring prefill

Use identical prompts, batch/chunk sizes, context, backend, expert-cache
budget, and warmup for Q4 and Q8 runs. Record the GPU, exact commit, model
hashes, residency/SSD mode, and any non-default controls. Alternate Q4/Q8 run
order and retain individual results; compare medians across repeated runs.
Report one-time preparation separately from steady prefill throughput.

The CUDA and ROCm projection benchmarks are built with
`make cuda-q4-prefill-bench CUDA_ARCH=sm_121` and
`make rocm-q4-prefill-bench ROCM_ARCH=gfx1151`. Run their `--help` for shapes
and timing options. Metal has `metal-q4-dense-pair-bench`,
`metal-q4-prefill-pair-bench`, `metal-q4-mm-tail-cull-bench`, and
`metal-q4-attn-out-a-direct-bench` build targets.

Kernel timings isolate a dispatch or projection. They cannot establish an
end-to-end improvement or guarantee that Q4 prefill is faster than Q8.

## Port provenance

The clean history starts at upstream `6289c516273979173abbc062209a81dd3706b804`.
Q4 functionality was selected from
`aprojq4-dense-attention` at `dff1543f33bdfb6d2e6023413cc3093fd093b8fa`, after
the experimental Q4 kernel cleanup. The port retains the automatic Q4 paths
and their required helpers, adapted to upstream's runtime and Metal queue.
DSpark experiments, unrelated MoE/IQ2 changes, removed kernel candidates,
compiled binaries, and historical benchmark reports are outside this port.

The commits separate common CPU/model support, GPU implementations with
their regression fixtures, and this usage/validation reference. Benchmark
artifacts and tester results belong outside the source commits.

| Commit | Review scope |
| --- | --- |
| `1fcd662` | CPU Q4 dispatch, model/indexer validation, direct GGUF conversion and production CPU/quantizer tests |
| `c9474f6` | Shared GPU ABI, Metal/CUDA/ROCm kernels, prompt preflight, cache lifetime, native/host fixtures and benchmark sources |

The GPU commit keeps backend implementations and their shared signatures
together. Its Metal lifecycle fixture releases a transient-only batch before
an explicit command wait, checking completion and output parity before the
source mapping can be unmapped.
