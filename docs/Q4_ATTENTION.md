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
  --ssd-streaming \
  --ctx 4096 --prefill-chunk 128 \
  --host 127.0.0.1 --port 8000
```

Omitting `--ssd-streaming-cache-experts` lets the runtime size the cache for
the model, context, and available working-set budget. A plain number such as
`16` requests only 16 expert slots across the model, not 16 GiB; this can
increase SSD reads during decode. Use an explicit budget only when matching
a benchmark configuration or tuning for the available memory. The streaming
path still needs memory for non-expert weights, activations, KV state, and
backend workspace. See the [SSD memory guide](SSD_STREAMING.md).
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

On eligible pre-M5 Metal devices, direct grouped Q4 output-A and fused Q4
output-B/HC prefill accept up to 8192 tokens. Output-A handles partial tiles;
output-B/HC requires a multiple of 32 tokens and remains resident-only.
Their tile scratch is fixed-size, while activation storage grows with the
batch. The separate Q-A/KV shared-input path retains its 256-token limit
because it uses a fixed 2 MiB staging buffer. Metal4 dispatch is unchanged.

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

### Decode defaults retained from the development branch

The original clean port omitted some automatic decode optimizations outside
the Q4 matrix kernels. The follow-up restores these specific paths:

- CUDA HC split/normalization at width 4096 and one row: sixteen blocks produce
  the weighted sum, one block preserves the reference reduction order, and
  sixteen blocks store normalized output. Scratch must be available before
  submitting a writer; overlapping buffers retain the reference kernel.
- CUDA Q8_0 activation quantization: a warp shuffle performs the same maximum
  reduction without six block barriers. Quality mode and the rollback retain
  the shared-memory implementation; the Q8_K part of dual quantization keeps
  its original reduction and rounding.
- CUDA GB10 F16 compressor projections: restore the fused pair and state
  store used in `d12f4480`/`ff749b84`. For K=4096, widths 256/1024 at ratio 4
  and width 512 at ratio 128, an immutable transposed half2 cache coalesces
  paired weight loads and retains the ordered 32-lane sum. Cache construction
  happens outside capture; a cache miss during capture or an exhausted cache
  uses the fused canonical-layout kernel. Quality and other GPU paths retain
  the existing implementation. The cache is bounded to 1 GiB per loaded model
  and released after graph invalidation at model teardown.
- CUDA GB10 shared-expert Q8 down projection with HC expansion: restore the
  aligned weight loads from the same historical snapshots. The activation
  quantizer, accumulation and HC epilogue match the canonical Q8 path. This
  uses existing model artifacts and adds no weight cache allocation.
- ROCm gfx1151 F16 compressor projections: fuse the existing shared-input
  KV/gate pair with its state write for K=4096, widths 256/1024 at ratio 4 and
  width 512 at ratio 128. The wave32 products and reduction remain unchanged;
  the final FP32 values feed both projection outputs and ring state. No weight
  cache or activation conversion is added. Quality mode, graph diagnostics
  and other devices retain the previous path. Buffer and alias checks precede
  the writer, and a launch failure cannot request a fallback replay.
- Metal Q8 matvec and paired matvec with four SIMD groups: remove a redundant
  barrier while retaining each kernel's reduction and output ownership.
- Metal SSD shared-expert Q8 gate/up at four or eight SIMD groups: the same
  barrier reduction applies to the fused SwiGLU producer.
- Metal fused F16 compressor pair/quad, including the compound Q8 QKV+quad
  kernel: independent KV/gate partial-sum planes share one publication
  barrier. The final FP32 values write both projections and ring state
  directly, preserving the original reduction order and APE addition. This
  removes the device-memory barrier and projection reload from the previous
  fused epilogue. Existing device and shape gates remain in force.
- Metal M1 IQ2_XXS paired matvec: compute the eighth sign bit with integer
  population count instead of loading a random entry from the shared sign
  table. All 128 masks, floating-point operations, reductions, output stores
  and scratch requirements are unchanged. Selection is automatic at shader
  compilation; other Apple GPU generations retain their existing path.
- CUDA and ROCm raw IQ2_XXS routed experts: pass the seven-bit code directly
  to the existing population-count helper, eliminating a redundant lookup
  of the expanded sign byte. The grid-LUT decode and tile8 kernels also drop
  their 128-byte shared/LDS sign table and its initialization. Grid and
  activation staging still require the existing barriers. ROCm applies the
  same simplification to direct IQ2 dequantization in the WMMA path. Packed
  integer weights, DP4A operations, FP scaling and dispatch remain unchanged.

These are automatic decode paths, not new Q4 quantization formats. They also
apply to compatible models whose attention remains Q8. A separate restored
ROCm correctness fix assigns one writer to each raw-KV ring cell when a batch
is larger than the ring; only its newest rows survive.

On M1 Max, the IQ2 sign change reduced median paired kernel latency by 5.0%
with eight rotating weight sets (198 MiB), and 7.2% with one warm set
(24.8 MiB). Each fast-math comparison used 24 alternating A/B samples of
128 dispatches. The isolated kernel passed bitwise checks in strict and fast
modes. These timings exclude CPU encoding and SSD reads; unstable
single-dispatch cache-pressure measurements are excluded. They do not
establish a 10% improvement in model throughput.

An eight-run SSD-streaming comparison against `20e728e` on the same M1 Max
(24 GPU cores, 32 GiB RAM) used A/B/B/A followed by B/A/A/B. With the
IQ2XXS-w2Q2K-AProjQ4-SExpQ8-OutQ8 model, prompt `narrami la storia di roma`,
`--ctx 4096 --prefill-chunk 128 --nothink --temp 0 -n 50` and the same
automatic cache budget, mean generation was 4.84 versus 4.965 tokens/s
(+2.6% observed). All eight runs generated identical output. Individual
results varied from 4.62 to 5.32 tokens/s, so this small full-model difference
needs longer measurements before treating it as a stable throughput gain.
This 16-token prompt takes the SSD decode-style prefill path; it does not
measure the 128-token Q4 matrix kernel. Larger Q4 matrix tiles were tested
separately and excluded because they did not reliably improve that shape.

The CUDA/ROCm IQ2 port targets raw decode (including CUDA SSD streaming) and
the prefill paths that call these helpers. CUDA's aligned-artifact decode,
raw MMQ and D2R paths already reconstruct signs with population count. The
aligned-SoA MMQ loader retains its existing `ksigns64` packed-mask table;
that table replaces additional mask-expansion instructions and is a separate
optimization. Consequently this port does not imply a speedup for the main
MMQ prefill path. The port still requires native CUDA/HIP build validation
and correctness/performance runs on NVIDIA/AMD hardware; the M1 timing
results above do not establish a gain on either backend.

## Validation

The host suite requires no model or GPU:

```sh
make test-cpu-q4 test-quantizer-indexer-q4
make test-q4-preflight-host
make test-gpu-iq2-signs-host
make test-cuda-hc-split-norm-host test-cuda-q8-quantize-host \
  test-rocm-raw-kv-store-host
make test-cuda-f16-compressor-host test-cuda-q8-hc-aligned-host
make test-rocm-f16-compressor-host test-rocm-q4-prefill-dispatch-host
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
make test-metal-decode-defaults test-metal-decode-fusions test-metal-f16-compressor
make test-metal-iq2-signs test-metal-ssd-experts
# Larger activation allocations; each fixture runs separately.
make test-metal-q4-prefill-long
```

The decode fusion oracle checks 68 bitwise cases with strict math and again
with fast math: F16 pair/quad and compound QKV+quad, both APE types, ring
positions, and Q8/HC SIMD-group variants. The F16 integration fixture checks
64 additional cases through the real backend APIs in eager and batched
command modes. Long-prefill fixtures compare direct output-A at 8191/8192
tokens and fused output-B/HC at 8192 against the existing separate paths,
including output guards and rejection beyond the dispatch limits.

The IQ2 sign oracle checks all 128 sign codes on the CPU, then compiles the
actual paired helper and ID, six-slot, address and masked-address wrappers
with the popcount specialization disabled and enabled. It compares raw
gate/up and SwiGLU outputs bitwise in strict and fast modes, including empty,
partial and full masks, padded strides, nonzero offsets and guards. The SSD
expert fixture separately exercises the production backend and cache eviction.

The CUDA/ROCm sign oracle extracts the current production helpers and checks
every combination of 128 sign codes and 256 grid entries, signed DP4A inputs,
complete 256-value blocks and batches of one to eight tokens. Its table-based
reference preserves each helper's original floating-point order. Host mode
uses ASan/UBSan in strict and fast builds; it models GPU intrinsics and does
not validate GPU compilation or kernel scheduling. Native modes compile and
run those same helpers on-device. They supplement the normal backend build
and full-model tests; they do not exercise the whole routed-MoE dispatcher.

On M1 Max (2026-09-10), a synthetic comparison against `080ac7d` measured
the fused F16 kernels with fast math, warm weights, one warm-up pair and
12 alternating old/new samples of 2048 dispatches each. Median GPU latency
fell from 51.16 to 39.87 microseconds for the K=4096, width=512, ratio=128
pair, and from 98.64 to 72.46 microseconds for the K=4096, widths=1024/256,
ratio=4 quad. These are component timings, excluding model execution, CPU
encoding and SSD reads. The standalone pair's existing automatic M3/M5 gate
is unchanged; its M1 measurement directly invokes the shader. The quad
integration fixture exercises the actual M1 backend dispatch.

On NVIDIA hardware, compile and run the native oracles:

```sh
make test-mmq-parity-cuda test-cuda-q4-epilogue CUDA_ARCH=sm_121
make test-cuda-iq2-signs CUDA_ARCH=sm_121
make test-cuda-hc-split-norm test-cuda-q8-quantize CUDA_ARCH=sm_121
make test-cuda-f16-compressor test-cuda-q8-hc-aligned CUDA_ARCH=sm_121
make test-cuda-q4-prefill-dequant test-cuda-q4-prefill-reduce \
  test-cuda-q4-prefill-norm CUDA_ARCH=sm_121
```

On Strix Halo, require a visible device so a skipped test cannot count as
validation:

```sh
make test-strix-rocm-q4-parity test-strix-rocm-q4-prefill
make test-rocm-iq2-signs ROCM_ARCH=gfx1151
make test-strix-rocm-q4-prefill-long
make test-rocm-f16-compressor ROCM_ARCH=gfx1151
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

ROCm Q4 tiled paths accept up to 8192 tokens per call, matching explicitly
requested large runtime chunks. Previously, the 4096-token TILE8/WMMA gates
sent larger batches to the legacy path. Aligned output-B staging and the Q-B
normalization/RoPE epilogue use the same extended limit. Their per-block
geometry and shared-memory footprint are unchanged; total batch scratch grows
with the token count. The transient F16 Q-B threshold remains 4096 tokens.

An 8192-token attention batch has 65536 flattened token/group rows. The Q8_K
producer splits these into two 32768-row launches to respect the portable
HIP grid-y limit; the GEMM remains one tiled launch per projection. Smaller
quantizer grids keep their previous dispatch. Every slab retains the original
batch row count and reduction mode, including a short final slab. A submitted
quantizer failure aborts the operation instead of replaying a fallback.

The host dispatch oracle exercises the 65535/65536 boundary, offsets, shape
selection and launch failures. The native long-prefill fixture compares large
batches with bounded reference calls. Native gfx1151 compilation, numerical
validation and model throughput still need to be measured; these changes do
not establish a speedup or close the reported Q4/Q8 gap by themselves. Compare
both revisions at each of `--prefill-chunk 4096` and `--prefill-chunk 8192`,
using the same model and balanced run order. For component measurements:

```sh
make rocm-q4-prefill-bench ROCM_ARCH=gfx1151
./speed-bench/rocm_q4_prefill_bench --tokens 4096,8192
make bench-rocm-f16-compressor ROCM_ARCH=gfx1151
```

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

### ROCm routing, Q2 staging and paired Q4 metadata

The ROCm prefill kernels remove three sources of repeated work:

- Stable expert routing uses one native wave per expert instead of one
  thread. A ballot and exclusive population count compact consecutive pair
  IDs in their original order. For 4096 tokens and six selected experts,
  gfx1151 scans 768 wave windows instead of 24576 serial iterations per
  expert. The algorithm still examines every pair for every expert; this is
  parallelization, not a reduction in asymptotic complexity or total bytes.
  The launch also supports wave64. Invalid IDs retain their previous handling.
- The resident IQ2/Q2 hot-expert down projection stages 32 K values at once
  on gfx1151 for batches of at least 128 tokens with F16 input/output. Two
  consecutive K16 WMMA steps reuse that stage in the original accumulation
  order. At K2048, inner block barriers fall from 256 to 128. After the final
  K barrier, the output tile reuses the dead A stage, saving 4096 bytes of
  scratch. Dynamic LDS falls from 9856 to 8832 bytes despite the larger K
  stage, plus the existing 256-byte pair table in both cases. Occupancy and
  elapsed time must still be checked on AMD hardware. Other dispatch cases
  retain K16 staging.
- Q4 DP4A and resident WMMA K64/K128 decode adjacent scale/minimum groups
  together. Ten 16-bit metadata accesses replace twenty scalar byte accesses
  per weight block in the source, with the same logical bytes and numerical
  operands. The compiler may already combine some older accesses: this is
  not a measured halving of load instructions or runtime. Activation
  quantization, matrix tiles, precision and projection selection are unchanged.

These changes add no runtime opt-in. The common MoE improvements can also
benefit models with Q8 attention and the same IQ2/Q2 experts. Limited-cache
SSD batches can return through separate streamed-expert kernels before
stable routing and hotlist WMMA; do not attribute the MoE changes to that
mode without confirming dispatch. The Q4 metadata helper also serves the
DP4A paths used by streaming and small prefill chunks.

The targeted validation and native microbenchmark commands are:

```sh
make test-rocm-q4-dot-host test-rocm-moe-prefill-host
make test-rocm-moe-prefill ROCM_ARCH=gfx1151
make bench-rocm-moe-prefill ROCM_ARCH=gfx1151
make test-strix-rocm-q4-prefill-long
make rocm-q4-prefill-bench ROCM_ARCH=gfx1151
./speed-bench/rocm_q4_prefill_bench --tokens 128,512,1024,4096
```

The Q4 host oracle checks every combination of contributing metadata bytes
and guarded packed/aligned DP4A outputs. The MoE host test executes extracted
routing and A/B staging code with wave32/wave64 simulations and sanitizers;
it does not emulate WMMA arithmetic or GPU synchronization. Native HIP
compilation, guard checks and timing remain required. Generated MoE test
sources and executables live in temporary directories.

One extra Apple clang Q4 check combining fast-math and ASan/UBSan reproduces
the same one-ULP scalar-oracle discrepancy on both the baseline and candidate
(`0x5255e6a8` versus `0x5255e6a9`, trial 0, N1, lane 7). Strict sanitizer and
ordinary strict/fast builds pass. Disabling FP contraction resolves this
extra comparison on both revisions; the production arithmetic and oracle have
not been relaxed to conceal it. This host compiler interaction is separate
from native AMD numerical validation.

For an end-to-end comparison, build separate baseline and candidate checkouts
with `make strix-halo ROCM_ARCH=gfx1151`; the baseline before these changes is
`a081850b5f8474f6ec5f68d429dc2746e547ef27`. Run the same AProjQ4 GGUF and prompt
with each executable, keeping ROCm version, power settings and memory budget
fixed. For example, from each checkout:

```sh
./ds4-bench --rocm -m /path/to/model-AProjQ4.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 8192 --ctx-alloc 8193 \
  --step-incr 2048 --gen-tokens 0 --prefill-chunk 128 \
  --warm-weights --csv /tmp/baseline-chunk128-resident-01.csv
```

Give every run its own CSV filename. Compare chunk 128 and chunk 2048
separately. For the SSD comparison, replace `--warm-weights` with
`--ssd-streaming --ssd-streaming-cache-experts 16GB`, or the tester's actual
cache budget, identically for both revisions. Discard one warmup per variant
and mode, then collect at least twelve runs in balanced ABBA/BAAB order.
Disable profiling and use identical runtime controls. Validate frontier
logits in separate runs with `--dump-frontier-logits-dir` pointing to an
existing directory; timing runs should omit dumps.

Each CSV frontier measures only its new suffix, here 2048 tokens. Aggregate
throughput is `sum(prefill_tokens) / sum(prefill_tokens / prefill_tps)`, not
the arithmetic mean of token rates. Report individual runs and spread,
alongside candidate/baseline throughput. The requested +8% requires a ratio
of at least 1.08 on the declared workload (about 7.41% less total time), with
noise small enough to distinguish the gain. Host correctness tests and
static work reductions do not establish that result.

### ROCm activation preparation reuse

The follow-up to `0fe51ea27a8e3520565cd470a61ac46630b72ad3` targets repeated
computation on gfx1151. It spends additional temporary memory to reuse
activation conversions and quantization:

| Path | Previous preparation | Reused preparation | Extra live scratch |
| --- | --- | --- | --- |
| Q4 Q-B K1024, M32768, N256–2048, resident K128 WMMA | Each of 128 output-row tiles converts the same F32 activation values to F16 | Convert each activation once, then copy half8 vectors into the same K128 LDS layout | At most 4 MiB |
| IQ2 paired gate/up K4096, E256, top-k 6, N128–2048 on gfx1151 | Quantize six expert assignments per token | Quantize each token once, then gather complete 144-byte MMQ records into the existing assignment layout | At most 9 MiB |

Q-B keeps the original weight decoding, F16 rounding, WMMA accumulation and
output epilogue. Its automatic selection requires tight strides and one
group; quality mode and SSD streaming retain their existing paths. The
existing dense temporary arena is checked for input/output aliases before
growth can free its old allocation. Allocation failure can retain the
original K128 path before conversion starts; a conversion or consumer launch
failure is reported without replay. The long-batch Q-B weight sidecar has
separate ownership and selection.

For IQ2 gate/up, the original D4 quantizer produces the compact token buffer.
The gather copies all four F32 scales and 128 quantized values, preserving
the original expert maps, MMQ consumers and padding. Unwritten map entries
still select quantized token zero. The compact pool allocation stays alive
through gather and both matrix multiplications on the caller's stream.
CUDA compilation and the direct gate/up Q8 path retain their previous code.
This shared MoE change can also benefit a model with Q8 attention and the
same IQ2 experts. It also applies to SSD calls that load the complete expert
table for a layer; selected/compact expert-table calls retain their separate
paths. Pool allocation failure retains the pool's existing fatal-error
semantics, rather than switching algorithms after preparation has started.

These are reductions in preparation work, not in the matrix multiplication
operation count. At N2048, Q-B performs 2,097,152 F32-to-F16 element conversions
instead of 268,435,456 across row tiles. The IQ2 path quantizes 2,048 source
rows instead of 12,288 assignments. Each adds one kernel launch, and the
gather still writes the full MMQ input. Neither ratio predicts end-to-end
token throughput; extra launches and traffic may offset the savings.

Run the host fixtures, then the actual HIP kernels on gfx1151:

```sh
make test-rocm-q4-activation-host test-rocm-mmq-quant-reuse-host
make test-rocm-q4-activation test-rocm-mmq-quant-reuse ROCM_ARCH=gfx1151
make bench-rocm-q4-activation bench-rocm-mmq-quant-reuse ROCM_ARCH=gfx1151
```

The Q-B native comparison includes conversion and K128 compute in candidate
timings. The MMQ native comparison includes the original output memset and
all quantize/gather launches, but excludes allocation, routing and MMQ
compute. Both microbenchmarks require whole-model timing before claiming a
prefill improvement. Host tests exercise the shared source with sanitizers;
they cannot validate GPU synchronization, generated instructions or speed.

A separate test-only Q4_K × Q8_K INT8 WMMA prototype preserves the eight
Q8_K K-block partitions, per-block scale/minimum correction and final
reduction tree. It uses a 16×16 tile, eight wave32 groups and 8 KiB of LDS.
Its fragment layout follows the existing RDNA3 MMQ implementation and the
[AMD WMMA description](https://gpuopen.com/learn/wmma_on_rdna3/). The source
requires sixteen INT8 WMMA instructions per K256 segment of the 16×16 tile
to keep eight independent 32-value scales per weight row. It is not selected
by inference: registers, native parity and elapsed time must be checked before integration. The
fixture's production-dot oracle is a numerical reference, not a substitute
for benchmarking the complete existing TILE8 kernel.

```sh
make test-rocm-q4-int8-wmma-host
make test-rocm-q4-int8-wmma ROCM_ARCH=gfx1151
```

Use the repeated end-to-end protocol above with `0fe51ea` as the baseline
for this follow-up, testing chunk 128 and chunk 2048 separately. Only the
MoE reuse applies at chunk 128. Compare collected pure-prefill CSVs with:

```sh
python3 speed-bench/compare_prefill.py \
  --baseline /tmp/baseline-chunk128-resident-*.csv \
  --candidate /tmp/candidate-chunk128-resident-*.csv --target-percent 8
```

The script checks matching frontiers, sums elapsed times within each run,
and reports per-frontier and aggregate medians, min/max and observed gains.
Use `--json` to include every run's rates. CSVs do not identify the model,
GPU or runtime options; verify those match before comparison. Meeting the
median threshold alone is not a confidence interval or a numerical parity
check. Native HIP compilation and the requested +8% remain unverified on
the macOS development machine.

### Metal activation preparation reuse

The Metal port applies the same reuse principle to the existing SIMDgroup
matrix paths on pre-M5 Apple GPUs. Q-B uses the existing
`kernel_mul_mm_q4_K_f16_rhs`; the fused IQ2 gate/up kernel gains an F16 input
specialization of the same template. Both keep weight decoding, F16 operand
rounding, accumulation order and output arithmetic.

| Path | Automatic selection | Reused preparation | Fixed queue scratch |
| --- | --- | --- | --- |
| Direct Q4 Q-B | K1024/M32768, N128–2048, resident, pre-M5, no quality/TP/concurrent encoder | One F32-to-F16 conversion instead of one per 64-row output band (512 bands for a full token tile) | 4 MiB |
| Fused IQ2 gate/up + SwiGLU | IQ2/Q2 experts, K4096/M2048, E256/top-k 6, N512–2048, resident, pre-M5, TP1 | One token-compact F16 input shared by all selected experts and output-row tiles | 16 MiB |

Metal's fused MoE pair consumes floating-point activations. The ROCm Q8
record gather therefore has no direct counterpart here: the new input stays
token-compact, and the original expert map selects its rows. Gate and up
remain fused, and the down projection consumes the same F16 intermediate.
Existing M5 MPP and packed-activation paths keep their selection, as do SSD,
quality and small-batch paths. Existing Q-B weight sidecars are tried before
the direct projection and retain their own activation preparation.

The new buffers have fixed capacities, so later batches never replace a
buffer still referenced by an unretained command buffer. The serial queue
and tracked resource hazards order reuse; the copy encoder ends before the
consumer starts. Cleanup drains submitted and unsubmitted work before
releasing scratch. Allocation or pipeline failure before conversion can use
the original path. Encoding failure after conversion propagates without
replaying matrix computation. No new runtime environment controls are added.

Native validation and preparation-inclusive microbenchmarks are available
without loading a GGUF:

```sh
make test-metal-q4-activation test-metal-q4-activation-runtime
make test-metal-moe-activation
make bench-metal-q4-activation
make bench-metal-moe-activation
```

The Q-B runtime fixture checks automatic selection against the explicit F32
oracle, input/output guards, aliases, fallback cases, three consecutive GPU
producers sharing one scratch buffer, and cleanup with an unsubmitted batch.
It passes 36 cases in each of retained and unretained command-buffer modes
on M1 Max. The standalone Q-B comparison also covers incomplete output/token
tiles. Its benchmark selects the same boundary-check specialization and copy
thread count as the runtime, includes conversion and encoder boundaries, and
rotates eight synthetic weight matrices. The MoE fixture extracts the actual
IQ2 decoder, map, copy and paired kernels, comparing complete F16 outputs in
strict and fast Metal compilation modes. Its benchmark includes copy and
paired compute but excludes the common map and down projection.

Q-B timing with the exact runtime specializations was too variable to
establish a stable percentage gain on the local M1 Max. The IQ2 case at
128 tokens likewise showed no distinguishable gain, so its automatic reuse
starts at 512 tokens. These limits are deliberate: fewer conversions alone
do not prove faster matrix computation.

Two isolated M1 Max benchmark runs passed all 81 MoE cases, including strict
and fast parity checks and the production-shape timing cases. With
K4096/M2048, 256 experts and top-k 6, 14 samples per arm in balanced ABBA
order gave the following gate/up + SwiGLU throughput gains, including the
activation copy:

| Tokens | First run, means | Second run, means | Second run, medians |
| --- | --- | --- | --- |
| 128 (benchmark only) | +0.14% | +0.32% | +0.20% |
| 512 | +1.81% | +1.82% | +1.88% |
| 2048 | +2.61% | +3.34% | +3.44% |

These synthetic resident-weight measurements use uniform routing. Actual
expert distributions and the common map/down work can change the benefit.

These checks validate local kernels and queue integration. Model-level
prefill throughput, SSD runs and M2–M4 performance require separate
measurements; a stage speedup does not establish an 8% end-to-end gain. Use
the repeated CSV protocol above with `--metal` and `b09f8e3` as the baseline
for this port. Keep chunk size in CSV filenames and compare frontier logits
separately from timing.

### CUDA grouped output-A candidate

An isolated benchmark evaluates the eight-token grouped kernel from
[adamlawi/ds4 at b4922c9](https://github.com/adamlawi/ds4/blob/b4922c9614eaed560c4988739b03648652f5b09f/ds4_cuda.cu#L30189).
It reuses each Q4 block across up to eight tokens and consumes the native
head/output layout. Its Q8_K quantizer splits large token/group row counts
across launches so `grid.y` never exceeds 65535. The inference backend does
not include this candidate and no new runtime switch is introduced.

```sh
# Host layout, launch-boundary and production dot-helper checks.
make test-cuda-q4-grouped-tok8-host

# Native correctness checks, followed by balanced CUDA-event measurements.
# Select the architecture of the GPU being tested.
make bench-cuda-q4-grouped-tok8 CUDA_ARCH=sm_121 Q4_TOK8_TOKENS=512
```

`Q4_TOK8_DEVICE` selects the device (default 0). The native fixture checks
token/row tails, a batch exceeding 65535 quantizer rows, guarded buffers and
graph replays on a non-default stream. It compares against a scalar Q8_K
reference and reports numerical differences from the grouped Q8_1 MMQ path.
On GPUs other than GB10 it also measures the pack/MMQ/scatter sequence used
by the current fallback. Timings include activation quantization and all
projection work, with scratch allocation and warmup outside the samples.
This is an eager comparison: the candidate owns preallocated scratch and
MMQ uses its warmed pool. Inference configurations that register persistent
MMQ scratch can have different overhead.

Q8_K and Q8_1 activation quantization are different: diagnostic error metrics
are not a model-quality acceptance test. Native CUDA compilation, timing and
full-model quality measurements are required before changing inference
dispatch. Host sanitizer checks do not establish those results. Generated
translation units and executables are created in temporary directories.

## Measuring decode recovery

The F16 compressor and aligned Q8 HC restorations above recover two enabled
paths absent from `670e6b9`; they are not a measured recovery of the reported
GB10 +6.5%. The public `adamlawi/aprojq4-dense-attention` head `b4922c9` predates
the bisected `d12f4480`/`ff749b84` snapshots. The latter are the sources for
these restorations. The upstream aligned-Q8-pair scratch change at the
`d12f4480` performance step was already present in `670e6b9`.

On GB10, compare the same AProjQ4 file, fixed prompt, context and token limit
against `670e6b9`, alternating executable order and checking generated output.
Record startup/cache preparation separately. A same-binary rollback disables
both restored paths with `DS4_CUDA_NO_F16_PAIR_COMPRESSOR_STORE=1` and
`DS4_CUDA_NO_Q8_FUSED_ALIGNED=1`. Kernel timings are available with
`make bench-cuda-f16-compressor CUDA_ARCH=sm_121`; these use prebuilt cached
weights and report eight balanced rounds for separate, fused canonical and
fused transposed kernels. They do not measure model throughput or cache setup.

The F16 host oracle compares all four output/state arrays, untouched ring rows
and guards for 72 input/position/type cases, including UINT32_MAX positions,
under both contraction modes with ASan/UBSan. Native mode additionally checks
three graph replays with fresh inputs; 106 host policy checks cover cache
lifetime, aliases, capture and allocation/launch failures. The aligned Q8 HC
oracle checks 40 kernel cases and 31 dispatch/fault cases in strict and fast
builds, plus 8192 independent integer-dot comparisons. Native compilation,
GPU correctness and performance of these two restorations remain unverified
locally.

Compare the same Q4 GGUF, prompt, generated-token limit, sampling parameters,
context, and cache budget. Changing the SSD expert cache is a separate
experiment from changing kernels. Alternate the two binaries in A/B and B/A
order and compare generated text as well as throughput.

For the CUDA HC path, use the same binary with and without its rollback:

```sh
./ds4 --cuda --temp 0 --nothink -n 400 -c 131072 \
  -m /path/to/model.gguf --prompt-file /path/to/prompt.txt
DS4_CUDA_NO_HC_SPLIT_NORM_SPLIT4096=1 \
  ./ds4 --cuda --temp 0 --nothink -n 400 -c 131072 \
  -m /path/to/model.gguf --prompt-file /path/to/prompt.txt
```

Repeat in reversed order. This isolates the HC implementation; it does not
measure Q4 versus Q8 or reproduce every optimization in the old branch.
Test the Q8 activation reduction separately with
`DS4_CUDA_DISABLE_Q8_QUANT_WARP_REDUCE=1`; set both rollbacks to compare the
combined restored CUDA defaults with their reference implementations.
The host HC fixture checks the extracted arithmetic and dispatch policy; its
native CUDA mode checks actual kernels and captured graph replays. CUDA/HIP
device validation and the reported DGX Spark throughput recovery require
those devices and cannot be established by host simulations.

## Port provenance

The clean history starts at upstream `6289c516273979173abbc062209a81dd3706b804`.
Q4 functionality was selected from
`aprojq4-dense-attention` at `dff1543f33bdfb6d2e6023413cc3093fd093b8fa`, after
the experimental Q4 kernel cleanup. The port retains the automatic Q4 paths
and their required helpers, adapted to upstream's runtime and Metal queue.
The decode follow-up retains the narrowly scoped HC/Q8 defaults described
above. DSpark experiments, other MoE/IQ2 changes, removed kernel candidates,
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
