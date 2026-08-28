## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

### Metal decode schedule A/B

Build the balanced, same-engine Metal decode comparison with:

```
make metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench \
  -m ds4flash.gguf \
  --include-selection
```

The harness prefills two sessions and alternates both variant order and
variant-to-session assignment. It aborts unless every full-vocabulary logit
row is bit-identical and, with `--include-selection`, both variants select the
same non-EOS token. Use `--candidate-env NAME` to measure a rollback control,
or `--help` to compare explicit split schedules. Pass `--ssd-streaming` for a
model larger than RAM; the harness then skips full-weight warmup while keeping
both variants in the same engine and expert cache. SSD runs can use
`--ssd-streaming-cold`, `--ssd-streaming-cache-experts N`, and
`--ssd-streaming-preload-experts N` to hold the cache policy constant.

This paired harness is the exact-logit gate. Since its two sessions share the
expert cache, confirm SSD throughput separately with one process and engine per
variant before promoting a scheduling change. Environment variables consumed
while the engine opens, including the Metal streaming `F_NOCACHE` controls,
also require separate processes: `--candidate-env` changes them too late to
create a different model descriptor inside this harness.

To compare the default pre-M5 ratio-4 compressor pack/transpose fusion with the
legacy decode path, including token selection, use:

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION \
  --include-selection \
  --tokens 1024
```

### Metal prefill variant A/B

Build the balanced prefill comparison. To compare the default resident pre-M5
MXFP4 pair tail-SIMDgroup cull against the original pair kernel, make the
rollback path the candidate:

```
make metal-prefill-variant-bench
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL
```

To isolate the default routed-down tail-SIMDgroup cull from the retained pair
default, use its down-specific rollback as the candidate:

```
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_TAIL_SIMDGROUP_CULL
```

The harness uses one Metal engine and fresh sessions for every run. It warms
both variants with at least 32 tokens, alternates control/candidate order in
ABBA and BAAB blocks, poisons host logit buffers before copying, and aborts
unless every final full-vocabulary logit row is bit-identical. Defaults are an
8192-token prefix, an automatically sized 8193-token context, and two repeats;
use `--help` to override them. SSD-prefill variants can add `--ssd-streaming`,
`--ssd-streaming-cold`, `--ssd-streaming-cache-experts N`, and
`--ssd-streaming-preload-experts N`. Since those paired runs intentionally
share one engine and expert cache, confirm any SSD throughput win separately
with one process and engine per variant before promotion.
Environment variables consumed while the engine opens, including the Metal
streaming `F_NOCACHE` controls, cannot be compared with `--candidate-env` in
this harness and likewise require separate processes.

### Metal Q4_K generic-MM tail cull

Build and run the GGUF-free kernel-only comparison with:

```
make metal-q4-mm-tail-cull-bench
./speed-bench/metal_q4_mm_tail_cull_bench
```

The default `4096 -> 1024` shape models a Flash Q-A projection. Use
`--in-dim 1024 --out-dim 32768` to measure `attn_q_b`. Both arms dispatch the
checked-in production Metal kernels with resident rotating Q4_K weights and
GPU timestamps. The harness covers `N=9,16,17,31,33,47,63,65`, alternates
ABBA/BAAB, and requires bit-exact outputs plus intact input/output canaries.
No GGUF access, SSD I/O, upload, readback, or CPU wall time is included in a
measured command buffer.

### Resident ROCm Q4_K prefill

Build the production-dispatch A/B harness on a ROCm host with:

```
make rocm-q4-prefill-bench ROCM_ARCH=gfx1151
./speed-bench/rocm_q4_prefill_bench
```

The fixture copies four rotating sets of synthetic GGUF-layout Q4_K weights
to device memory before warmup and forces SSD streaming off. HIP events then
measure only the activation quantizer and Q4 projection kernels. The three
default comparisons are:

- `dense`: legacy versus TILE8 at the Flash Q-A `K=4096,M=1024` shape;
- `pair`: two TILE8 calls versus the fused Q-A/KV
  `K=4096,M=(1024+512)` path;
- `qb`: TILE8 versus TILE4 at the production `attn_q_b`
  `K=1024,M=32768` shape.

The default token set is `9,17,33,128,512`, covering the first row after each
small token-tile boundary plus medium prefill. Use `--full` for
`9,16,17,31,32,33,128,512,4096`, or select a focused run such as:

```
./speed-bench/rocm_q4_prefill_bench \
  --case qb --tokens 9,17,33,128,512,4096 --samples 8
```

Every case rotates identical resident weight sets between arms, alternates
ABBA/BAAB, checks the outputs bit-for-bit, and verifies allocation guards.
Fixture creation, the host-to-device residency copy, warmup, oracle readback,
and environment-gate changes are outside the reported HIP-event intervals.
`candidate_delta_pct` is negative when the candidate is faster; the companion
`speedup_pct` reports the positive speedup convention.

### Resident CUDA Q4_K prefill

Build the production-API kernel harness on a CUDA host with an explicit
architecture:

```
make cuda-q4-prefill-bench CUDA_ARCH=sm_121
./speed-bench/cuda_q4_prefill_bench --path mmq
```

The default `dense`, `pair`, and `qb` cases use the same production shapes and
token set as the ROCm harness. The synthetic GGUF-layout weights are copied by
the backend into a `cudaMalloc` allocation before warmup. A CUDA test hook
checks the backend-owned pointer provenance and device attributes of every
dense, KV, and q_b range in every rotating weight set; a global free-memory
delta is printed only as a diagnostic and is not accepted as residency proof.
CUDA events measure the production backend GPU interval, including
stream-ordered scratch allocation/free, tail clears, activation quantization,
Q4 projection, and output sanitization. Model uploads, output poisoning, full
finite-output scans, sampled CPU Q4_K oracles, canary checks, and warmups are
outside the event interval.

`cuda_use_mmq()` caches its first decision for the life of the process, so the
dense and `attn_q_b` MMQ-versus-Q8_K comparison deliberately uses separate
processes. The MMQ process also enables a test-only strict control, so an MMQ
rejection fails the run instead of silently measuring the Q8_K fallback. Run
both orders to balance thermal/order drift:

```
# ABBA
./speed-bench/cuda_q4_prefill_bench --path legacy --case dense
./speed-bench/cuda_q4_prefill_bench --path mmq    --case dense
./speed-bench/cuda_q4_prefill_bench --path mmq    --case dense
./speed-bench/cuda_q4_prefill_bench --path legacy --case dense

# BAAB
./speed-bench/cuda_q4_prefill_bench --path mmq    --case dense
./speed-bench/cuda_q4_prefill_bench --path legacy --case dense
./speed-bench/cuda_q4_prefill_bench --path legacy --case dense
./speed-bench/cuda_q4_prefill_bench --path mmq    --case dense
```

Repeat with `--case qb` for `K=1024,M=32768`. Each result records the immutable
path as `path=legacy` or `path=mmq`; do not toggle `DS4_CUDA_MMQ` around calls
inside another harness. The default MMQ `pair` case is a true in-process
ABBA/BAAB comparison between two public dense calls and the public fused pair
API, with bit-exact pair outputs. Prefill pair is skipped under `--path legacy`
because the CUDA pair API intentionally returns control to two independent
dense projections for `N>8` when MMQ is disabled.

### Resident IQ2/Q2 MoE prefill on ROCm and CUDA

The backend-neutral fixture uses the production `N=4096`, 256-expert, top-6
IQ2_XXS/Q2_K geometry and a deterministic routing distribution containing
every 32-row tail from 1 through 31. Weights and tensors are resident and SSD
streaming is disabled, so the reported GPU-event intervals isolate kernel
work rather than storage throughput. The fixture needs roughly 4 GiB of
explicit host/device storage in addition to backend runtime overhead.

On a wave32 ROCm host, build and run the real balanced A/B with:

```
make rocm-iq2-moe-prefill-bench ROCM_ARCH=gfx1151
./speed-bench/gpu_iq2_moe_prefill_bench_rocm
```

The baseline sets the dominant rollback
`DS4_ROCM_DISABLE_IQ2_MOE_WMMA_TAIL_CULL=1`; the candidate sets
`DS4_ROCM_ENABLE_IQ2_MOE_WMMA_TAIL_CULL=1`. The harness alternates ABBA/BAAB,
requires bit-exact intermediate scratch and final tensors, verifies allocation
canaries, and prints `cudaEvent` time for only the IQ2 gate/up and Q2 down
rocWMMA kernels. The candidate remains opt-in until real-hardware results show
a repeatable win. A wave64 device takes the scalar fallback and therefore
cannot produce a valid candidate timing.

On CUDA, build and run the measurement-only current path with an explicit
architecture:

```
make cuda-iq2-moe-prefill-bench CUDA_ARCH=sm_121
./speed-bench/gpu_iq2_moe_prefill_bench_cuda
```

CUDA prints `DS4_CUDA_MOE_PROFILE` stage times for the resident IQ2 MMQ path
and performs a structural/canary oracle. Every marked call must produce exactly
one completed fast-path profile record or the harness fails. It intentionally
does not claim an A/B result: CUDA's cooperative D2R CTA has different
synchronization and tail semantics, so the ROCm/Metal wave-cull selector cannot
be copied safely.
On a discrete CUDA GPU with enough VRAM, prefix the run with
`DS4_CUDA_COPY_MODEL=1` to keep the raw expert weights in device memory and
remove mapped-host/PCIe stalls from the measured kernel interval; this adds
about 1.7 GiB of device storage. Leave it unset on GB10/UMA when measuring the
normal aligned-artifact production selector, because forcing a raw model copy
changes that residency path.
