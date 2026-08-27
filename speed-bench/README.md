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
