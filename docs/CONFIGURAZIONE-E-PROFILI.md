**English** | [Italiano](CONFIGURAZIONE-E-PROFILI.it.md)

# Configuration and execution profiles

This document explains how to read and apply the configuration. The complete
table of every key, value and default remains the
[Configuration Reference](../README.md#configuration-reference), which is the
authoritative source for individual parameters.

The historical `DS4_*` knobs described here belong to the DeepSeek V4 backend
unless stated otherwise. The GUI exposes them only when the runtime descriptor
declares the corresponding capability; the future Qwen backend will have its
own profile and will not automatically inherit expert cache, NSA or Q4
geometries.

## Configuration layers

Configuration comes from three layers:

1. GUI settings persisted in `UserDefaults`;
2. environment variables read by the engine and the demo;
3. local parameters of server, distribution, MCP, agents and downloads.

In the app the GUI exports the main `DS4_*` at startup and whenever they
change. For these knobs the value persisted by the app takes precedence over
the process environment. In the demo, in tests and in command-line tools the
environment takes precedence instead.

## When a value is read

Most memory, format and pipeline options are fixed at model load. Changing
them requires an unload/reload. Prefill options marked as updatable are
re-read on every call. Some sampling or UI settings apply on the next turn
without reloading the weights.

Before a benchmark, always record:

- GGUF and hash/size;
- RAM and Mac model;
- context and prompt;
- warm or cold cache;
- all non-default knobs;
- warm-up excluded from the measurement.

## Knob categories

| Category | Examples | Main risk |
|---|---|---|
| memory/I/O | dense stream, pread, mlock, bundle, MetalIO | RAM pressure or SSD contention |
| cache | expert slots, Q4 cache, on-disk KV | space and invalidation |
| prefill | chunk, union, route batch, FFN batch | transient memory peak |
| kernels | fusions and simdgroup count | GPU compatibility and parity |
| diagnostics | route profile, DIAG, warm-up | overhead that skews the measurement |
| quality | derived Q4/Q8, active experts | numerical or qualitative divergence |

## Lossless and lossy

Not all toggles mean the same thing:

- streaming, layout, prefetch, caches and fusions declared bit-identical do
  not reduce the model's information;
- some fusions or matmuls change the accumulation order and may differ by a
  few ulp;
- Q4/Q8 requantization and reducing the active experts are lossy.

Every shared profile must state explicitly which options are lossy. Disabling
`DENSE_Q4`, `QKV_Q4` and `SHARED_Q4` at the same time does not by itself
guarantee perfect quality if other lossy knobs remain or if the source GGUF is
heavily quantized.

## Measured low-RAM profile

The profile used as the baseline on 16 GB Macs favors:

- streaming of the dense weights;
- direct or bundled expert reads;
- best-effort pinning of hot buffers;
- an expert cache sized without saturating memory;
- Q4 for the large projections, when the quality trade-off is accepted;
- a prefill chunk large enough to amortize the weights;
- fusions and async pipeline already validated.

The best value depends on the actual RAM headroom. Applications open in
parallel can push macOS into compression/swap and cut decode by up to orders
of magnitude; before blaming MetalIO or a kernel for the slowdown, check
memory pressure and swap.

## Quality-oriented profile

To isolate quality:

1. use the GGUF of the desired quality;
2. keep `DS4_ACTIVE_EXPERTS=6`;
3. disable lossy derived caches (`DENSE_Q4`, `QKV_Q4`, `SHARED_Q4`,
   `COMP_Q8`);
4. use deterministic sampling or a fixed seed;
5. keep lossless streaming, bundles and caches if they help performance;
6. compare logits or text against the same prompt and template.

Lossless options should not be disabled on principle: changing the I/O path
does not change the weights being read.

## Diagnostic profile

`DS4_DIAG=1` enables the demo's extended report. `DS4_PROFILE_ROUTE=1` adds
synchronizations to separate substages and must not be used for the final
tok/s number. Use one knob at a time and keep usage file, prompt and cache
unchanged.

Example command shape:

```sh
DS4_DIAG=1 \
DS4_USAGE_FILE=/percorso/profilo.usage.json \
DS4_WARMUP=4 \
swift run -c release DS4Demo /percorso/modello.gguf 32 "Prompt di controllo"
```

The demo's full operational commands are in
[`Sources/DS4Demo/README.md`](../Sources/DS4Demo/README.md).

## Caches and derived files

- `.usage.json` contains routing frequencies, not modified weights;
- `.q4dense` and compressor caches contain derived representations;
- `.expbundle` reorders the same expert bytes for contiguous reads;
- KV checkpoints depend on model, tokens and a compatible configuration;
- distributed workers keep verified files in their own managed store.

A cache must be validated by size, version and model identity. If corruption
is suspected, remove only the affected derived file, not the original GGUF.

## Precedence in distribution

The coordinator sends the workers a whitelist of performance knobs. A worker
must not replace them with its own defaults after assignment. Lossy options
and sidecars travel in typed fields to prevent implicit configurations.

See [INFERENZA-DISTRIBUITA.md](INFERENZA-DISTRIBUITA.md).

## Documentation rule

When a new parameter is introduced:

1. document its default, read time and dependencies;
2. state whether it is bit-identical, equivalent or lossy;
3. add it to the Configuration Reference;
4. update the README of the owning folder;
5. if it is propagated to workers, update and test the whitelist;
6. add a reproducible A/B when it is a performance knob.

## GLM 5.2 knobs

The GLM backend has its own `DS4_GLM_*` namespace (setting a DeepSeek
`DS4_*` knob never affects GLM, and vice versa). All engine optimizations
are defaults; the knobs exist for A/B and diagnostics, are mirrored in the
GLM settings UI, and every value is dumped at engine init in the engine log
(`DS4 glm: knob …`). Measured verdicts are on an M1 Pro 16 GB with the
published IQ2_XXS.

| Knob | Default | Notes |
|---|---|---|
| `DS4_GLM_MTLIO` | 1 | 0 measured +18% decode on 16 GB (MetalIO contends with the decode commits); GUI preset sets 0 |
| `DS4_GLM_ACTIVE_EXPERTS` | 8 | routed experts per token; the shared expert always runs. 6 = −25% expert I/O, mild quality trade-off |
| `DS4_GLM_RESIDENT_LAYERS` | adaptive | floor of 3 dense layers under RAM pressure (extra residents get paged: ~+750 ms/token measured) |
| `DS4_GLM_FUSE` | 1 | commit fusion (half the synchronous waits) |
| `DS4_GLM_MOE_BATCH` | 1 | batched MoE (all routed experts in two dispatches) |
| `DS4_GLM_GPU_ROUTER` | 1 | fused GPU router (−18% prefill) |
| `DS4_GLM_MLOCK` | 1 | wire resident weights (head 433 → 39 ms/token) |
| `DS4_GLM_READ_SPLIT` | 4 | parallel prefill layer reads (prefill only; serial in decode by design) |
| `DS4_GLM_SG` / `DS4_GLM_NSG` | 1 / 4 | cooperative kernel dispatch and simdgroups per threadgroup |
| `DS4_GLM_STREAM_SLOTS` | 3 (4 fused) | layer staging slots (~250 MiB each) |
| `DS4_GLM_EXPERT_ARENA` | 24 | shared staged-expert arena slots (~10 MiB each) |
| `DS4_GLM_SPEC_EXPERTS` | off | expert speculation; N ≥ 2 = top-N only. Measured net-zero on a saturated SSD |
| `DS4_GLM_SPEC_K` | off | prompt-lookup speculative decode (demo) |
| `DS4_GLM_LAYERQ4` / `DS4_GLM_LAYERQ4_DIR` | 1 / sibling→managed | Q4 layer sidecar pack (`<gguf>.q4dense`) |
| `DS4_GLM_BUNDLE_DIR` | sibling→managed | legacy expert bundle pack (`<gguf>.expbundle`) |
| `DS4_GLM_NOCACHE` | off | F_NOCACHE reads; measured counterproductive as default |
| `DS4_GLM_USAGE_FILE` | `<gguf>.glm-usage.json` | usage imatrix persistence ("off" disables) |
| `DS4_GLM_AUTOTUNE` / `DS4_GLM_BUILD_LAYERQ4` | off | demo prepasses: auto-tune / sidecar pack build |

Dispatch knobs are re-read at every engine init (`GLM52DispatchKnobs`), so a
GUI toggle takes effect on the next model load without restarting the app.
