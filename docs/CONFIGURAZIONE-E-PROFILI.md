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
