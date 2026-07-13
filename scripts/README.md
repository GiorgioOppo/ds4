# Scripts

Build and analysis scripts.

Performance methodology and configuration precedence are documented in
[`docs/VALUTAZIONE-DEMO-PERF.md`](../docs/VALUTAZIONE-DEMO-PERF.md) and
[`docs/CONFIGURAZIONE-E-PROFILI.md`](../docs/CONFIGURAZIONE-E-PROFILI.md).

- **`embed_kernels.sh`** regenerates
  `Sources/DS4Metal/Runtime/Generated/KernelSources.swift` from `metal/*.metal`, embedding
  kernel source code into the binary. It is invoked by `make embed-kernels`.
- **`bench.sh`** runs the prefill/decode knob benchmark matrix with the release
  `DS4Demo` binary and collects the key lines of every run into a single report:
  `scripts/bench.sh <gguf> <prompt-file> [report.txt] [case ...]`.
  Its built-in matrix intentionally preserves the historical 16-slot baseline
  used by the July 2026 measurements. It is an A/B harness, **not** the current
  GUI preset (22 slots, extended Q4 cache, and MetalIO with automatic fallback).
  Record every effective `DS4_*` value in a report before comparing it with a
  current app run.
- **`gguf_spectrum.py`** and **`gguf_to_graph.py`** are GGUF compression-analysis
  tools (singular-value spectrum inspection and factorized-graph export). See
  [`README-analisi.md`](README-analisi.md).
