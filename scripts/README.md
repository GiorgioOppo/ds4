**English** | [Italiano](README.it.md)

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
- **`metal_ab.sh`** validates one Metal runtime knob in two separate release
  processes. It fixes greedy sampling, disables speculative decode and usage
  persistence, compares generated token ids and bounded full-vocabulary logits,
  and reports prefill/decode speedup. The trace is retained in memory with
  copy-on-write during the timed region and serialized only afterwards, so trace
  I/O is not counted in throughput. Usage:
  `scripts/metal_ab.sh <gguf> <prompt-file> <DS4_KNOB> [base] [candidate] [max-new] [out-dir]`.
  `metal_ab_compare.py` is its standard-library, memory-mapped analyzer; run
  `python3 scripts/metal_ab_compare.py --self-test` without a model to verify it.
  The default numerical gate is `atol=1e-4, rtol=1e-4`; set
  `DS4_AB_ATOL=0 DS4_AB_RTOL=0` when a change promises bit parity. One pair is
  exploratory: repeat with `DS4_AB_ORDER=candidate-first` before promoting a
  knob, because cache warmth, thermal state and memory pressure can favor
  either process. The
  runner respects an existing `DEVELOPER_DIR`; otherwise it automatically uses
  `/Applications/Xcode.app/Contents/Developer` when the active
  `xcode-select` points at an incompatible standalone CommandLineTools install.
- **`metal_autotune.py`** searches a combined configuration with multi-pass
  coordinate ascent. It sweeps small non-monotonic hardware grids, walks
  ordered values in both directions while performance grows, freezes one
  usage-imatrix seed per process, rejects RAM/swap-contaminated runs, requires
  `PASS_EXACT` for lossless knobs, confirms finalists in ABBA order, and writes
  a crash-safe resumable report plus `final-env.sh`. The standard profile tunes
  the safe decode/I/O knobs; prefill knobs require a separate prompt of at
  least 1024–2048 tokens. It deliberately excludes quantization and
  reduced-expert changes. See
  [`docs/AUTOTUNING-METAL.md`](../docs/AUTOTUNING-METAL.md). Run the model-free
  tests with `python3 scripts/metal_autotune.py --self-test`.
- **`gguf_spectrum.py`** and **`gguf_to_graph.py`** are GGUF compression-analysis
  tools (singular-value spectrum inspection and factorized-graph export). See
  [`README-analisi.md`](README-analisi.md).
