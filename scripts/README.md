# Scripts

Build and analysis scripts.

- **`embed_kernels.sh`** regenerates
  `Sources/DS4Metal/Runtime/KernelSources.swift` from `metal/*.metal`, embedding
  kernel source code into the binary. It is invoked by `make embed-kernels`.
- **`bench.sh`** runs the prefill/decode knob benchmark matrix with the release
  `DS4Demo` binary and collects the key lines of every run into a single report:
  `scripts/bench.sh <gguf> <prompt-file> [report.txt] [case ...]`.
- **`gguf_spectrum.py`** and **`gguf_to_graph.py`** are GGUF compression-analysis
  tools (singular-value spectrum inspection and factorized-graph export). See
  [`README-analisi.md`](README-analisi.md).
