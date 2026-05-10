# DeepSeek-V4-Pro-MacOS — Documentation

Swift + Metal port of DeepSeek-V4 (Pro & Flash variants) inference for
Apple Silicon. The model is a 1.6T-param (Pro) / 284B-param (Flash) MoE
with FP4 + FP8 mixed-precision weights. This project converts the
HuggingFace checkpoint into a Mac-friendly format, then runs token
generation natively on Metal.

## Quick start

```bash
# 1. Build (the first time compiles all 23 Metal kernels into default.metallib).
swift build -c release

# 2. Convert a HuggingFace checkpoint to BF16 sharded.
.build/release/converter \
  --hf-ckpt-path /Volumes/DATA/V4-Flash-HF \
  --save-path   /Volumes/DATA/V4-Flash-bf16 \
  --n-experts 256        # from config.json

# 3. Run inference.
.build/release/deepseek /Volumes/DATA/V4-Flash-bf16 \
  "Ciao" --mode chat --max-tokens 50
```

Full setup, troubleshooting, and resume after a crash: [`USAGE.md`](USAGE.md).

## Document index

Open these in order if you're new to the project; pick directly if you
already know what you need.

| Doc | When to read |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Big picture: data flow, module dependency graph, memory model, where Metal kernels sit vs Swift wrappers. **Read this before any source file.** |
| [`GLOSSARY.md`](GLOSSARY.md) | One-stop reference for every domain term used in the source (MLA, MoE, HC, FP4-E2M1, E8M0, RoPE, YaRN, DSML, …). |
| [`USAGE.md`](USAGE.md) | Prerequisites, build, convert, run, resume after crash, troubleshooting checklist. |
| [`EXAMPLES.md`](EXAMPLES.md) | Code recipes: load a tensor, dispatch a kernel, add a new Layer, generate a token, parse a chat completion. |
| [`MODULES.md`](MODULES.md) | Per-file reference for `Sources/`. Purpose + public API + dependencies of every Swift file. Use as a directory index. |
| [`KERNELS.md`](KERNELS.md) | Per-kernel reference for `Sources/DeepSeekKit/Kernels/*.metal`. Inputs, outputs, dispatch shape, function-constant indices. |
| [`DTYPES.md`](DTYPES.md) | Bit layouts of FP8/FP4/E8M0/BF16/F16/F32, conversion math, fusion at convert time. |
| [`MEMORY.md`](MEMORY.md) | mmap strategy, page-cache behavior, KV cache + Compressor state lifecycle, footprint per phase. |
| [`TESTING.md`](TESTING.md) | The 15 XCTest files, what each verifies, how to add a new one, tolerance choices. |
| [`DEVELOPING.md`](DEVELOPING.md) | Contributor guide: setup, conventions, recipes to add kernel/layer/CLI flag, common pitfalls we hit. |
| [`PERFORMANCE.md`](PERFORMANCE.md) | Where time goes today, profiling instructions, planned optimizations with effort estimates. |
| [`PYTHON-MAPPING.md`](PYTHON-MAPPING.md) | Cross-walk: Swift file → `Reference/inference/{model,kernel,convert}.py` line ranges. |
| [`ROADMAP.md`](ROADMAP.md) | What's implemented, what's stubbed, what's deferred and why. |

## Reading order

Three paths depending on why you're here:

- **Operativo (just want to run it)**: [README](README.md) →
  [USAGE](USAGE.md) → [EXAMPLES](EXAMPLES.md) →
  [ROADMAP](ROADMAP.md#known-limitations) for the gotchas.
- **Architetturale (want to understand it)**: [README](README.md) →
  [ARCHITECTURE](ARCHITECTURE.md) → [GLOSSARY](GLOSSARY.md) →
  [DTYPES](DTYPES.md) → [MEMORY](MEMORY.md).
- **Contributore (want to modify it)**: [README](README.md) →
  [ARCHITECTURE](ARCHITECTURE.md) → [MODULES](MODULES.md) →
  [DEVELOPING](DEVELOPING.md) → [TESTING](TESTING.md). When you hit a
  perf concern, dip into [PERFORMANCE](PERFORMANCE.md).

## Repository layout

```
DeepSeek-V4-Pro-MacOS/
├── Package.swift                  SwiftPM manifest, MetalLibPlugin wired in here
├── Plugins/MetalLibPlugin/        Build-tool plugin that compiles .metal → default.metallib
├── Sources/
│   ├── DeepSeekKit/               Library: all model + tensor + IO logic
│   │   ├── *.swift                Top-level types (Config, Tensor, Device, ...)
│   │   ├── Encoding/              Chat encoder (port of encoding_dsv4.py)
│   │   ├── Kernels/               Metal shaders (.metal)
│   │   └── Layers/                Swift wrappers around the kernels + composition
│   ├── deepseek/                  CLI: token generation
│   └── converter/                 CLI: HF safetensors → Mac-friendly format
├── Tests/DeepSeekKitTests/        XCTest target; one *Tests.swift per kernel/module
├── Reference/                     Upstream Python (read-only, source of truth)
│   ├── inference/                 model.py, kernel.py, generate.py, convert.py
│   └── encoding/                  encoding_dsv4.py + golden tests
└── docs/                          (you are here)
```

## License

MIT, mirroring the upstream model release (`Reference/LICENSE`).
