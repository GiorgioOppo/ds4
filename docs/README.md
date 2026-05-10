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
| [`USAGE.md`](USAGE.md) | Prerequisites, build, convert, run, resume after crash, troubleshooting checklist. |
| [`MODULES.md`](MODULES.md) | Per-file reference for `Sources/`. Purpose + public API + dependencies of every Swift file. Use as a directory index. |
| [`KERNELS.md`](KERNELS.md) | Per-kernel reference for `Sources/DeepSeekKit/Kernels/*.metal`. Inputs, outputs, dispatch shape. |
| [`PYTHON-MAPPING.md`](PYTHON-MAPPING.md) | Cross-walk: Swift file → `Reference/inference/{model,kernel,convert}.py` line ranges. Useful when reading either side. |
| [`ROADMAP.md`](ROADMAP.md) | What's implemented, what's stubbed, what's deferred and why. Limitations and known issues. |

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
