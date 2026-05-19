# DeepSeek-V4-Pro-MacOS — Documentation

Swift + Metal port of DeepSeek-V4 (Pro & Flash variants) for Apple
Silicon, plus a native SwiftUI desktop client that also drives
remote OpenAI-compatible models through OpenRouter.

Where to start depends on what you came here for.

## Reading order

- **Just want to run it.**
  [`../README.md`](../README.md) (English) or
  [`../README.it.md`](../README.it.md) (Italiano) →
  [`USAGE.md`](USAGE.md) for the flag reference and the OpenRouter
  setup walkthrough → [`EXAMPLES.md`](EXAMPLES.md) for ready-made
  recipes.
- **Want to understand the model itself.**
  [`MODEL.md`](MODEL.md) (or [`MODEL.it.md`](MODEL.it.md) in italiano)
  is the canonical reference for the DeepSeek-V4 architecture: MLA,
  MoE, Hyper-Connections, Compressor, Indexer, MTP, RoPE/YaRN, the
  forward pass, the dtypes and the KV cache.
- **Want to understand how the runtime and the app are built around it.**
  [`ARCHITECTURE.md`](ARCHITECTURE.md) for the big picture (engine
  data flow + desktop app + remote backend) →
  [`GLOSSARY.md`](GLOSSARY.md) for the jargon →
  [`DTYPES.md`](DTYPES.md) and [`MEMORY.md`](MEMORY.md) for the
  on-device specifics.
- **Want to modify it.**
  [`MODEL.md`](MODEL.md) → [`ARCHITECTURE.md`](ARCHITECTURE.md) →
  [`MODULES.md`](MODULES.md) for the per-file map →
  [`DEVELOPING.md`](DEVELOPING.md) → [`TESTING.md`](TESTING.md). Dip
  into [`PERFORMANCE.md`](PERFORMANCE.md) when a perf concern lands.

## Document index

| Doc | When to read |
|---|---|
| [`MODEL.md`](MODEL.md) · [`MODEL.it.md`](MODEL.it.md) | **Canonical model reference**. Every component of the DeepSeek-V4 transformer: ModelConfig, MLA, MoE, HyperConnections, Compressor, Indexer, MTP, RoPE/YaRN, the forward pass, dtypes per-component, weight naming, KV cache lifecycle, snapshot/restore. Italian counterpart available. |
| [`ISTRUZIONI.md`](ISTRUZIONI.md) | **Tutorial passo-passo (italiano)** dal Mac vuoto al primo token. Inizia da qui se è la prima volta. |
| [`USAGE.md`](USAGE.md) | Operational reference: CLI flags, GUI walkthrough, OpenRouter onboarding, troubleshooting checklist. |
| [`TOOLS.md`](TOOLS.md) | The native code-agent toolbox (`read / write / edit / shell / apply_patch / webfetch / …`). Categories, statuses, how to add a new one. |
| [`AGENT-MODES.md`](AGENT-MODES.md) | Plan vs Build operating modes, the permission flow that gates dangerous / mutating tools, and where the mode is switched. |
| [`GGUF.md`](GGUF.md) | GGUF reader MVP — supported / unsupported dtypes, what `GGUFFile` can do today, the path to a full inference loader. |
| [`GAP-ANALYSIS-LLAMACPP.md`](GAP-ANALYSIS-LLAMACPP.md) | Structured comparison vs llama.cpp — chat-template dispatcher, tokenizer formats, GGUF support, sampling, what still differs. |
| [`EXAMPLES.md`](EXAMPLES.md) | Recipes: send a message via OpenRouter, register an MCP server, define an agent that delegates, invoke a native tool, dispatch a Metal kernel, load a tensor. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Big picture of the *engine* + *desktop app* around the model: engine data flow, the macOS app's state graph (`InferenceService` / `ModelState` / `ChatStore` / `NativeToolHost`), backend dispatch between local and OpenRouter. Defers to [`MODEL.md`](MODEL.md) for the model internals. **Read this before any source file.** |
| [`GLOSSARY.md`](GLOSSARY.md) | One-stop reference for every domain term in the source (MLA, MoE, FP4-E2M1, E8M0, RoPE, YaRN, DSML, MCP, Plan/Build mode, Skill, Slash command, Permission, …). |
| [`MODULES.md`](MODULES.md) | Per-file reference for `Sources/`. Purpose + public API + dependencies of every Swift file (engine + UI + DeepSeekTools). |
| [`KERNELS.md`](KERNELS.md) | Per-kernel reference for `Sources/DeepSeekKit/Kernels/*.metal`. Inputs, outputs, dispatch shape, function-constant indices. |
| [`DTYPES.md`](DTYPES.md) | Bit layouts of FP8/FP4/E8M0/BF16/F16/F32, conversion math, fusion at convert time. |
| [`MEMORY.md`](MEMORY.md) | mmap strategy, page-cache behavior, KV cache + Compressor state lifecycle, footprint per phase. |
| [`TESTING.md`](TESTING.md) | XCTest targets — what each verifies, what's NOT yet covered, how to add a new one, tolerance choices. |
| [`DEVELOPING.md`](DEVELOPING.md) | Contributor guide: setup, conventions, recipes (add a kernel / CLI flag / remote backend / MCP transport / native tool / Settings tab), common pitfalls. |
| [`PERFORMANCE.md`](PERFORMANCE.md) | Where time goes today, profiling instructions, planned optimisations with effort estimates. |
| [`PYTHON-MAPPING.md`](PYTHON-MAPPING.md) | Cross-walk: Swift file → `Reference/inference/{model,kernel,convert}.py` line ranges. |
| [`ROADMAP.md`](ROADMAP.md) | What's implemented (engine + desktop app + tool subsystem), what's stubbed, what's deferred and why. |
| [`GAP-ANALYSIS-OPENCODE.md`](GAP-ANALYSIS-OPENCODE.md) | Structured comparison vs opencode — what each project does, missing pieces, possible directions. |
| [`../TODO.md`](../TODO.md) | Living checklist of outstanding work, grouped by area (quantisation / parity / runtime / encoding / desktop app / perf / tooling / docs / testing). |

## Repository layout

```
DeepSeek-V4-Pro-MacOS/
├── Package.swift                  SwiftPM manifest; MetalLibPlugin wired in here
├── project.yml                    XcodeGen spec for the GUI app target
├── Plugins/MetalLibPlugin/        Build-tool plugin that compiles .metal → default.metallib
├── Sources/
│   ├── DeepSeekKit/               Library: all model + tensor + IO logic
│   │   ├── *.swift                Top-level types (Config, Tensor, Device, KV snapshot, …)
│   │   ├── Encoding/              Chat encoder / decoder (port of encoding_dsv4.py)
│   │   ├── Kernels/               Metal shaders (.metal)
│   │   └── Layers/                Swift wrappers around the kernels + composition
│   ├── DeepSeekUI/                SwiftUI desktop app
│   │   ├── State/                 InferenceService, ChatStore, ModelState,
│   │   │                          AgentLibrary, MCPClientPool, OpenRouter*, …
│   │   ├── Views/                 Chat surface, Settings tabs, sheets, pickers
│   │   ├── Utility/               AppSettings, PersistencePaths, MarkdownText
│   │   └── Resources/             Info.plist, Assets.xcassets
│   ├── deepseek/                  CLI: local token generation
│   └── converter/                 CLI: HF safetensors → on-disk variants (BF16/INT8/INT4/INT2)
├── Tests/DeepSeekKitTests/        XCTest target; one *Tests.swift per kernel/module
├── Reference/                     Upstream Python (read-only, source of truth)
│   ├── inference/                 model.py, kernel.py, generate.py, convert.py
│   └── encoding/                  encoding_dsv4.py + golden tests
└── docs/                          (you are here)
```

## Two backends, one chat surface

The desktop app accepts both **local** (Metal) and **remote**
(OpenRouter HTTP/SSE) endpoints through the same chat. Switching
between them is a toolbar menu action; conversations don't have to
care which one they're using, but a few features only work on one
side:

| | Local | OpenRouter |
|---|---|---|
| Token-level KV cache + fast-delta path | ✅ | n/a (provider-side) |
| DSV4 chat template | ✅ (default) | n/a (provider builds the prompt) |
| Jinja2 chat templates (Llama / Mistral / Qwen / …) | ✅ via dispatcher | n/a |
| GGUF reader (header + pass-through dtypes) | ✅ MVP — see [`GGUF.md`](GGUF.md) | n/a |
| Extended sampler suite (min-p / mirostat / tail-free / typical / freq / presence) | ✅ | partial — what OpenRouter accepts in the body |
| Native tools (DeepSeekTools) | ✅ (scaffolded; wiring tracked in TODO §8) | ✅ (same) |
| Tool calls via MCP | ✅ | ✅ |
| Plan / Build agent mode + permission gate | ✅ | ✅ |
| Sub-agent delegation | ✅ | not yet |
| KV snapshot/restore around delegation | ✅ | n/a |
| Reasoning content (`<think>` / `reasoning_content`) | ✅ | ✅ for R1 / o-series |
| Projects (pre-tokenised codebase splice) | ✅ | n/a |
| Crash recovery via `pendingTurn` | ✅ | ❌ (resend the message) |
| Cost banner | n/a | ✅ |

See [`ARCHITECTURE.md`](ARCHITECTURE.md#desktop-app-architecture)
for how `ChatStore.send` decides which path to take, and
[`USAGE.md`](USAGE.md#remote-models-openrouter) for the OpenRouter
onboarding flow.

The native code-agent tool subsystem is documented standalone in
[`TOOLS.md`](TOOLS.md); the operating-mode gate that filters
mutating/dangerous tools is in [`AGENT-MODES.md`](AGENT-MODES.md).

## License

MIT, mirroring the upstream model release (`Reference/LICENSE`).
