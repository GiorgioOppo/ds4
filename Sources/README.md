# Sources

All Swift code, split by **target**. The main dependency pipeline is:

```text
DS4Core -> DS4Metal -> DS4Engine -> DwarfStar
```

| Target | Type | Role |
|---|---|---|
| `DS4Core/` | library | Pure Swift core: GGUF mmap, tokenizer, sampler, model shape, chat/tool format. No Metal dependency. |
| `DS4Metal/` | library | Metal runtime, decode graph, KV cache, and GPU kernels; this is the Swift/Metal port of `ds4_metal.m`. |
| `DS4Engine/` | library | `InferenceService` actor, tools/agents, disk KV, sub-agents, model download, and distributed inference. |
| `DwarfStar/` | app | SwiftUI macOS GUI: chat, agents, projects, server, benchmark, diagnostics, distributed mode, and settings. |
| `DS4Demo/` | CLI | Minimal command-line demo for Metal bring-up and GGUF streaming/generation. |

Swift does not use folders as module boundaries. Inside a target, subdirectories
are only organizational; imports do not change. Boundaries between targets are
real and are declared through `dependencies` in `Package.swift`.
