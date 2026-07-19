**English** | [Italiano](README.it.md)

# DwarfStar Documentation

Index of the technical and operational documentation. Every code and test
folder also has a local `README.md` with responsibilities, dependencies, main
files and modification rules.

## Getting started

- [Project README](../README.md) — overview, quick start and complete
  configuration reference.
- [DOCUMENTAZIONE.md](DOCUMENTAZIONE.md) — broad guide to the app, workflows,
  panels and troubleshooting.
- [STRUTTURA-PROGETTO.md](STRUTTURA-PROGETTO.md) — modules, dependencies and
  folder map.
- [ARCHITETTURE-SUPPORTATE.md](ARCHITETTURE-SUPPORTATE.md) — backend matrix,
  GGUF detection, capabilities and rules for introducing Qwen and GLM 5.2.
- [GUIDA-SVILUPPO.md](GUIDA-SVILUPPO.md) — workflow for modifying code,
  documentation, kernels and the Xcode project.

## Engine and inference

- [`architectures/`](architectures/README.md) — separate documentation for
  DeepSeek V4, Qwen and the progressive GLM 5.2 port.
- [PIPELINE-INFERENZA.md](PIPELINE-INFERENZA.md) — full cycle from prompt to
  token, state ownership, prefill, decode and tool loop.
- [ARCHITETTURA-MOTORE.md](ARCHITETTURA-MOTORE.md) — model details,
  GGUF, tokenizer, NSA, MoE, quantization and graph.
- [BACKEND-METAL.md](BACKEND-METAL.md) — runtime, tensors, command buffers,
  wrappers, generated kernels and numeric validation.
- [DS4CORE-INFERENCE.md](DS4CORE-INFERENCE.md) — compact reference to the
  inference-facing components free of GPU dependencies.
- [`DS4Engine/Inference/FLUSSO-INFERENZA.md`](../Sources/DS4Engine/Inference/FLUSSO-INFERENZA.md)
  — details of the application service and its extensions.

## Configuration and performance

- [Configuration Reference](../README.md#configuration-reference) — the
  authoritative table of GUI settings, `DS4_*`, server, distribution and MCP.
- [CONFIGURAZIONE-E-PROFILI.md](CONFIGURAZIONE-E-PROFILI.md) — precedence,
  read timing, quality/performance profiles and rules for new knobs.
- [VALUTAZIONE-DEMO-PERF.md](VALUTAZIONE-DEMO-PERF.md) — historical demo
  measurements, bottlenecks and A/B runbook.
- [AUTOTUNING-METAL.md](AUTOTUNING-METAL.md) — multi-parameter search with
  coordinate ascent, ABBA comparison, logits gate and resumable checkpoint.
- [METAL-AB-M1-PRO-2026-07-16.md](METAL-AB-M1-PRO-2026-07-16.md) — bit-exact
  and performance results of the new specialized KV staging, packed copy and RoPE.
- [SELF-SPECULATIVE.md](SELF-SPECULATIVE.md) — design and dated measurements
  of self-speculative decode, currently an experimental opt-in.

Files with measurements state date and machine: they are experimental
snapshots, not universal defaults.

## Distribution

- [INFERENZA-DISTRIBUITA.md](INFERENZA-DISTRIBUITA.md) — horizontal and
  vertical topology, setup, files, KV, security and protocol v11.
- [EXPERT_PARALLELISM.md](EXPERT_PARALLELISM.md) — implementation status,
  costs and validation of the vertical split.
- [`Distributed/PROTOCOLLO.md`](../Sources/DS4Engine/Distributed/PROTOCOLLO.md)
  — messages and invariants close to the code.

## GUI, server, tools and data

- [GUI-SERVER-E-API.md](GUI-SERVER-E-API.md) — SwiftUI features, shared
  engine, HTTP/SSE and rules for new endpoints.
- [STRUMENTI-AGENTI-MCP.md](STRUMENTI-AGENTI-MCP.md) — tool registry, agents,
  sub-agents, MCP transports and security.
- [`Chat/FLOW.md`](../Sources/DwarfStar/Features/Chat/FLOW.md) — chat flow
  close to the feature.
- [`Server/HTTP-API.md`](../Sources/DwarfStar/Features/Server/HTTP-API.md) —
  local HTTP contracts and endpoint mapping.
- [`GESTIONE-MODELLI.md`](../Sources/DS4Engine/ModelManagement/GESTIONE-MODELLI.md)
  — download, tokens, sidecars and model lifecycle.
- [`FORMATO-CHECKPOINT.md`](../Sources/DS4Engine/Persistence/KV/FORMATO-CHECKPOINT.md)
  — KV checkpoints and persistent compatibility.
- [`SICUREZZA-PERCORSI.md`](../Sources/DS4Engine/Projects/SICUREZZA-PERCORSI.md)
  and [`Tools/SICUREZZA.md`](../Sources/DS4Engine/Tools/SICUREZZA.md) —
  filesystem boundaries and tool security.

## Testing, release and compliance

- [TESTING-E-VALIDAZIONE.md](TESTING-E-VALIDAZIONE.md) — strategy, commands,
  GPU skips, parity and checklists.
- [`Tests/METAL-TESTS.md`](../Tests/METAL-TESTS.md) — requirements and
  conventions specific to the Metal tests.
- [CRITTOGRAFIA.md](CRITTOGRAFIA.md) — cryptographic inventory, cleartext
  transports and export compliance notes with official sources.
- [UPSTREAM-SYNC.md](UPSTREAM-SYNC.md) — dated snapshot of the comparison
  with the upstream C project and the procedure for refreshing it.
- [`packaging/README.md`](../packaging/README.md) — bundle, signing and
  entitlements.

## Authoritative sources

- Kernels are modified in [`metal/`](../metal/README.md), not in the
  generated Swift file.
- The reference tool-calling template is in
  [`templates/`](../templates/README.md).
- The build and analysis scripts are described in
  [`scripts/`](../scripts/README.md).
- The complete source map starts from [`Sources/README.md`](../Sources/README.md).
- The test map starts from [`Tests/README.md`](../Tests/README.md).

## Maintenance rules

When a behavior changes:

1. update the README of the owning folder;
2. update the corresponding thematic document;
3. keep examples and defaults consistent with the code;
4. state date and hardware for performance measurements;
5. distinguish operational, experimental and design-only features;
6. verify relative links and paths after every move.

Markdown files placed inside targets are automatically excluded by SwiftPM and
by XcodeGen: they can stay next to the code without entering the binary.
