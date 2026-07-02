# DwarfStar — DeepSeek-V4 on macOS

DwarfStar is a native Swift / SwiftUI application for running
**DeepSeek-V4-Flash (284B MoE)** on Apple Silicon with a **pure-Swift Metal
inference engine**. The engine is a faithful port of upstream `ds4.c` /
`ds4_metal.m`: no C runtime engine, no prebuilt static library, and no external
process for normal inference. The 2-bit GGUF runs on a 16 GB MacBook by
streaming routed expert weights from SSD.

> Detailed documentation:
> [`docs/DOCUMENTAZIONE.md`](docs/DOCUMENTAZIONE.md) for usage, GUI, tools,
> agents, server, distributed inference, and troubleshooting ·
> [`docs/ARCHITETTURA-MOTORE.md`](docs/ARCHITETTURA-MOTORE.md) for engine
> internals · [`docs/CRITTOGRAFIA.md`](docs/CRITTOGRAFIA.md) for encryption and
> App Store export compliance.

## Architecture

```text
DwarfStar (SwiftUI)            <- chat · agents · projects · tuning · server ·
        |                         distributed worker · benchmark · diagnostics
   DS4Engine (Swift)           <- InferenceService actor, event streams,
        |                         DSML tools, agents, ProjectCache,
        |                         HTTP server, distributed coordinator/worker
   DS4Core + DS4Metal (Swift)  <- GGUF mmap, tokenizer, sampler, chat rendering,
        |                         Metal runtime, decode graph, expert cache
   metal/*.metal               <- kernel source of truth, embedded at build time
```

Correctness is the primary project rule. The Swift engine is validated by the
tests in `Tests/DS4CoreTests/`, with many kernels and graph stages checked
against the upstream behavior.

## Engine Facts

- **SSD streaming is the only model-load path.** Non-routed weights are no-copy
  `mmap` views backed by the OS page cache. On each token, only the routed
  experts selected for the current layer are gathered. The full model never has
  to fit in RAM.
- **Metal kernels are embedded in the binary.** `make embed-kernels` regenerates
  `Sources/DS4Metal/Runtime/KernelSources.swift` from `metal/*.metal`, so a
  packaged app does not need an on-disk `metal/` folder at runtime.
- **Fused MoE kernels cover the deployed expert formats.** Pair-SwiGLU and
  down-sum6 paths cover q4_K, q2_K, and iq2_xxs experts.
- **Expert slot-cache is optional and usage-driven.** A per-layer LRU cache can
  keep hot experts resident on GPU. Routing-frequency statistics are persisted
  per model and per agent, then used to pre-warm and redistribute cache slots.
- **Multi-turn KV reuse is append-only.** `InferenceService` tracks exact
  committed token ids. Each new turn prefills only the suffix that is not
  already in the KV. If generation is interrupted, the next turn rebuilds from
  committed ids because the NSA compressor is recurrent and cannot rewind.
- **Tool calling uses native DSML.** The model's `｜DSML｜` control token opens an
  XML-style call format rendered from the GGUF chat template. Tool results are
  returned as `<tool_result>...</tool_result>` inside a user turn.
- **Layer-major prefill amortizes I/O.** Prefill runs in chunks: each layer's
  weights are loaded once per chunk and applied to all prompt tokens. The routed
  FFN phase gathers the union of experts for that chunk rather than 6 experts
  per token.

## The App

| Tab | Purpose |
|---|---|
| **Chat** | Streaming markdown chat, collapsible reasoning, live tool calls, multi-turn KV reuse, text-file attachments, active project menu, near-context-full warnings, Local or Distributed mode. |
| **Settings** | Shared model path, context size, memory knobs, local model load, and distributed coordinator route. |
| **Agents** | Role editor: system prompt, icon, tools, JSON import/export, and per-agent expert-usage profile. |
| **Project** | Library of imported folders with sandbox bookmarks. Project tools explore the active project without consuming chat context until a tool reads content. |
| **Tuning** | Expert cache slots, hit-rate, per-layer routing concentration, and the usage imatrix. |
| **Server** | Native in-process OpenAI/Anthropic-compatible HTTP server. |
| **Worker** | Runs this Mac as a distributed worker that owns a contiguous layer slice. |
| **Benchmark** | Native throughput benchmark for prefill and generation across growing context sizes, local or distributed. |
| **Diagnostics** | Native tokenizer dump and chat-template/tool-format inspection. |

## Built-In Tools and Agents

Built-in DSML tools live one per file under
`Sources/DS4Engine/Tools/Builtins/`.

- **Project index tools:** `project_list`, `project_read`, `project_search`,
  `project_write`, `project_edit`.
- **Raw project-root file tools:** `file_read`, `file_lines`, `file_write`,
  `file_add`, `file_modify`.
- **Utilities:** `git` (local whitelist, no network), `calculator`,
  `add`, `subtract`, `multiply`, `now`.
- **Sub-agents:** `agents_list`, `subagent_search`,
  `subagent_run(target, question, agent?, tools?)`. Sub-agents run in an
  isolated context with their own content-keyed KV prefix cache. The main chat
  receives only the delegated question and returned answer, not the sub-agent's
  internal work.

Default agents are **General**, **Coding**, **Code**, **Orchestrator**,
**Math**, **Writing**, **LaTeX**, and **Documentation**. Each agent is a system
prompt plus a tool allow-list plus a dedicated expert-usage profile.

## Quick Start

```sh
make                  # swift build
make xcodeproj        # regenerate DwarfStar.xcodeproj after adding files
swift run DwarfStar   # launch the app
make test             # unit tests
```

In the app, choose a GGUF from **Settings** with **Browse**. With App Sandbox
enabled, typed paths are not enough; the file picker grants a security-scoped
bookmark that persists across launches. Press **Load Model**, then open **Chat**.
The **Thinking** toggle enables reasoning-token handling; **Stop** cancels
generation and the next turn rebuilds KV if needed.

## HTTP Server

The Server tab starts an OpenAI/Anthropic-compatible server on
`Network.framework`. It is native and in-process: no subprocess, and GGUF weights
are mmap-shared with the chat engine. KV cache and GPU scratch are separate, so
chat and server requests still compete for compute and memory bandwidth.

| Method | Path | API |
|---|---|---|
| GET | `/v1/models`, `/v1/models/{id}` | OpenAI |
| POST | `/v1/chat/completions` | OpenAI Chat Completions, stream and non-stream |
| POST | `/v1/responses` | OpenAI Responses, stream and non-stream |
| POST | `/v1/completions` | OpenAI legacy completions |
| POST | `/v1/messages` | Anthropic Messages |

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","stream":true,
       "messages":[{"role":"user","content":"Hello"}]}'
```

## Distributed Inference

Distributed mode implements pipeline parallelism by contiguous layer ranges,
modelled on `ds4_distributed.c`.

- Each **worker** owns a layer slice, its weights, and the KV shard for that
  slice only.
- The **coordinator** owns embedding, sampling, prompt rendering, and tool
  execution.
- The HC hidden state (`nHC x nEmbd` floats, transported at 32/16/8 bit) flows
  through workers for every token.
- Workers must be started first. The coordinator route must cover all model
  layers contiguously: Flash has 43 layers, Pro has 61.

The per-node benefit is reduced expert I/O: each worker streams roughly `1/N` of
the routed experts, making the hot working set more likely to stay in RAM.
Prefill runs in chunks, default 32 tokens per distributed frame. Optional
worker-to-worker forwarding can reduce network hops, but requires the
coordinator's LAN return address.

## Repository Layout

```text
Makefile / Package.swift / project.yml
Sources/
  DS4Core/        GGUF parser, tokenizer, sampler, model shape,
                  chat/tool rendering, DSML parser
  DS4Metal/       Metal runtime, kernels, decode graph, StreamingDecoder,
                  expert slot-cache, no-copy GGUF weight loaders
  DS4Engine/      InferenceService, disk KV cache, tools, agents,
                  model downloader, distributed runtime
  DS4Demo/        CLI demo: Metal bring-up, GGUF audit, token streaming
  DwarfStar/      SwiftUI app, one feature folder per tab
metal/            Metal kernel source of truth
templates/        commented Jinja rewrite of the model chat template
scripts/          GGUF analysis tools and kernel embedding helpers
docs/             detailed English documentation
packaging/        .app bundle assembly and signing inputs
Tests/            unit and parity tests
```

Every major folder has a local `README.md` explaining ownership and how it
connects to the rest of the system.

## CLI Demo

```sh
swift run DS4Demo
swift run DS4Demo <model.gguf> 4
swift run DS4Demo <model.gguf> 32 "Explain RoPE briefly"
```

Full syntax:

```sh
swift run DS4Demo [gguf-path] [maxNew] [prompt]
```

| Argument | Default | Meaning |
|---|---|---|
| `gguf-path` | none | GGUF file to open. If omitted, the demo only brings up Metal and runs the GPU self-test. |
| `maxNew` | `4` | Number of tokens to generate. `0` runs only the one-token forward smoke test. |
| `prompt` | `"ciao come stai? rispondi in 1 parola"` | User prompt rendered through the model chat template. Because arguments are positional, pass `maxNew` before passing a prompt. |

Advanced demo and engine knobs are documented in
[`Sources/DS4Demo/README.md`](Sources/DS4Demo/README.md). Common ones:
`DS4_TYPES_ONLY=1`, `DS4_DIAG=1`, `DS4_WARMUP=N`,
`DS4_USAGE_FILE=<path|off>`, `DS4_ACTIVE_EXPERTS=1...6`,
`DS4_EXPERT_CACHE_SLOTS=N`, `DS4_EXPERT_PREAD=1`,
`DS4_RAW_RING=1`, `DS4_RESIDENT_DENSE=1`, `DS4_Q8_NSG=1...8`,
`DS4_PREFETCH=1`, `DS4_PREFETCH_EXPERTS=N`, `DS4_FUSED_MOE=0`,
and `DS4_PROFILE_ROUTE=1`.

## Packaging a `.app`

```sh
make app          # -> build/DwarfStar.app, release, ad-hoc signed
open build/DwarfStar.app
```

For distribution, sign with a Developer ID and notarize. See
`packaging/make_app.sh`. The sandbox entitlements include `network.client`,
`network.server`, and user-selected file access with app-scope bookmarks for
models and projects.

## Status

Working and verified on a MacBook Pro M1 Pro 16 GB:

- model load and streaming chat on the 2-bit GGUF;
- thinking/reasoning handling;
- multi-turn KV reuse;
- text-file attachments;
- DSML tool calling with built-ins;
- agents and per-agent expert profiles;
- project library;
- tuning panel and expert cache controls;
- disk-KV cache, default on;
- native HTTP server, verified for OpenAI `chat/completions` and Anthropic
  `messages` streaming.

Implemented, still requiring broader on-device validation:

- `/v1/responses`;
- isolated-context sub-agents with main-KV snapshot/restore and per-file/project
  content-keyed KV cache;
- newer default agents: Orchestrator, LaTeX, Documentation;
- distributed inference protocol, worker/coordinator, UI, and benchmark;
- numerical parity and multi-Mac distributed runs.

## Performance Knobs

**Selected-expert read-ahead (`DS4_WILLNEED_EXPERTS`, default ON).**
The engine calls `madvise(WILLNEED)` on the experts selected by the router just
before gather. This is not speculative: it asks the OS to read exactly the slabs
that will be copied. It reduces cold page faults, is a no-op when hot, and does
not change numerics.

**Resident dense weights (`DS4_RESIDENT_DENSE`).**
Copies roughly 5 GB of non-expert per-layer weights into wired buffers instead
of leaving them as evictable mmap views. This helps when expert streaming evicts
hot dense weights from the page cache, causing `route/attn`, embedding, and head
weights to re-fault from SSD every token. It can improve throughput on 24/32 GB
machines, but can hurt on 16 GB by increasing memory pressure.

**Q8 matvec tuning (`DS4_Q8_NSG`, default `4`).**
Controls simdgroups per threadgroup for dense Q8_0 matvecs. It does not change
results; it changes occupancy and latency hiding. Sweep `2/4/6/8` on the target
Mac when tuning.

**Experimental opt-ins.**
`DS4_RAW_RING` keeps raw KV in an `nSWA` ring, reducing raw-KV memory for long
contexts. `DS4_PREFETCH` and `DS4_PREFETCH_EXPERTS` add speculative read-ahead
for the next layer and likely experts. Validate these with parity tests and
on-device profiling before relying on them.

## Known Limits

On 16 GB systems, decode is I/O-bound. That is the physics of streaming a 284B
MoE model from SSD; distributed inference is the intended mitigation. KV memory
still scales with context, so keep context modest on low-RAM systems. The
raw-KV ring reduces only the raw cache, not every compressed KV row. Server,
distributed inference, diagnostics, and benchmark are native in-process panels;
no subprocess-driven UI panels remain.
