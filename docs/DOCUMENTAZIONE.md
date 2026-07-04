# DwarfStar Documentation

This is the user-facing and developer-facing guide for DwarfStar: what the app
does, how the pure-Swift DS4 engine is organized, how to operate the GUI and CLI,
and how the advanced panels fit together.

DwarfStar is both:

- a macOS SwiftUI application for local and distributed DeepSeek-V4 inference;
- a pure-Swift Metal port of the DS4 engine, used by the app, the CLI demo, the
  native HTTP server, the benchmark panel, diagnostics, and distributed workers.

For lower-level engine details, see
[`ARCHITETTURA-MOTORE.md`](ARCHITETTURA-MOTORE.md). For App Store export
compliance, see [`CRITTOGRAFIA.md`](CRITTOGRAFIA.md).

## Contents

1. [Overview](#1-overview)
2. [Layered Architecture](#2-layered-architecture)
3. [Core Engine Concepts](#3-core-engine-concepts)
4. [CLI Demo: `DS4Demo`](#4-cli-demo-ds4demo)
5. [SwiftUI App: `DwarfStar`](#5-swiftui-app-dwarfstar)
6. [User Workflow](#6-user-workflow)
7. [Automatic Configuration and Hardware Presets](#7-automatic-configuration-and-hardware-presets)
8. [Memory, Streaming, and GUI Defaults](#8-memory-streaming-and-gui-defaults)
9. [Advanced Panels](#9-advanced-panels)
10. [Build, Run, and Package](#10-build-run-and-package)
11. [Troubleshooting](#11-troubleshooting)
12. [Glossary](#12-glossary)

## 1. Overview

DwarfStar runs DeepSeek-V4-Flash locally on Apple Silicon by streaming a 284B MoE
model from SSD. Only a small routed subset of the model is touched for each
token. The project avoids the old subprocess approach: the SwiftUI app talks
directly to the Swift engine through actors and async streams.

The same engine implementation powers:

- the interactive Chat tab;
- tool calling and isolated sub-agents;
- the native OpenAI/Anthropic-compatible HTTP server;
- the distributed coordinator and worker runtime;
- benchmark and diagnostics panels;
- the `DS4Demo` command-line bring-up and profiling executable.

The important design constraint is that the app must behave like the CLI demo:
same model parser, same tokenizer, same decode graph, same sampling logic, same
streaming assumptions. The GUI adds state management and convenience, not a
second inference implementation. In local GUI mode, Chat, Server, and Benchmark
reuse the single `InferenceService` loaded in Settings so the app does not
double resident Q4 buffers, `mlock`ed buffers, or GPU scratch on low-RAM Macs.

## 2. Layered Architecture

```text
DwarfStar (SwiftUI)
  Chat · Settings · Agents · Project · Tuning · Server · Worker · Benchmark · Diagnostics
        |
DS4Engine
  InferenceService actor · HTTP server · tools · agents · ProjectCache
  DiskKVStore · distributed coordinator/worker · model downloader
        |
DS4Core + DS4Metal
  GGUF mmap parser · tokenizer · sampler · model shape · DSML rendering/parser
  Metal runtime · GPUTensor · graph context · kernels · StreamingDecoder
        |
metal/*.metal
  Source of truth for kernels, embedded into Swift at build time
```

### Source Map

| Path | Responsibility |
|---|---|
| `Sources/DS4Core/` | Pure Swift, no Metal dependency: GGUF parser, tokenizer, sampler, model shape, chat rendering, DSML parsing. |
| `Sources/DS4Metal/` | Metal runtime, tensors, kernel wrappers, decode graph, streaming decoder, expert slot-cache, no-copy weight loading. |
| `Sources/DS4Engine/` | Application-level service layer: inference actor, tool registry, disk KV, downloader, distributed runtime. |
| `Sources/DS4Demo/` | CLI demo and diagnostic executable for bring-up, GGUF audit, and token streaming. |
| `Sources/DwarfStar/` | SwiftUI application, organized one folder per tab or feature area. |
| `metal/` | Editable Metal kernel source. `make embed-kernels` regenerates the Swift embedded version. |
| `templates/` | Commented copy of the model chat template for review and documentation. |
| `scripts/` | GGUF analysis helpers and kernel embedding scripts. |
| `docs/` | Detailed project documentation. |
| `Tests/` | Unit, kernel, graph, parser, and service tests. |

## 3. Core Engine Concepts

### SSD Streaming

DwarfStar uses SSD streaming as the default and only model-load model. Non-routed
weights are `mmap` views and can be kept hot by the OS page cache. Routed experts
are gathered per token: only the selected experts for the current layer are
copied or read into the GPU path.

The point is not to make a 284B model "fit" in RAM. The point is to keep the
active working set small enough that the machine can progress token by token.

### MoE and Routed Experts

DeepSeek-V4-Flash is a Mixture-of-Experts model. The router selects a top-k set
of experts for each token and layer. The default active count is 6. Reducing the
active expert count with `DS4_ACTIVE_EXPERTS` lowers I/O but changes model
quality and should be treated as a degraded mode or diagnostic experiment.

### KV Reuse

Conversations are append-only. The service tracks exactly which token ids have
already been committed to the decoder KV. A new user turn only prefills the new
suffix. If a generation is interrupted, the service can rebuild the KV from
committed ids before continuing, because all committed text is known exactly.

### Tool Calling

The model uses DSML, an XML-style tool-call format opened by the `｜DSML｜`
control token. DwarfStar renders tool declarations through the same template
surface the model was trained on and parses generated calls with `ToolCallParser`.

Built-in tools run automatically. Non-built-in calls can be surfaced for manual
results. Sub-agent calls are intercepted by the engine and run in an isolated
decoder context.

### Expert Usage Imatrix

The engine records which experts the router selected. This usage profile is
persisted per model and per agent. It is used to pre-warm expert cache slots and,
when usage-driven allocation is enabled, to allocate more slots to layers whose
routing is more concentrated.

## 4. CLI Demo: `DS4Demo`

`DS4Demo` is a minimal engine harness. It intentionally avoids the GUI and is
useful when you want a clean measurement path.

### Usage

```sh
swift run DS4Demo
swift run DS4Demo /path/model.gguf 4
swift run DS4Demo /path/model.gguf 32 "Explain RoPE briefly"
```

The first form only starts Metal and runs the self-test. The second and third
forms open a GGUF, run one forward smoke test, and generate tokens if `maxNew` is
greater than zero.

### Step-by-Step Behavior

1. Create `MetalRuntime`, compiling embedded kernels from `KernelSources.swift`.
2. Run the GPU self-test.
3. If no GGUF path was passed, exit.
4. Open the GGUF with no-copy mmap and no CPU prefetch.
5. Optionally run `DS4_DIAG` disk and MTP diagnostics.
6. Optionally run `DS4_TYPES_ONLY` dtype/tokenizer audit and exit.
7. Detect routed MoE quantization and mixed-precision layers.
8. Create `StreamingDecoder.fromGGUFExpertCachedMapped`.
9. Load or initialize the usage imatrix.
10. Run a one-token forward pass and verify finite logits.
11. If `maxNew > 0`, tokenize the prompt with the model chat template.
12. Prefill prompt tokens layer-major.
13. Greedy-decode tokens, stream bytes to `stdout`, and log timings to `stderr`.
14. Save usage imatrix and print the decode profile.

See `Sources/DS4Demo/README.md` for the full list of environment variables and
diagnostic examples.

## 5. SwiftUI App: `DwarfStar`

The app is a stateful wrapper around the same engine. It adds model selection,
security-scoped bookmarks, persistent chats, tools, agents, project indexing,
server mode, distributed mode, diagnostics, and tuning panels.

### App Tabs

| Tab | Role |
|---|---|
| **Chat** | Main local or distributed conversation. |
| **Settings** | Shared model, context, execution mode, memory/I/O knobs, local load, distributed route. |
| **Agents** | Editable roles with prompts, icons, and tool allow-lists. |
| **Project** | Imported project library and active project cache. |
| **Tuning** | Expert slot-cache controls and usage imatrix. |
| **Server** | OpenAI/Anthropic-compatible native HTTP server. |
| **Worker** | Run this Mac as a distributed layer worker. |
| **Benchmark** | Native throughput charting across context frontiers. |
| **Diagnostics** | Tokenizer dump and chat-template/tool-format inspection. |

### Chat State: `ChatStore`

`ChatStore` is the main local chat view-model. It owns or proxies:

- shared app settings (`modelPath`, `contextSize`);
- local `InferenceService`;
- persistent chat sessions;
- staged text-file attachments;
- selected agent and tool settings;
- sampling settings;
- memory knobs such as expert bundle, expert pread, dense streaming, Q4
  attention projections, `mlock`, disk KV, raw-KV ring, and expert cache slots;
- tuning information and expert usage controls;
- live generation status and stream consumption.

Persistent chats are stored as JSON files under Application Support. A reopened
chat is visible immediately, but the engine KV may not contain that conversation
yet. On the next send, `ChatStore` re-primes from visible history; disk KV can
restore already-known prefixes.

### Event Streaming

`InferenceService` returns async events: status, reasoning tokens, visible text,
tool-stream text, completed tool calls, and errors. `ChatStore` mirrors these
events into `UIMessage` rows. Tool calls can trigger automatic tool execution,
manual result collection, or sub-agent execution.

## 6. User Workflow

### Step 1 — Select and Configure the Model

Open **Settings** and choose a GGUF with **Browse**. In sandboxed builds, this is
required because the picker grants a security-scoped bookmark. The bookmark is
stored so the same model can reopen on the next launch.

Select a context size. The default is RAM-aware: low-RAM systems default to a
small context because KV memory grows with context and competes with the page
cache needed for expert streaming.

Optional memory and I/O settings:

- **Expert cache slots:** GPU-resident expert LRU budget per layer.
- **Experts via direct pread:** bypass page cache for expert slabs.
- **Expert bundle sidecar:** store each expert's gate/up/down slabs contiguously
  so a cache miss is one sequential read. The sidecar duplicates the expert
  region on disk; sandboxed builds create it under Application Support when they
  cannot write next to the GGUF.
- **Dense-weight streaming:** stream dense layer weights through a small staging
  ring one layer ahead of compute. This is the preferred low-RAM alternative to
  keeping all dense weights resident.
- **Pin hot buffers in RAM:** best-effort `mlock()` for hot resident buffers so
  the memory compressor does not turn every token into a slow reread.
- **Q4 attention projections:** lossy speed path that requantizes the largest
  attention projections to Q4_K and caches them under Application Support.
- **Disk KV:** checkpoint KV prefixes to disk for session/server reuse.
- **Raw-KV ring:** keep raw sliding-window KV constant-size.

### Step 2 — Load

Press **Load Model**. The app opens the GGUF, initializes Metal, builds the
decoder, applies memory/I/O environment knobs, configures disk KV, applies the
selected agent, and switches to ready state. Load time depends on model size and
whether one-time sidecars or caches must be created, especially the expert
bundle and Q4 dense cache.

### Step 3 — Chat

In **Chat**, type a message and press Return. The store:

1. folds any staged text-file attachments into the user turn;
2. appends the visible user message to the transcript;
3. sends either an incremental suffix or a re-primed history to the engine;
4. streams reasoning, visible text, and tool-call markup;
5. executes built-in tools as needed;
6. persists the chat session and updates context usage.

### Thinking, Stop, and New Chat

**Thinking** toggles reasoning-aware prompt rendering. Reasoning tokens are shown
in a collapsible block. **Stop** cancels the current generation. **New Chat**
starts a fresh persisted conversation with the current agent.

### Tools

The tool sheet controls whether tools are declared and which built-ins are
enabled. Compact declarations reduce prefill tokens by sending a shorter
`name(parameters)` declaration, while full declarations stay closer to the
training template.

Tool results are inserted back into the conversation as user-side tool-result
turns. Built-ins run automatically. Unknown tools can be answered manually.

### Sub-Agents

`subagent_run` delegates a focused question to an isolated decoder context. The
sub-agent can receive a project file, the whole project map, or a plain task. It
uses its own content-keyed KV prefix cache, may run tools internally, and returns
a concise answer. The main chat stores only the call and answer.

### Text-File Attachments

The composer can import text files. Files are decoded as UTF-8 or Latin-1 and
folded into the next user prompt between clear delimiters. The transcript shows
file-name badges; full file content enters the model only when you send.

## 7. Automatic Configuration and Hardware Presets

The app detects physical RAM and proposes conservative defaults:

| RAM tier | Default |
|---|---|
| `<24 GB` | 2-bit quant, context 4096, SSD streaming expected. |
| `<80 GB` | 2-bit quant, context 8192, most hot weights can stay in page cache. |
| `<200 GB` | 2-bit quant, context 32768. |
| Very large RAM | Q4 may fit; context 32768. |

These presets do not change correctness; they reduce the chance of memory
pressure. You can still raise context manually up to 1M, but KV and scratch
memory scale with it.

## 8. Memory, Streaming, and GUI Defaults

The current GUI defaults are tuned for the measured fast path on a 16 GB Apple
Silicon machine:

| Setting | GUI default | Why |
|---|---|---|
| Expert cache | `16` slots/layer | Keeps hot routed experts resident when routing is concentrated. |
| Expert pread | ON below 24 GB RAM | Bypasses page cache for expert slabs so dense weights are not evicted. |
| Expert bundle | ON | Turns scattered expert miss reads into one contiguous burst. |
| Dense-weight streaming | ON below 24 GB RAM | Uses a small staging ring instead of a multi-GB dense resident set. |
| `mlock` hot buffers | ON | Avoids memory-compressor churn on hot shared Metal buffers. |
| Q4 attention projections | ON | Lossy speed path that removes large Q8 projections from per-token SSD traffic. |
| Disk KV | ON | Reuses known prefixes across chats, reloads, and server requests. |
| Raw-KV ring | OFF | Available as an experiment; full KV is the conservative default. |

Most of these values are persisted and apply on the next model load. The app
performs a one-time migration from older experimental defaults to this faster
profile; future user changes are preserved.

### Page Cache vs Wired Buffers

No-copy mmap lets the OS decide what remains resident. This is excellent when
RAM is sufficient, but on small systems expert streaming can evict dense weights.
`DS4_EXPERT_PREAD=1` bypasses page cache for expert slabs, while
`DS4_DENSE_STREAM=1` streams dense layer tensors through a small staging ring.
`DS4_RESIDENT_DENSE=1` is still available for CLI experiments and RAM-rich
systems, but the app decides resident dense automatically from RAM and prefers
dense streaming on low-RAM systems.

### Expert Bundle

`DS4_EXPERT_BUNDLE=1` uses a sidecar where every routed expert's gate, up, and
down slabs are contiguous. Numerics do not change: the sidecar is a reordered
copy of bytes already present in the GGUF. The tradeoff is disk space. On a
Flash 2-bit model the sidecar can be tens of GB, so the app checks space and
logs when creation is skipped.

The Settings panel can generate the bundle immediately. In sandboxed builds, the
app first tries to reuse a readable `<model>.expbundle` next to the GGUF; if it
cannot write there, it builds under `Application Support/DwarfStar/expert-bundle`.

### Dense Streaming, `mlock`, and Q4 Dense Cache

Dense streaming reads layer `i+1` while the GPU computes layer `i`. It uses
`pread + F_NOCACHE`, avoids page-cache churn, and takes precedence over resident
dense weights.

`DS4_MLOCK=1` pins hot buffers best-effort. This includes expert-cache pools,
dense-stream staging, output-head resident buffers, and Q4 dense buffers. Failure
to pin is not fatal; it only means macOS may compress or page those buffers.

`DS4_DENSE_Q4=1` is a lossy speed path. It requantizes the largest Q8 attention
projections to Q4_K, caches the result under
`Application Support/DwarfStar/q4-cache`, and keeps the reduced buffers resident.
Disable it when you want the closest full-Q8 behavior rather than maximum
single-machine throughput.

### KV Cache

KV memory grows with context. Disk KV stores prefix checkpoints so later requests
or sessions with the same prefix can restore instead of redoing prefill. This is
especially useful for stateless HTTP requests that resend the same conversation
prefix.

### Raw-KV Ring

NSA sliding-window attention reads only the recent raw rows. `DS4_RAW_RING=1`
keeps raw KV in a ring of `nSWA` rows, reducing raw-KV memory. It does not remove
all compressed KV state.

### Expert Cache

The expert slot-cache is per-layer. Each slot holds one expert. On 2-bit Flash,
one slot costs roughly 6.9 MB per layer. Slot budgets should be tuned with the
Tuning tab: look at hit-rate and per-layer concentration. Low hit-rate means the
cache is not paying for its wired memory.

### Route/Attention Profiling

`DS4_PROFILE_ROUTE=1` splits decode route/attention into subphases such as
compressor, Q projection, KV projection, attention, and output. It adds extra
command-buffer synchronization, so use it to understand ratios and bottlenecks,
not to measure absolute speed.

## 9. Advanced Panels

### Server (`ServerView` + `LocalServer`)

The server is native and in-process. It exposes the single shared engine loaded
in Settings: no subprocess, no second model copy, and no separate resident-Q4 or
`mlock` allocation. `InferenceService` is an actor, so chat, server requests,
and local benchmark calls are serialized safely.

It exposes:

- `/v1/models`
- `/v1/chat/completions`
- `/v1/responses`
- `/v1/completions`
- `/v1/messages`

Load the model in Settings before starting the server. Stopping the server only
unbinds the listening socket; the shared chat engine stays alive.

### Distributed (`DistributedView` + `DS4Engine/Distributed/*`)

Distributed mode splits layers across workers. The Worker tab starts a listener
for one slice. Settings configures the coordinator peer list, activation bit
width, prefill chunk size, and optional worker-to-worker forwarding. Chat renders
the distributed conversation when the app mode is **Distributed**.

Distributed tool calls execute on the coordinator Mac, so project tools refer to
the coordinator's active project.

### Benchmark

The benchmark panel measures prefill and generation throughput at increasing
context sizes. In Local mode it reuses the loaded shared engine when the chat is
idle; the run mutates KV and is refused while generation is active. In
Distributed mode it reuses the connected coordinator, so it must not overlap a
distributed chat generation.

### Diagnostics

Diagnostics opens the GGUF only for tokenizer metadata. It can dump token ids for
arbitrary text and show the model chat template plus tool format. This replaces
the old subprocess-driven `ds4 --dump-tokens` workflow.

### Model Downloads

The download sheet uses the native Swift `ModelDownloader`: resumable HTTP Range
downloads from Hugging Face, `.part` resume files, and optional SHA-256 content
verification. Downloads go to `<scriptDir>/gguf`.

## 10. Build, Run, and Package

```sh
make
make xcodeproj
swift run DS4Demo
swift run DwarfStar
make embed-kernels
make app
```

`make embed-kernels` must be run after editing files in `metal/`. It regenerates
`Sources/DS4Metal/Runtime/KernelSources.swift`, which is what the app and CLI
compile into the binary.

`make app` builds `build/DwarfStar.app` and signs it ad-hoc. For distribution,
sign with a Developer ID and notarize.

### Path Resolution

`AppEnvironment` resolves development and bundle paths:

- in development, it uses the configured project root and GGUF folder;
- in a `.app`, it resolves resources through the bundle;
- model and project files selected by the user are stored via security-scoped
  bookmarks.

## 11. Troubleshooting

| Symptom | Likely Cause | What to Try |
|---|---|---|
| Model fails to open under the packaged app | Sandbox access missing | Choose the GGUF with **Browse** instead of typing a path. |
| Output is nonsense | Quantization mismatch or wrong GGUF | Run `DS4_TYPES_ONLY=1 swift run DS4Demo <gguf>` and compare expected dtypes. |
| Very slow decode on 16 GB | SSD expert streaming or dense rereads dominate | Use the GUI fast defaults: expert pread, dense streaming, `mlock`, Q4 attention cache, expert bundle, moderate context. |
| First load takes a long time | Expert bundle or Q4 dense cache is being built | Watch the engine log. Later loads reuse the sidecar/cache. |
| Expert bundle is skipped | Not enough writable disk space or sandbox cannot write next to model | Use the Settings bundle directory under Application Support or free disk space. |
| Resident dense makes things worse | Wired memory pressure | Prefer dense streaming on 16 GB systems; resident dense is automatic in the GUI and mainly useful on RAM-rich systems or CLI A/B tests. |
| Expert cache does not help | Routing is too uniform or cache too small | Check Tuning hit-rate and per-layer concentration; compare uniform vs usage-driven allocation. |
| Distributed chat cannot connect | Route incomplete or workers not started | Start workers first and ensure slices cover every layer contiguously. |
| Server works but chat slows down | Shared GPU/SSD resources | Avoid simultaneous chat and server generation on the same Mac. |
| Build cannot write Swift/clang cache in sandbox | Toolchain cache outside writable roots | Build outside the managed sandbox or configure writable cache paths. |

## 12. Glossary

| Term | Meaning |
|---|---|
| GGUF | Model container format used by the DS4 engine. |
| MoE | Mixture-of-Experts; routed FFN experts selected per token/layer. |
| Routed expert | Expert tensor selected by the router for the current token. |
| Non-routed weights | Always-needed dense weights such as attention projections and shared FFN. |
| KV cache | Attention key/value state accumulated over the context. |
| Raw-KV ring | Constant-size raw sliding-window KV buffer. |
| NSA | Native Sparse Attention / compressor path used by DeepSeek-V4. |
| HC | Hyper-Connection hidden state transported through layers/workers. |
| DSML | DeepSeek tool-call markup opened by the `｜DSML｜` token. |
| Usage imatrix | Per-layer expert routing-frequency table used for cache tuning. |
| Slot-cache | GPU-resident LRU cache of hot experts. |
| Expert bundle | Sidecar that stores each expert's slabs contiguously for faster miss reads. |
| Dense streaming | Per-layer dense-weight staging path that overlaps SSD reads with compute. |
| Q4 dense cache | Cached requantized Q4_K copies of large attention projections. |
| `mlock` | Best-effort request to keep hot buffers resident and out of the memory compressor. |
| Prefill | Processing prompt tokens before generation. |
| Decode | Token-by-token generation after prefill. |
