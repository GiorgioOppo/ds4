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

The first three layers are hash-routed: expert selection comes from the token id
through the `ffn_gate_tid2eid` table, exactly like the C reference, not from the
hidden state. This matters for expert prefetching (the selection is known before
compute) and for distributed shards that cover those layers.

At load the engine validates GGUF metadata the same way the C loader does
(a port of `config_validate_model`) and selects the declared profile. The local
runtime is wired for the Flash profile (43 layers, 256 experts); a valid Pro
GGUF is refused with an explicit error instead of being decoded with Flash
shapes into garbage. Numerics also follow the C reference: the RMSNorm and
Hyper-Connection epsilons default to `1e-6`, matching `DS4_DEFAULT_RMS_EPS`
and `DS4_DEFAULT_HC_EPS` in `ds4.c`.

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

See `Sources/DS4Demo/README.md` for diagnostic examples, and the root README's
[Configuration Reference](../README.md#configuration-reference) for the
complete list of `DS4_*` environment variables.

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
| **MCP** | Configure external MCP servers whose tools extend the built-ins. |
| **Project** | Imported project library and active project cache. |
| **Tuning** | Expert slot-cache controls and usage imatrix. |
| **Server** | OpenAI/Anthropic-compatible native HTTP server. |
| **Worker** | Run this Mac as a distributed worker; the coordinator assigns GGUF, settings, and layer slice. |
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

Sandboxed builds can read ONLY the picked `.gguf` file: sidecar caches sitting
next to it (`.q4dense`, `.expbundle`, e.g. built by the CLI demo) are invisible,
so the app would re-create them inside its own container — a slow first load and
gigabytes duplicated. **Grant Model Folder Access…** grants the whole model
folder so those sidecars are reused directly; doing it once per model folder is
recommended.

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
- **Expert look-ahead:** speculatively prefill the next layer's cache slots
  while the current layer computes (0 = hash layers only, which are always
  prefetched exactly).

Every knob behind these switches, plus the CLI-only ones, is documented in the
root README's [Configuration Reference](../README.md#configuration-reference),
which is the complete parameter reference.

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

Chat sampling uses the configured temperature and repetition penalty with a
fixed top-k of 40 (the llama.cpp default). Sampling over the entire 129k
DeepSeek vocabulary let the noisy tail of a 2-bit model occasionally pick a
Chinese token and drag the whole answer into Chinese; the cap removes that tail
without hurting variety. The HTTP server still honors per-request `top_k`.

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

### Analyzing a GitHub Repository from Chat

The Coding and Code agents expose `github_clone`: the model downloads a PUBLIC
GitHub repository as an HTTPS tarball (the request is pinned to
`codeload.github.com`, owner/name/ref are strictly validated, and the archive
is size-capped), extracts it under
`Application Support/DwarfStar/github-projects` — any symlinks in the archive
are stripped, and the project tools independently refuse paths that resolve
outside the root — and imports it into the project index as the ACTIVE
project, replacing the current one. The tool's
result is deliberately small: a directory tree with file counts plus the
documentation files to read first (README, root `.md` files, `docs/`).

From there the model works context-frugally, the same way as with a folder
imported from the Project tab: structure via `project_tree`/`project_list`,
file names via `project_find`, code search via `project_search` (optionally
scoped to a subfolder), and targeted reads via `project_read` in line-capped
chunks. Nothing enters the conversation KV until a tool actually returns
content, so a large repository can be analyzed without prefilling it into the
context. `github_clone` is not grantable to sub-agents because it swaps the
shared active project.

Cloned repositories are also first-class citizens in the GUI: they appear
automatically in the **Project tab** (and in the chat/distributed Project
menus) as library entries marked with a download icon and named
`<owner>-<repo> (GitHub)`, already active right after the clone. From there
you can switch back and forth between clones and user-imported folders with
**Activate**, exactly like any other project. Removing a cloned entry deletes
the copy from `Application Support/DwarfStar/github-projects` (it can be
re-cloned anytime); removing a user-imported folder never touches the disk.

### MCP Servers

The **MCP** tab connects the app, as a Model Context Protocol CLIENT, to
external tool servers. Each configured server is either:

- **stdio** — the app spawns the server (`npx`, `uvx`, or any executable) as a
  child process and speaks newline-delimited JSON-RPC over stdin/stdout;
- **HTTP** — Streamable-HTTP transport: JSON-RPC is POSTed to the server URL
  (with optional headers such as an Authorization token), for remote servers or
  local ones the sandbox cannot spawn.

At connect time the app runs the `initialize` handshake and fetches
`tools/list`. Each server tool then appears next to the built-ins — in the chat
Tool sheet under "MCP Tools" and in every agent's tool list — under the
namespaced name `mcp_<server>_<tool>` (e.g. `mcp_fs_read_file`). When the model
calls one, the app forwards `tools/call` to the server and feeds the text
result back into the conversation; server errors and timeouts come back as
error results the model can react to. Sub-agents cannot be granted MCP tools.

Configs persist across launches and can be imported/exported in the standard
`{"mcpServers": …}` JSON shared with Claude Desktop, Cursor, and VS Code.
Note for sandboxed (App Store) builds: stdio child processes inherit the app
sandbox, so servers needing broad file or network access should run outside
the app and be reached over HTTP. Dev builds (`swift run`, `make app`) are
unsandboxed. Engine-side code lives in `Sources/DS4Engine/Tools/MCP/`
(`MCPManager`, `MCPClient`, transports); the panel is
`Sources/DwarfStar/Settings/MCPServersView.swift`.

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

Beyond the static presets, Settings offers three measurement-driven tools that
tune the engine for the actual machine. All three require a loaded model, keep
the chat unusable while they run, and print a report to the panel and the
engine log.

### Prefill Benchmark (quick ~3 min / full ~10 min)

The **Benchmark** section in Settings measures the two prefill knobs the engine
re-reads on every prefill call — `DS4_PREFILL_UNION` (max experts per group in
a prefill union) and `DS4_PREFILL_CHUNK` (tokens per chunk) — then applies and
persists the fastest combination:

- **Quick** compares union 192 vs 256 on a 256-token prefill. It deliberately
  skips union 64: at short scale noise can make it win, while on real prefills
  it is the catastrophic value.
- **Full** compares union 64/192/256 on a 512-token prefill, then chunk
  512 vs 1024 with the winning union on a 1024-token prefill (below 512 tokens
  a second chunk does not exist, so quick mode cannot measure it).

The applied values are shown in the "Attivi (prefill)" row.

### Per-Machine Auto-Tune (~15-25 min)

**Auto-tune macchina** finds the best LOAD-time knobs for this chip and RAM —
expert cache slots, dense-stream ahead depth, async FFN pipeline, expert
look-ahead, and `DS4_Q8_NSG` (Q8 matvec reduction partitioning, the one knob
tied to GPU core count) — with a coordinate descent: one model reload per
candidate, measured with a short warmed-up decode.

- Candidates are RAM-gated (a 16 GB machine never tries 32 slots; a 96 GB Max
  does), and after the first memory-pressure collapse larger candidates are
  skipped.
- The metric is the steady-state p99 decode speed, but a candidate only counts
  if it is STABLE. Stability is judged from the temporal profile of per-token
  speeds: a swap spiral has a tail slower than its head (progressive
  degradation), which is exactly what separates "more slots = more hits" from
  "more slots = swap spiral". A cold start has the opposite signature and does
  not disqualify a configuration.
- A candidate must beat the incumbent by more than 2% (anti-noise margin);
  otherwise the persisted value stays.

The winners are applied, persisted per chip/RAM, and summarized in the
"Attivi (load)" row (slots, dense-ahead, async FFN, look-ahead, q8nsg). On
error the knobs are restored to their starting values.

### "Align to fast demo config" and One-Time Migrations

The **Align to fast demo config** button resets every memory toggle to the
measured-fast set: slots 16, pread + dense streaming + `mlock` + Q4 + bundle
ON, dense-ahead 2, look-ahead 0, raw ring OFF. The app also runs one-time
migrations at startup that move defaults persisted by older experimental builds
to the current measured configuration (most recently: slots 16, dense-ahead 2,
look-ahead 0); future manual changes are always preserved.

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
| Async FFN pipeline | ON | Commits the routed FFN asynchronously so the GPU no longer drains between layers (measured +10%, token-identical output). |
| Dense-stream ahead | `2` | Staging ring reads one layer ahead of compute. |
| Expert look-ahead | `0` | Speculative prefill measured neutral; the hash layers are always prefetched exactly. |

Most of these values are persisted and apply on the next model load. The app
performs one-time migrations from older experimental defaults to this measured
profile (see Section 7); future user changes are preserved. The complete list
of engine environment knobs, including the CLI-only ones, lives in the root
README's [Configuration Reference](../README.md#configuration-reference).

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
dense weights. `DS4_DENSE_AHEAD` controls the staging-ring depth (default 2 =
one layer ahead, the measured optimum; the auto-tune explores 1-3).

Two lossless refinements are ON by default inside the dense stream:

- `DS4_RESIDENT_COMP` keeps the four NSA compressor projections resident
  (~0.6 GB) instead of streaming them — they are read every token on 41 of 43
  layers, the single densest repeat-read in the stream. Same bytes, identical
  numerics; `=0` restores full streaming as a tight-RAM fallback.
- `DS4_LAZY_IDX` skips staging the indexer SCORING projections (~360 MB/token)
  when the sparse top-k selection provably cannot activate at the current
  context size. The proof is recomputed on every load, so a larger context
  re-enables the staging automatically.

Whether the sparse indexer path activates at all is governed by
`DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` (default 1024, same env override
and allowed values as the C): decode keeps attention dense over all compressed
rows until their count exceeds the threshold, because around the ~2K frontier
the sparse path's score/top-k setup costs more than the smaller dense scan.

`DS4_MLOCK=1` pins hot buffers best-effort. This includes expert-cache pools,
dense-stream staging, output-head resident buffers, and Q4 dense buffers. Failure
to pin is not fatal; it only means macOS may compress or page those buffers.

`DS4_DENSE_Q4=1` is a lossy speed path. It requantizes the largest Q8 attention
projections to Q4_K, caches the result under
`Application Support/DwarfStar/q4-cache`, and keeps the reduced buffers resident.
The first-launch conversion checkpoints its partial cache between batches, so a
load interrupted mid-requant resumes from the completed tensors on the next
launch instead of restarting from zero (progress and any cache-write failures
are logged in the engine log as `DS4 q4cache:` lines). Disable it when you want
the closest full-Q8 behavior rather than maximum single-machine throughput.

### Async FFN Pipeline and GPU Kernel Knobs

`DS4_ASYNC_FFN=1` (default ON) commits the routed-FFN work asynchronously so
the GPU does not drain between layers. The Metal queue is in-order, so the
output is token-identical to the synchronous path — measured +10% on M1 Pro.
Setting it to `0` restores the historical synchronous waits and is kept as a
debugging parachute.

`DS4_FUSED_HC` (default ON) fuses the Hyper-Connection reduce tail —
split + collapse + RMSNorm — into one dispatch instead of three; it runs twice
per layer, saving ~170 dispatches/token. Same math; only the RMSNorm reduction
order differs (±1 ulp class). `=0` restores the unfused path.

`DS4_Q8_NSG` sets the simdgroups per threadgroup in the Q8 matvecs (a pure
occupancy knob — identical numerics). The optimum depends on GPU core count:
4 is the reference (best on M1 Pro); wider GPUs (Max/Ultra) may prefer 6-8.
The engine re-reads it on every model load, which is how the auto-tune
explores it.

### KV Cache

KV memory grows with context. Disk KV stores prefix checkpoints so later requests
or sessions with the same prefix can restore instead of redoing prefill. This is
especially useful for stateless HTTP requests that resend the same conversation
prefix.

Checkpoints are moved between disk and the KV buffers in per-layer batches, in
both directions, so a restore never materializes the whole file in RAM: each
layer is read (F_NOCACHE), imported into the decoder, and freed before the next
one is loaded — peak memory is one layer instead of ~3× the checkpoint size of
the old load-everything path. Saving is symmetric: the background writer is the
sole owner of the exported snapshot and drops each layer's buffers as soon as
they are written, so RAM falls during the write instead of holding the full
checkpoint until the end.

### Raw-KV Ring

NSA sliding-window attention reads only the recent raw rows. `DS4_RAW_RING=1`
keeps raw KV in a ring of `nSWA` rows, reducing raw-KV memory. It does not remove
all compressed KV state.

### Expert Cache

The expert slot-cache is per-layer. Each slot holds one expert. On 2-bit Flash,
one slot costs roughly 6.9 MB per layer. Slot budgets should be tuned with the
Tuning tab: look at hit-rate and per-layer concentration. Low hit-rate means the
cache is not paying for its wired memory.

`DS4_EXPERT_LOOKAHEAD` (Settings stepper, default 0) prefills the next layer's
slots while the current layer computes: exact for the hash-routed layers
(always on — the token-id selection is known in advance), top-N from the usage
prior when greater than 0. Speculative I/O runs only in the SSD-idle window and
yields to the real gather, so a wrong guess wastes only idle bandwidth.

`DS4_PREAD_SPLIT` (default 1) splits each expert slab into N disjoint ranges
read concurrently during the direct (`F_NOCACHE`) slot-cache fill, raising the
NVMe queue depth for the same bytes. The historical single-`pread` path is the
default; values up to 8 are accepted.

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

Distributed mode splits layers across workers, and the COORDINATOR defines
each worker's job: the Worker tab only starts an idle listener; at connect time
the coordinator partitions the layers across the peer list (in order, last
slice owns the output head) and sends each worker the GGUF, the context size,
the expert-cache budget, and its slice. Each worker resolves the GGUF locally
(the coordinator's exact path, else a same-named file next to its local GGUF
from Settings), loads its engine, and replies ready. Settings configures the
coordinator peer list, activation bit width, prefill chunk size, and optional
worker-to-worker forwarding. Chat renders the distributed conversation when
the app mode is **Distributed**.

Distributed tool calls execute on the coordinator Mac, so project tools refer to
the coordinator's active project.

Robustness (protocol v2): the coordinator validates each worker's protocol
version at connect; every chat turn or benchmark run carries a `session` id
that workers echo in their results, so a reply left in a socket buffer by a
stopped turn is discarded instead of corrupting the next one. Workers validate
every WORK frame (payload size, layer bounds, position) before running it and
serve one turn at a time — a competing coordinator receives an explicit error.
Stop propagates to the cluster generation task and takes effect at the next
chunk boundary. Frames are plaintext TCP with no authentication: run
distributed mode only on trusted networks.

File distribution (protocol v5): workers need no local files at all. After the
version handshake the coordinator offers a manifest — name, size, and SHA-256
of the GGUF and (when enabled) the expert-bundle sidecar and the Q4 requant
cache (derived files travel instead of being rebuilt on every worker) — and each worker
requests only what it is missing, verifying against its managed store
(`Application Support/DwarfStar/dist-models`) and hash-matching local files.
The transfer streams in 4 MB chunks with the hash accumulated inline, so the
enormous setup happens once: subsequent connects verify cached manifests in
milliseconds and send nothing. The sidecar on/off decision travels in the
ASSIGN, like every other setting.

KV continuity (protocol v4): turns no longer re-prefill the whole conversation
every time. The coordinator reuses the in-memory prefix committed by the last
clean turn when the re-rendered conversation extends it exactly; on a cold
start (or when the coordinator's disk-KV setting is on) it negotiates a
restore across all shards — each worker keeps slice-keyed disk checkpoints,
saved after clean turns and restored only when EVERY shard holds the same
prefix; any mismatch falls back to a cold prefill. The ASSIGN also carries the
usage imatrix, so each worker pre-warms its expert slot-cache (and persists
its own slice-refined profile between sessions).

Token-id routing (protocol v7): WORK frames carry the chunk's token ids. The
first three layers route experts by token id (`ffn_gate_tid2eid`), so a shard
that covers them cannot route from the HC state alone.

Resumable transfers (protocol v8): the file offer carries a chained-hash
checkpoint list per file — one SHA-256 every 256 MB, each folded over the
previous, so checkpoint `k` commits to the whole prefix (a 70 GB GGUF adds
~9 KB to the offer). A worker keeps its `.part` file across disconnects AND
sessions, validates it block-by-block against the chain, truncates to the last
good checkpoint, and answers with a per-file resume offset; the coordinator
streams from there. Transport errors during a peer's setup are retried up to 3
times (semantic errors — version mismatch, bad slice, worker-reported errors —
are not), and each attempt re-sends at most 256 MB.

Coordinator performance knobs (protocol v9): the ASSIGN carries the
coordinator's measured `DS4_*` performance environment, restricted to a
whitelist on both sides (`Dist.perfKnobKeys` in
`Sources/DS4Engine/Distributed/DistProtocol.swift`) so the wire can never set
arbitrary environment on a worker. All whitelisted knobs are performance-only
and cannot change the shard's numerics; the one lossy knob (`DS4_DENSE_Q4`)
travels as a typed field and its cache as a file. Before v9 a worker with
factory defaults ran without dense streaming/`mlock`/pread and measured
0.37 tok/s where the same hardware did 2.7 locally — the coordinator's measured
configuration is now part of the job definition.

Worker setup runs IN PARALLEL: file transfer and engine load of every peer
proceed together, so the route activates in max(worker setup) instead of the
sum. With files already distributed (from the second connect on), N workers
become ready in the time of one; on a cold start the transfers share the
coordinator's bandwidth, but one worker's load still overlaps the others'
transfers. Route order remains the peer-list order.

The full Q4 requant cache transferred by the coordinator also serves worker
slices directly: cache records are matched by key (layer, field), so a shard
loads the complete `.q4dense` in about half a second instead of requantizing
its slice from scratch. Partial-slice writes go to a suffixed
`<cache>.L<lo>-<hi>` file and never overwrite the full cache.

### Benchmark

The benchmark panel measures prefill and generation throughput at increasing
context sizes. In Local mode it reuses the loaded shared engine when the chat is
idle; the run mutates KV and is refused while generation is active. In
Distributed mode it reuses the connected coordinator, so it must not overlap a
distributed chat generation.

The generation series in the chart and report is the steady-state p99 of the
per-token speeds, not the mean: the mean is dragged down by the cold first
tokens, while p99 is the cruise speed the run settles into (the log prints both).
The chart draws a fine 0.1 t/s gridline because the A/B differences that matter
are in the 0.05-0.3 t/s range.

This panel measures at increasing context frontiers; the Settings tab has its
own benchmark and auto-tune buttons that measure and APPLY configuration knobs
(Section 7). The currently applied values appear in Settings as the
"Attivi (prefill)" and "Attivi (load)" rows.

### Diagnostics

Diagnostics opens the GGUF only for tokenizer metadata. It can dump token ids for
arbitrary text and show the model chat template plus tool format. This replaces
the old subprocess-driven `ds4 --dump-tokens` workflow.

### Model Downloads

The download sheet uses the native Swift `ModelDownloader`: resumable HTTP Range
downloads from Hugging Face, `.part` resume files, and optional SHA-256 content
verification. Downloads go to `<scriptDir>/gguf`.

Authentication is configured in **Settings → Hugging Face**: paste a read-only
token from `huggingface.co/settings/tokens` and press Save. The token is stored
in the macOS **Keychain** (never UserDefaults), shown afterwards only in
redacted form, and sent by the downloader as `Authorization: Bearer` — needed
for gated/private repositories and to avoid anonymous rate limits. Remove
deletes it from the Keychain. When no token is saved, the downloader falls back
to the `HF_TOKEN` environment variable, then `~/.cache/huggingface/token`; the
download sheet shows which source, if any, is active.

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
| Load is refused with "unsupported DeepSeek4 shape" | The GGUF declares the Pro profile | The local runtime supports only the Flash profile (43 layers, 256 experts); use a Flash GGUF. |
| First load rebuilds sidecars that already exist next to the GGUF | Sandbox can read only the picked file | Use **Grant Model Folder Access…** in Settings so `.q4dense`/`.expbundle` next to the model are reused. |
| Output is nonsense | Quantization mismatch or wrong GGUF | Run `DS4_TYPES_ONLY=1 swift run DS4Demo <gguf>` and compare expected dtypes. |
| Very slow decode on 16 GB | SSD expert streaming or dense rereads dominate | Use the GUI fast defaults: expert pread, dense streaming, `mlock`, Q4 attention cache, expert bundle, moderate context. |
| First load takes a long time | Expert bundle or Q4 dense cache is being built | Watch the engine log. Later loads reuse the sidecar/cache. |
| Load seems stuck on "Riquantizzazione Q4 (solo il primo avvio)" with CPU pegged | The one-time Q8→Q4_K conversion takes minutes — or HOURS in an unoptimized Debug build (Xcode Run default, plain `swift build`) | Watch the percentage next to the stage label (it moves in 0.1% steps) and the engine log: a `DS4 q4cache: ATTENZIONE: build di DEBUG` line means you should rebuild in Release (`make app` or the Xcode Release configuration). Partial checkpoints are written every 16 tensors, so even a force quit resumes from where it stopped; if no `.q4dense` ever appears, look for `DS4 q4cache:` write errors (disk space, cache folder). |
| Expert bundle is skipped | Not enough writable disk space or sandbox cannot write next to model | Use the Settings bundle directory under Application Support or free disk space. |
| Resident dense makes things worse | Wired memory pressure | Prefer dense streaming on 16 GB systems; resident dense is automatic in the GUI and mainly useful on RAM-rich systems or CLI A/B tests. |
| Expert cache does not help | Routing is too uniform or cache too small | Check Tuning hit-rate and per-layer concentration; compare uniform vs usage-driven allocation. |
| Distributed chat cannot connect | Route incomplete or workers not started | Start workers first and ensure slices cover every layer contiguously. |
| Distributed connect fails with a version mismatch | Coordinator and worker run different builds | Update every Mac to the same DwarfStar build; the protocol version must match exactly. |
| Distributed file transfer interrupted | Network hiccup mid-transfer | Nothing to do: the worker keeps its `.part`, setup retries up to 3 times, and each retry resumes from the last 256 MB checkpoint. |
| Auto-tune reports an unstable baseline or progressive collapse | Memory pressure from other apps or too-large candidates | Free RAM (close other apps) and rerun; collapsing candidates are skipped automatically. |
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
| Hash layer | One of the first 3 layers, whose experts are selected by token id via the `tid2eid` table instead of the router. |
| Slot-cache | GPU-resident LRU cache of hot experts. |
| Expert bundle | Sidecar that stores each expert's slabs contiguously for faster miss reads. |
| Dense streaming | Per-layer dense-weight staging path that overlaps SSD reads with compute. |
| Q4 dense cache | Cached requantized Q4_K copies of large attention projections. |
| `mlock` | Best-effort request to keep hot buffers resident and out of the memory compressor. |
| Prefill | Processing prompt tokens before generation. |
| Decode | Token-by-token generation after prefill. |
