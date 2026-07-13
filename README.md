# DwarfStar — DeepSeek-V4 on macOS

DwarfStar is a native Swift / SwiftUI application for running
**DeepSeek-V4-Flash (284B MoE)** on Apple Silicon with a **pure-Swift Metal
inference engine**. The engine is a faithful port of upstream `ds4.c` /
`ds4_metal.m`: no C runtime engine, no prebuilt static library, and no external
process for normal inference. The 2-bit GGUF runs on a 16 GB MacBook by
streaming routed expert weights from SSD.

> Documentation starts at [`docs/README.md`](docs/README.md). The
> [Configuration Reference](#configuration-reference) below is the
> authoritative parameter table; dedicated guides cover the
> [inference pipeline](docs/PIPELINE-INFERENZA.md),
> [Metal backend](docs/BACKEND-METAL.md),
> [distributed inference](docs/INFERENZA-DISTRIBUITA.md),
> [GUI/server](docs/GUI-SERVER-E-API.md) and
> [testing](docs/TESTING-E-VALIDAZIONE.md).

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
  `Sources/DS4Metal/Runtime/Generated/KernelSources.swift` from `metal/*.metal`, so a
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
  XML-style call format produced by the compiled `ChatRenderer`, implemented
  against the reference DeepSeek-V4 template. The runtime does not interpret
  the GGUF's Jinja template; compact mode deliberately shortens tool declarations
  to reduce prefill. Tool results return inside a user turn as
  `<tool_result>...</tool_result>`.
- **Layer-major prefill amortizes I/O.** Prefill runs in chunks: each layer's
  weights are loaded once per chunk and applied to all prompt tokens. The routed
  FFN phase gathers the union of experts for that chunk rather than 6 experts
  per token.

## The App

| Tab | Purpose |
|---|---|
| **Chat** | Streaming markdown chat, collapsible reasoning, live tool calls, multi-turn KV reuse, text-file attachments, active project menu, near-context-full warnings, Local or Distributed mode. |
| **Settings** | Shared model path, context size, RAM-aware execution mode, memory/I/O knobs, local model load, and distributed coordinator route. |
| **Agents** | Role editor: system prompt, icon, tools, JSON import/export, and per-agent expert-usage profile. |
| **MCP** | External MCP servers (stdio child process or Streamable HTTP): connection status, exposed tools, `mcpServers`-JSON import/export. |
| **Project** | Library of imported folders (sandbox bookmarks) and GitHub clones made in chat via `github_clone` (listed automatically). Project tools explore the active project without consuming chat context until a tool reads content. |
| **Tuning** | Expert cache slots, hit-rate, per-layer routing concentration, and the usage imatrix. |
| **Server** | Native in-process OpenAI/Anthropic-compatible HTTP server. |
| **Worker** | Runs this Mac as a distributed worker: it starts idle and the coordinator assigns its job — GGUF, settings, and layer slice. |
| **Benchmark** | Native throughput benchmark for prefill and generation across growing context sizes, local or distributed. |
| **Diagnostics** | Native tokenizer dump and chat-template/tool-format inspection. |

## Built-In Tools and Agents

Built-in DSML tools live one per file under
`Sources/DS4Engine/Tools/Builtins/`.

- **Project index tools:** `project_list`, `project_read`, `project_search`,
  `project_write`, `project_edit`, `project_reload` (re-index after
  out-of-band changes; `git stash` re-indexes automatically).
- **GitHub import:** `github_clone` downloads a public repository (HTTPS
  tarball, no git binary or credentials) into Application Support and makes it
  the active project, returning a compact orientation summary (tree +
  documentation files). The Coding and Code agents use it to analyze a repo
  with the project tools — structure via `project_tree`, targeted code search
  via `project_find`/`project_search` — instead of reading every file into the
  chat context. Clones also appear automatically in the Project tab (and the
  chat Project menu) as regular library entries, so they can be re-activated
  from the GUI at any time.
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

Beyond the built-ins, the **MCP** tab connects the app to external
[Model Context Protocol](https://modelcontextprotocol.io) servers (stdio or
Streamable HTTP). Their tools appear next to the built-ins — in the chat Tool
sheet and in agent tool lists — as `mcp_<server>_<tool>`, and calls are
forwarded to the server via `tools/call` (see `Sources/DS4Engine/Tools/MCP/`).

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

The current GUI preset favors the measured fast 16 GB path: 22 expert-cache
slots, raw-KV ring off, direct expert `pread`, dense-weight streaming,
best-effort `mlock`, the full Q4 set (`DENSE_Q4`, `QKV_Q4`, `SHARED_Q4`), disk
KV, expert bundle and MetalIO. Prefill uses union/chunk/route-batch 256/512/32;
dense read-ahead is 2, asynchronous FFN is on, and Q8/MoE/dense-Q4 NSG are all
4. Most memory toggles apply on the next model load.

## HTTP Server

The Server tab starts an OpenAI/Anthropic-compatible server on
`Network.framework`. It is native and in-process and exposes **the single shared
engine** loaded in Settings — no subprocess and no second model copy (a second
engine would double the resident-Q4 + mlocked buffers and OOM on 16 GB). The
`InferenceService` actor serializes callers, so a server request waits for an
in-flight chat turn and vice versa. Benchmark reuses the same shared engine.

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

Host, port, API key, CORS, and the accepted per-request sampling parameters
are documented in the
[Configuration Reference](#http-server-server-tab).

## Distributed Inference

Distributed mode implements pipeline parallelism by contiguous layer ranges,
modelled on `ds4_distributed.c`.

- Each **worker** owns a layer slice, its weights, and the KV shard for that
  slice only.
- The **coordinator** owns embedding, sampling, prompt rendering, and tool
  execution.
- The HC hidden state (`nHC x nEmbd` floats, transported at 32/16/8 bit) flows
  through workers for every token.
- Workers start FIRST but idle (no model loaded): at connect time the
  coordinator partitions the layers across the peer list, assigns each worker
  its job (GGUF, context size, cache budget, layer slice — the last slice also
  runs the output head), and waits for every worker to load and reply ready.
  Flash has 43 layers. Although metadata recognition and the download catalog
  know the 61-layer Pro profile, both current local and distributed execution
  reject it because the runtime and router geometry are Flash-specific.

The per-node benefit is reduced expert I/O: each worker streams roughly `1/N` of
the routed experts, making the hot working set more likely to stay in RAM.
Prefill runs in chunks, default 32 tokens per distributed frame. Optional
worker-to-worker forwarding can reduce network hops, but requires the
coordinator's LAN return address. All cluster settings (peer list, activation
bits, forwarding, and what the ASSIGN carries) are documented in the
[Configuration Reference](#distributed-settings--worker-tab).

Protocol v10 also implements **vertical expert parallelism**: the coordinator
runs the complete dense backbone and partitions the 256 routed experts of every
layer across worker masks. Worker SSD gathers can then run in parallel, at the
cost of roughly one network round-trip per routed layer. This mode requires a
wired link with RTT below about 1 ms; it is intentionally unsuitable for Wi-Fi.
See [`docs/INFERENZA-DISTRIBUITA.md`](docs/INFERENZA-DISTRIBUITA.md).

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
docs/             architecture, operation, development and validation guides
packaging/        .app bundle assembly and signing inputs
Tests/            unit and parity tests
```

Every source, test and operational folder has a local `README.md` explaining
ownership, dependencies, extension rules and how it connects to the rest of
the system.

The complete source map and placement rules are in
[`docs/STRUTTURA-PROGETTO.md`](docs/STRUTTURA-PROGETTO.md).

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

The demo honors every engine environment variable listed in the
[Configuration Reference](#configuration-reference) below; the profiling
workflow and per-knob prose live in
[`Sources/DS4Demo/README.md`](Sources/DS4Demo/README.md).

## Configuration Reference

Everything configurable in the project, in one place. Configuration lives in
three layers:

1. **GUI settings** — persisted in `UserDefaults` and applied by the app.
   Most of them work by exporting the matching engine environment variable at
   launch and whenever you change them, so the GUI is the source of truth for
   those knobs inside the app.
2. **Engine environment variables (`DS4_*`)** — read directly by the engine.
   The same knobs work in the app, the CLI demo, and the tests. In a bare
   process (demo/tests) the engine defaults below apply; in the app, the GUI
   settings overwrite the main ones at startup.
3. **Panel-local configuration** — HTTP server, distributed cluster, MCP
   servers, agents, and model downloads, each configured in its own tab.

Unless stated otherwise, engine knobs are read **when the model is loaded**:
change them, then (re)load the model. The hot-reload exceptions are
`DS4_PREFILL_UNION`, `DS4_PREFILL_CHUNK`, and
`DS4_PREFILL_ROUTE_BATCH`, which are re-read during every prefill call.

### GUI Settings (Settings Tab)

#### Model and engine

| Setting | `UserDefaults` key | Default | What it is / what it does |
|---|---|---|---|
| Model path | `DS4ModelPath` | dev builds: `<DS4_ROOT>/gguf/DeepSeek-V4-Flash-…-imatrix.gguf`; app bundle: empty | The GGUF the shared engine loads. Under the App Sandbox use **Browse**: the file picker grants a security-scoped bookmark (persisted as `ds4.modelBookmark` / `ds4.modelDirBookmark`) — a typed path alone cannot be reopened on the next launch. |
| Context size | `DS4ContextSize` | RAM-tiered: 4096 (<24 GB), 8192 (<80 GB), 32768 (above) | Maximum KV positions (`maxKeys`). KV caches and scratch scale with it — a 1M context allocates tens of GB of KV and starves the expert page cache, which is why the default follows RAM. The stepper still goes up to 1M for machines that can afford it. |
| Mode | `DS4EngineMode` | `Local` | `Local` runs the single in-process engine; `Distributed` routes chat/benchmark through the coordinator configured in the Distributed section. |
| Thinking (Chat toggle) | not persisted | off | Enables reasoning-token handling for the next turn: the model's chain-of-thought streams into a collapsible section. |

#### Sampling (chat)

| Setting | `UserDefaults` key | Default | What it is / what it does |
|---|---|---|---|
| Temperature | `DS4Temperature` | `0.6` | Softmax temperature. Lower = more focused and less drift — helps on the aggressively-quantized 2-bit model. |
| Repetition penalty | `DS4RepPenalty` | `1.1` | Values >1 discourage repeats and break the repeat-loop collapse quantized models fall into. `1.0` = off. |
| top-k | fixed | `40` | Not exposed in the GUI. Sampling over DeepSeek's full 129k vocabulary lets one noisy tail token drag a whole answer into Chinese at normal temperatures; capping at 40 cuts the tail without hurting variety. (The CLI demo is greedy by default and exposes its own env sampler knobs; the HTTP server takes per-request values.) |

#### Memory / I/O toggles

Each toggle persists a `UserDefaults` key and exports the engine variable
shown; all of them apply on the **next model load**. Defaults are the measured
fast 16 GB profile (a one-time migration aligns old installs; the
**Align to fast demo config** button re-applies it).

Benchmark provenance: unless a row says otherwise, the comparative figures in
this preset were measured on a **MacBook Pro M1 Pro with 16 GB unified memory**,
using the DeepSeek-V4 Flash 2-bit imatrix model, with the final preset validated
on **2026-07-13**. They are reference measurements, not guarantees for a
different chip, SSD, memory pressure, model build, or prompt.

| Setting | `UserDefaults` key | Engine variable | App default | What it is / what it does |
|---|---|---|---|---|
| Expert cache slots | `DS4ExpertCacheSlots` | `DS4_EXPERT_CACHE_SLOTS` | `22` | Per-layer LRU GPU cache for routed experts, ~6.9 MB wired per slot per layer on the 2-bit model. `0` = off. The 2026-07-13 M1 Pro comparison found 22 faster than 20 without observed memory collapse (about 70% hit-rate); without `mlock` or the full Q4 preset, smaller pools can be safer. |
| Disk KV cache | `DS4DiskKV` | — (service API) | on | Checkpoints completed generations to disk and restores matching prefixes on cold starts, so reloading the app does not re-prefill old conversations. |
| Disk KV budget | `DS4DiskKVBudgetKTok` | — (service API) | `1000` (= 1M tokens) | Total checkpoint budget in thousands of tokens across all saved conversations (~22 KB/token on the 2-bit model, so 1M ≈ 22 GB on disk). The live window is still `contextSize`. |
| Raw-KV ring | `DS4RawRing` | `DS4_RAW_RING` | off | Keeps raw KV only for the 128-row sliding-attention window instead of the whole context, making raw-KV RAM constant. Experimental; does not shrink the compressed KV. |
| Expert read-ahead | `DS4WillNeed` | `DS4_WILLNEED_EXPERTS` | on | `madvise(WILLNEED)` on exactly the experts the router just selected, before the gather. Non-speculative, numerics unchanged. |
| Direct expert reads | `DS4ExpertPread` | `DS4_EXPERT_PREAD` | on | Reads expert slabs with `pread`+`F_NOCACHE` straight into destination buffers, bypassing the page cache so ~1 GB/token of expert churn stops evicting dense weights. Measured ~+20% tok/s alone on the reference 16 GB machine. The bare property fallback is RAM-gated, but the current fast preset explicitly enables it. |
| Dense-weight streaming | `DS4DenseStream` | `DS4_DENSE_STREAM` | on | Streams each layer's dense tensors through a staging ring instead of keeping ~6 GB resident, overlapping SSD and GPU work. Identical numerics; the current fast preset explicitly enables it. |
| Hot-buffer pinning | `DS4MLock` | `DS4_MLOCK` | on | Best-effort `mlock()` of hot resident buffers (expert pools, resident head, staging ring, ~3.3 GB at defaults). Stops macOS from compressing buffers between tokens (measured −38% ms/token on 16 GB). |
| Q4 attention cache | `DS4DenseQ4` | `DS4_DENSE_Q4` | on | **Lossy.** Requantizes the three giant attention projections Q8→Q4_K at load and keeps them resident (~1.4 GB), removing ~4.6 GB/token from the SSD stream. Greedy output can diverge while staying coherent. Requires dense streaming; the first load writes a `.q4dense` cache. |
| Q4 q/kv cache | `DS4QkvQ4` | `DS4_QKV_Q4` | on | **Lossy; requires `DS4_DENSE_Q4`.** Also requantizes the remaining q_a and kv attention projections, keeping them resident and removing roughly another 0.7 GB/token from the dense stream. |
| Q4 shared FFN cache | `DS4SharedQ4` | `DS4_SHARED_Q4` | on | **Lossy; requires `DS4_DENSE_Q4`.** Requantizes the shared-expert gate/up/down projections and keeps them resident, leaving more SSD bandwidth for routed-expert gathers. |
| Expert bundle | `DS4ExpertBundle` | `DS4_EXPERT_BUNDLE` | on | Sidecar file with each expert's gate/up/down slabs contiguous: a cache miss becomes one ~7 MB sequential read instead of three scattered ones (measured gather 2.7 → 4.8 GB/s). Same bytes, same numerics. Built once next to the GGUF (or in Application Support); needs ~73 GB free and is skipped with a log when space is short. A Settings button builds/verifies it on demand. |
| MetalIO expert loading | `DS4MetalIO` | `DS4_MTLIO` | on | Loads expert-bundle ranges directly into slot-cache Metal buffers during decode. The app exports `DS4_MTLIO_MIN_GBS=4.0`: two sustained 64 MiB windows below 4 GB/s trigger fallback to `pread`. Prefill continues to use parallel `pread`. |
| Route/attn profiling | `DS4ProfileRoute` | `DS4_PROFILE_ROUTE` | off | Splits `route/attn` into five timed subphases in the engine log. Diagnostic only: the extra syncs slow generation, so read ratios, not absolute tok/s. |
| Prefill expert union | `DS4PrefillUnion` | `DS4_PREFILL_UNION` | `256` | Max experts grouped per layer-major prefill read. Hot-reloadable; the quick benchmark compares 192/256 and the full benchmark also measures 64. The app preset is 256; the independent bare-engine default remains 192. |
| Prefill chunk | `DS4PrefillChunk` | `DS4_PREFILL_CHUNK` | `512` | Tokens per prefill chunk. Every chunk reloads all layers' dense weights, so larger chunks amortize that fixed cost on long prompts. Hot-reloadable; benchmarked against 1024. |
| Prefill route batch | `DS4PrefillRouteBatch` | `DS4_PREFILL_ROUTE_BATCH` | `32` | Tokens whose route/attention dispatches share one command buffer. Hot-reloadable; the full benchmark measures 16/32/64/128 and requires a >2% improvement before replacing the persisted value. |

Five more knobs are persisted without a GUI toggle (set them via
`defaults write` for experiments; the auto-tune below explores them for you):

| `UserDefaults` key | Engine variable | App default | What it is / what it does |
|---|---|---|---|
| `DS4ExpertLookahead` | `DS4_EXPERT_LOOKAHEAD` | `0` | Speculative slot-cache look-ahead: while layer *i* computes, pre-fill layer *i+1*'s pool with its top-N usage-prior experts. `0` = only the hash-routed layers 0–2 (whose selection is exact) are prefetched. |
| `DS4DenseAhead` | `DS4_DENSE_AHEAD` | `2` | Staging-ring read-ahead depth (1–3) for dense streaming. `2` keeps layers i+1 and i+2 in flight (+1.5% measured, ~150 MB per extra slot). |
| `DS4AsyncFFN` | `DS4_ASYNC_FFN` | on | Asynchronous routed-FFN pipeline (+10% measured, token-identical output — correctness guaranteed by the in-order Metal queue). `false` is the debug parachute restoring synchronous waits. |
| `DS4Q8NSG` | `DS4_Q8_NSG` | `4` | Simdgroups per threadgroup for dense Q8_0 matvecs. It changes occupancy and the partition of floating-point partial sums, so low bits can differ; 4 is best on M1 Pro, while wider GPUs (Max/Ultra) may prefer 6–8. |
| `DS4MoeNSG` | `DS4_MOE_NSG` | `4` | Simdgroups per threadgroup for routed MoE/Q4 id kernels. It partitions output rows rather than the K reduction; occupancy depends on the GPU, so auto-tune measures it independently from Q8 NSG. |

The preset also exports `DS4_DENSE_Q4_NSG=4` directly. It has no independent
`UserDefaults` key; bare engine processes inherit `DS4_MOE_NSG` when it is not
set explicitly.

Resident dense weights (`DS4_RESIDENT_DENSE`) have **no GUI toggle**: the app
enables them automatically on ≥24 GB machines and clears any stale persisted
value (on 16 GB they slow the real app down; dense streaming takes precedence
anyway). The env knob remains for the demo.

Example — change a hidden knob on a dev build, then reload the model in the app:

```sh
defaults write com.dwarfstar.app DS4DenseAhead -int 3     # xcodegen/Xcode builds
defaults write org.ds4.dwarfstar DS4DenseAhead -int 3     # `make app` bundles
```

#### Benchmark and auto-tune (Settings buttons)

- **Prefill benchmark — quick** (~3 min): warms the loaded engine, then compares
  `DS4_PREFILL_UNION=192/256` on a 256-token context with 4 decode tokens. It
  intentionally excludes 64 because short-run noise can favor a value known to
  cause excessive rereads at realistic prompt lengths.
- **Prefill benchmark — full** (~15 min): compares union 64/192/256 on 512
  context tokens with 8 decode tokens; compares chunk 512/1024 on 1024 context
  tokens; then compares route batch 16/32/64/128 on 512 context tokens. A route
  batch winner replaces the current value only when it is more than 2% faster.
  Both modes apply and persist their winners. Chat is unavailable during the run
  because the engine is a serial actor.
- **Auto-tune** (~15–25 min): per-machine coordinate descent over expert-cache
  slots, dense-ahead, async FFN, expert look-ahead, `DS4_Q8_NSG`, and
  `DS4_MOE_NSG` — one model reload per candidate, RAM-gated candidate lists,
  swap-collapse detection from the temporal speed profile, and a >2% win margin
  against noise. Winners are applied and persisted.

#### On-disk locations

| Path | Contents |
|---|---|
| `~/Library/Application Support/DwarfStar/kv-cache/` | Disk-KV checkpoints (chat + HTTP server). |
| `~/Library/Application Support/DwarfStar/q4-cache/` | `.q4dense` requant caches: ~1.4 GB for the base dense-Q4 trio and larger when QKV/shared projections are included. The sandboxed app cannot write next to the GGUF. |
| `~/Library/Application Support/DwarfStar/expert-bundle/` | Expert-bundle sidecars built when the model folder is not writable (tens of GB). |
| `~/Library/Application Support/DwarfStar/dist-models/` | Files received via distributed transfer (GGUF, sidecars), keyed by SHA-256 manifest. |
| `~/Library/Application Support/DwarfStar/github-projects/` | Repositories imported by the `github_clone` tool, one `<owner>-<repo>` folder each (replaced on re-clone). |
| `~/Library/Application Support/DwarfStar/expert-usage-<model>-<agent>.json` | Per-model, per-agent usage imatrix (routing history used to pre-warm the expert cache). |
| `<resources>/gguf/` | Model download destination (`.part` files while downloading). |

### Engine Environment Variables (`DS4_*`)

Defaults below are the **engine defaults for a bare process** (CLI demo,
tests). Inside the app, the GUI settings above own the overlapping knobs.
"on"/"off" knobs marked `=0 disables` default to on. Each row states its
numerical behavior: I/O/layout changes can preserve values, fusion or MM paths
can change floating-point accumulation order, and requantization or fewer active
experts are deliberately lossy. Assume bit identity only where it is stated.
Workflow guidance ("which knob to try first") is in
[`Sources/DS4Demo/README.md`](Sources/DS4Demo/README.md).

#### Paths and integration

| Variable | Values / default | What it is / what it does |
|---|---|---|
| `DS4_ROOT` | path; default `/Users/oppog/Downloads/ds4-main` | Dev-mode project root: where non-bundled builds look for `gguf/` models and the default model path. Irrelevant for a packaged `.app`. |
| `DS4_USAGE_FILE` | path or `off`; default `<gguf>.usage.json` | Demo/CLI location of the usage imatrix (the router's historical expert choices, used to pre-warm the cache). The app ignores it and keeps per-model/per-agent files in Application Support. `off` = cold run. |
| `DS4_PROMPT_FILE` | file path; unset | Demo only: reads the prompt from this UTF-8 file, overriding the positional prompt. Equivalent to a positional `@/path/file`, but avoids shell argument splitting for long prompts. `~` is expanded. |
| `DS4_Q4_CACHE_DIR` | dir; default: next to the GGUF | Where `.q4dense` requant caches are read/written. The app sets it to Application Support (sandbox can't write next to the model). |
| `DS4_BUNDLE_DIR` | dir; default: next to the GGUF | Fallback directory for the `.expbundle` sidecar (a sibling of the GGUF is always tried first). The app sets it to Application Support. |
| `DS4_SEARCH_URL` | URL template containing `%@`; default `https://html.duckduckgo.com/html/?q=%@` | Endpoint used by the built-in `web_search` tool; the query is percent-encoded into `%@` and results are parsed from DuckDuckGo-style HTML. |
| `HF_TOKEN` | token; default: `~/.cache/huggingface/token` | Hugging Face token used by the model downloader for authenticated downloads (bare processes: demo/tests). In the app, a token saved in Settings → Hugging Face (Keychain) takes precedence over this variable. |

#### Diagnostics (demo-oriented)

| Variable | Values / default | What it is / what it does |
|---|---|---|
| `DS4_TYPES_ONLY` | `=1`; off | GGUF audit mode: prints critical tensor dtypes, tokenizer special ids, and prompt tokenization, then exits before building the decoder. |
| `DS4_DIAG` | `=1`; off | Full streaming diagnostic: prints active knobs, measures SSD bandwidth, then reports per-layer routing, expert concentration, cache allocation, and gather bandwidth vs the SSD ceiling. |
| `DS4_WARMUP` | int ≥0; `0` (`min(4,maxNew-1)` with DIAG) | Excludes the first N generated tokens from the decode profile (cold-cache and wiring costs). |
| `DS4_PROFILE_ROUTE` | `=1`; off | Splits `route/attn` into timed subphases (compressor, Q/KV, attention, output). Adds sync overhead — read ratios, not tok/s. |
| `DS4_PROMPT_MAX_CHARS` | int; `12000` | Demo only: truncation limit for `@/path/file` prompts so prompt + generation fit the demo's fixed 4096-position KV. |

#### Demo sampling and self-speculative decode

These variables affect `DS4Demo` only. The default remains greedy so historical
performance runs stay comparable.

| Variable | Values / default | What it is / what it does |
|---|---|---|
| `DS4_DEMO_TEMPERATURE` | float; `0` | Sampling temperature. `0` selects greedy argmax; values above zero enable sampling. |
| `DS4_DEMO_TOP_K` | int; `0` | Demo top-k. `0` means the sampler's full-vocabulary path; normally pair a finite value with temperature >0. |
| `DS4_DEMO_TOP_P` | float; `1` | Nucleus cutoff used when sampling. |
| `DS4_DEMO_MIN_P` | float; `0` | Relative min-p cutoff used when sampling. |
| `DS4_DEMO_REPEAT_PENALTY` | float; `1` | Repetition penalty; values >1 penalize recently emitted ids. |
| `DS4_DEMO_REPEAT_LAST_N` | int ≥0; `64` | Number of recent ids considered by the repetition penalty. |
| `DS4_SPEC_K` | int; unset/`0`/`1` = off, ≥2 enables | Self-speculative greedy window. Draft candidates are rolled back and verified in one full-configuration batch; it is automatically disabled when temperature sampling or repetition penalty is active. |
| `DS4_SPEC_DRAFT_EXPERTS` | int; `2` | Active routed experts used by the draft pass, clamped to at least 1 and below the full configured expert count. Changes draft cost, not the full-config verification rule. |

#### Expert streaming and cache

| Variable | Values / default | What it is / what it does |
|---|---|---|
| `DS4_ACTIVE_EXPERTS` | `1...6`; `6` | **Changes output.** Uses fewer routed experts per token than the trained 6: lower I/O and gather time at a quality cost. Degraded low-RAM mode / expert-I/O cost probe. |
| `DS4_EXPERT_CACHE_SLOTS` | int; `0` (off) | Per-layer LRU GPU cache for experts (~6.9 MB wired per slot per layer on 2-bit). 8 is the effective minimum; the app preset uses 22. |
| `DS4_EXPERT_CACHE_UNIFORM` | `=1`; off | Disables usage-driven slot redistribution (by default, layers with concentrated routing get more slots at the same budget). For A/B. |
| `DS4_EXPERT_LOOKAHEAD` | int; `0` | Speculative pool pre-fill of layer i+1 with its top-N usage-prior experts during layer i's compute. Hash layers 0–2 are always prefetched exactly. Requires the slot cache. |
| `DS4_EXPERT_PREAD` | `=1`; off | `pread`+`F_NOCACHE` expert reads that bypass the page cache, preventing expert churn from evicting dense weights. Often the single best knob on 16 GB. |
| `DS4_PREAD_SPLIT` | `1...8`; `1` | With expert pread: splits each slab into N concurrent 16 KB-aligned range reads to raise NVMe queue depth (the disk only peaks at ~24 requests in flight). Same bytes. |
| `DS4_EXPERT_BUNDLE` | `=1`; off | Builds/reuses the `<gguf>.expbundle` sidecar with contiguous per-expert slabs: one ~7 MB sequential read per miss instead of three scattered ones. Duplicates the expert region on disk (tens of GB). |
| `DS4_MTLIO` | `=1`; off | Metal fast resource loading from the expert bundle directly into slot-cache `MTLBuffer` destinations during decode. Prefill deliberately stays on the established parallel-`pread` path. Experimental: benchmark against pread per Mac. |
| `DS4_MTLIO_MIN_GBS` | float; `1.5` | MetalIO circuit breaker: timings are aggregated into 64 MiB windows and two consecutive windows below this bandwidth trigger permanent fallback to `pread`. Small batches and isolated stalls no longer disable MetalIO. The app preset explicitly exports `4.0`; the bare-engine default remains 1.5. |
| `DS4_POOL_INTERLEAVE` | `=0` disables; on | Slot-cache pool layout with gate/up/down contiguous per slot, so a bundle miss is ONE pread. `=0` restores the historical 3-buffer layout for parity checks. |
| `DS4_WILLNEED_EXPERTS` | `=0` disables; on | `madvise(WILLNEED)` on exactly the experts the router selected, right before the gather. Non-speculative. |
| `DS4_PREFETCH` | `=1`; off | `madvise` read-ahead of the NEXT layer's non-routed weights during the current layer's compute. Can help or hurt depending on SSD saturation. |
| `DS4_PREFETCH_EXPERTS` | int; `0` | With `DS4_PREFETCH=1`: also prefetches N likely experts from the usage prior. Speculative. |

#### Dense weights, attention, and KV

| Variable | Values / default | What it is / what it does |
|---|---|---|
| `DS4_DENSE_STREAM` | `=1`; off | Streams dense per-layer weights through a `pread`+`F_NOCACHE` 2-slot ring, one layer ahead (~300 MB staging instead of ~6 GB resident). Identical numerics; takes precedence over `DS4_RESIDENT_DENSE`. |
| `DS4_DENSE_AHEAD` | `1...3`; `1` | Read-ahead depth of the dense staging ring (requires dense streaming). Each extra slot costs ~150 MB and contends with expert I/O on the same disk. |
| `DS4_RESIDENT_DENSE` | `=1`; off | Copies ~5 GB of non-expert weights into wired buffers instead of evictable mmap views. Helps on 24/32 GB; can hurt on 16 GB. |
| `DS4_RESIDENT_COMP` | `=0` disables; on | Keeps the four NSA compressor projections (~0.6 GB, read every token on 41 of 43 layers — the densest repeat-read) resident instead of streamed. `=0` is the tight-RAM fallback. |
| `DS4_COMP_Q8` | `=1`; off (requires resident compressors) | **Lossy, experimental.** Requantizes the four resident compressor projections F16→Q8_0, roughly halving their RAM and per-token GPU reads. The first run writes a validated `.q8comp.Lx-y` cache; compare quality before enabling permanently. |
| `DS4_LAZY_IDX` | `=0` disables; on | Skips staging the indexer scoring projections (~360 MB/token) when the sparse top-K provably cannot activate at the current context size. Lossless: only never-executed work is dropped. |
| `DS4_MLOCK` | `=1`; off | Best-effort `mlock()` of hot resident buffers (expert pools, resident head, staging). Prevents the macOS memory compressor from squeezing once-per-token buffers. |
| `DS4_DENSE_Q4` | `=1`; off (requires `DS4_DENSE_STREAM=1`) | **Lossy.** Requantizes the three giant attention projections Q8→Q4_K, keeps them resident (~1.4 GB), removing ~4.6 GB/token of SSD traffic. First load writes `<gguf>.q4dense` (or `DS4_Q4_CACHE_DIR`); delete it to force a re-requant. |
| `DS4_QKV_Q4` | `=1`; off (requires `DS4_DENSE_Q4=1`) | **Lossy.** Also requantizes the q_a and kv projections to Q4_K and keeps them resident, removing the remaining medium attention slabs from the dense stream. |
| `DS4_SHARED_Q4` | `=1`; off (requires `DS4_DENSE_Q4=1`) | **Lossy.** Also requantizes the shared-expert FFN projections to Q4_K and keeps them resident, freeing disk bandwidth for the expert gather. |
| `DS4_RAW_RING` | `=1`; off | Raw KV kept in an `nSWA` (128-row) ring: constant raw-KV memory for long contexts. Does not shrink the compressed KV. |

#### Prefill

| Variable | Values / default | What it is / what it does |
|---|---|---|
| `DS4_PREFILL_CHUNK` | int; `512` | Tokens per prefill chunk (hot-reloadable). Larger chunks amortize the per-chunk full dense-weight reload on long prompts. |
| `DS4_PREFILL_UNION` | int ≥6; `192` | Max experts grouped per layer-major prefill read (hot-reloadable). Governs prefill bytes/token; `64` caused excessive rereads in the reference benchmark. This is the **bare-engine default**; the current app preset exports 256. Costs roughly 1.3 GB ×2 transient at the larger unions. |
| `DS4_PREFILL_FFN_BATCH` | `=0` disables; on | Encodes all of a group's token-FFNs into one Metal command buffer instead of one per token (the per-token sync was a dominant prefill cost). Identical numerics. |
| `DS4_PREFILL_ROUTE_BATCH` | int; `32` (`0`/`1` off) | Batches up to N consecutive tokens' route/attention into one command buffer, cutting phase-A syncs N×. Identical numerics; indexer-active tokens fall back to per-token. |
| `DS4_GPU_INDEXER_TOPK` | `=0` disables; on | Keeps long-context NSA indexer score → exact top-K mask → attention in one GPU command buffer. Removes one CPU readback/wait per active indexer layer; `=0` restores the CPU heap for parity diagnostics. |
| `DS4_PREFILL_MM` | `=1`; off (opt-in) | **Changes accumulation order** (close but not bit-identical). Routed + shared prefill FFN through matrix-matrix kernels: expert weights read once per tile for all the group's tokens. Flash-shape quants only. |

#### Kernels and numerics

| Variable | Values / default | What it is / what it does |
|---|---|---|
| `DS4_FUSED_MOE` | `=0` disables; on | Fused MoE kernels (2 dispatches instead of 5). `=0` selects the non-fused path for numerical A/B; rounding may differ. |
| `DS4_FUSED_HC` | `=0` disables; on | Fused HC-reduce tail (split+collapse+RMSNorm in one dispatch, ~170 dispatches/token saved). Same math, ±1 ulp reduction-order difference. |
| `DS4_DENSE_Q4_KERNEL` | `=0` disables; on | Dedicated single-matrix Q4_K matvec for resident dense/shared projections. Uses the same dequantization and reduction as the historical `k=1` MoE wrapper, but removes expert-ID/stride work. `=0` restores the wrapper for bit-parity A/B. |
| `DS4_FUSED_ROUTER_PROBS` | `=0` disables; on | Vectorized router `sqrt(softplus(logit))` in one dispatch instead of two full 256-value passes. `=0` restores the two-dispatch path for parity tests. |
| `DS4_FUSED_ROUTER_FINALIZE` | `=0` disables; on | Fuses router top-6 selection and bit-identical selected-weight normalization, removing one dispatch per routed layer (43 per token). |
| `DS4_FUSED_COMP_PROJ` | `=0` disables; on | Pairs the compressor KV+gate matvecs in one dispatch for both F16 and `DS4_COMP_Q8`, sharing activation reads. Same per-matrix reduction order; `0` restores two dispatches. |
| `DS4_ASYNC_FFN` | `=0` disables; on | Asynchronous routed-FFN commit: the next layer's route wait lands on an already-running FFN (+10% measured, token-identical). `=0` is the synchronous debug path. |
| `DS4_Q8_NSG` | `1...8`; `4` | Simdgroups per threadgroup for dense Q8_0 matvecs. Different values repartition the K reduction and may change the last floating-point bits; sweep 2/4/6/8 per machine and validate output when exact parity matters. |
| `DS4_MOE_NSG` | `1...8`; `4` | Simdgroups per threadgroup for routed MoE/Q4 id kernels. It partitions output rows and is tuned independently from the dense Q8 K-split. |
| `DS4_DENSE_Q4_NSG` | `1...8`; inherits `DS4_MOE_NSG` | Simdgroups per threadgroup for resident dense/grouped Q4_K projections. Row-partitioned and bit-identical; separated from routed-expert occupancy for per-GPU tuning. |
| `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` | `64/128/256/512/1024/2048/4096`; `1024` | Number of compressed KV rows above which decode attention switches from the dense scan to the sparse NSA-indexer path (below it, the top-k setup costs more than it saves). Also feeds the `DS4_LAZY_IDX` can-ever-activate proof. |

Example — the measured low-RAM profile in the demo (what the app applies by
default):

```sh
DS4_EXPERT_CACHE_SLOTS=22 DS4_RAW_RING=0 DS4_WILLNEED_EXPERTS=1 \
DS4_EXPERT_PREAD=1 DS4_DENSE_STREAM=1 DS4_MLOCK=1 \
DS4_DENSE_Q4=1 DS4_QKV_Q4=1 DS4_SHARED_Q4=1 \
DS4_PREFILL_UNION=256 DS4_PREFILL_CHUNK=512 DS4_PREFILL_ROUTE_BATCH=32 \
DS4_EXPERT_BUNDLE=1 DS4_MTLIO=1 DS4_MTLIO_MIN_GBS=4.0 \
DS4_POOL_INTERLEAVE=1 DS4_PREFILL_FFN_BATCH=1 DS4_GPU_INDEXER_TOPK=1 \
DS4_DENSE_Q4_KERNEL=1 DS4_FUSED_ROUTER_PROBS=1 \
DS4_FUSED_ROUTER_FINALIZE=1 DS4_FUSED_COMP_PROJ=1 \
DS4_EXPERT_LOOKAHEAD=0 DS4_DENSE_AHEAD=2 DS4_ASYNC_FFN=1 \
DS4_RESIDENT_DENSE=0 DS4_RESIDENT_COMP=1 DS4_LAZY_IDX=1 DS4_PROFILE_ROUTE=0 \
DS4_Q8_NSG=4 DS4_MOE_NSG=4 DS4_DENSE_Q4_NSG=4 \
  swift run DS4Demo /path/model.gguf 32 "Explain RoPE briefly."
```

### HTTP Server (Server Tab)

Configuration is set in the Server tab before pressing Start (in-memory, not
persisted). The server exposes the single shared engine loaded in Settings.

| Setting | Default | What it is / what it does |
|---|---|---|
| Host | `127.0.0.1` | Bind address. Loopback by default; put TLS in front before binding beyond loopback — the listener itself is plain HTTP. |
| Port | `8000` | TCP port. The base URL becomes `http://<host>:<port>/v1`. |
| Max tokens per response | `1024` (64–8192) | Cap used when a request body does not specify `max_tokens`. |
| CORS | off | Adds `Access-Control-Allow-Origin: *` for browser clients. |
| API key | empty (no auth) | When set, every `/v1` request must send `Authorization: Bearer <key>` (OpenAI style) or `x-api-key: <key>` (Anthropic style); otherwise 401. |

The model id is the basename of the loaded GGUF (there is no per-request model
switching). Request bodies are capped at 32 MB; requests time out after 60 s.

Per-request parameters (mapped onto the engine's sampler):

| Body field | Endpoints | Effect |
|---|---|---|
| `temperature`, `top_p`, `top_k` | all | Standard sampling controls. |
| `min_p`, `seed` | `chat/completions` | Min-p cutoff; deterministic seed. |
| `max_tokens` / `max_completion_tokens` / `max_output_tokens` | all / chat / Responses | Generation cap (falls back to the server's Max tokens). |
| `stream` | all | SSE streaming (default `false`). |
| `reasoning_effort` (chat), `reasoning.effort` (Responses), `thinking.type:"enabled"` (Anthropic) | as listed | `high`/`xhigh`/`medium` enable thinking; `low`/`minimal`/absent disable it. |
| `tools`, `messages`/`input`/`prompt` | as per API | Tool specs and conversation content. |

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-key" \
  -d '{"model":"deepseek-v4-flash","stream":true,"temperature":0.3,
       "max_tokens":256,
       "messages":[{"role":"user","content":"Hello"}]}'
```

### Distributed (Settings + Worker Tab)

The **worker** side has one setting: the listen **port** (default `9100`, not
persisted). Workers start idle; everything else is assigned by the
coordinator at connect time.

The **coordinator** side (Settings, in-memory):

| Setting | Default | What it is / what it does |
|---|---|---|
| Workers (one `host:port` per line) | `127.0.0.1:9100` | The peer list, **in layer order**: layers are partitioned contiguously across it and the last slice also runs the output head. |
| Activation bits | `32` (32/16/8) | Wire width of the HC hidden state (`nHC × nEmbd` floats) flowing through workers each token: float32, float16, or scaled int8 to cut LAN bandwidth. |
| Chunk prefill | `32` (1–256) | Tokens per distributed prefill frame. |
| Max tokens per response | `512` (16–4096) | Generation cap per distributed turn. |
| Worker-to-worker forwarding | off | Workers forward activations directly downstream instead of relaying through the coordinator; the terminal worker replies to the return address. |
| Return host / port | empty / `9099` | The coordinator's LAN IP and listener port for forwarded replies (required when forwarding is on). |
| Vertical split | off | Uses protocol-v10 expert parallelism: this Mac runs the dense backbone and workers own expert masks across all layers. Requires a wired RTT below about 1 ms. |

The ASSIGN sent to each worker carries the GGUF (path, name, and — protocol
v5 — the file itself plus the expert-bundle and `.q4dense` sidecars, streamed
in 4 MB chunks with inline SHA-256 and stored under
`Application Support/DwarfStar/dist-models`), the context size, the
expert-cache budget, the disk-KV budget, the layer slice, the usage imatrix
for pre-warming, and the coordinator's performance knobs. The forwarded knob
whitelist is a **security boundary on which environment names may be set**, not
a promise that every allowed configuration is bit-identical:
`DS4_DENSE_STREAM`, `DS4_DENSE_AHEAD`, `DS4_MLOCK`, `DS4_EXPERT_PREAD`,
`DS4_PREAD_SPLIT`, `DS4_WILLNEED_EXPERTS`, `DS4_ASYNC_FFN`,
`DS4_EXPERT_LOOKAHEAD`, `DS4_Q8_NSG`, `DS4_LAZY_IDX`, `DS4_RESIDENT_COMP`,
`DS4_FUSED_HC`, `DS4_FUSED_MOE`, `DS4_RAW_RING`, `DS4_EXPERT_CACHE_UNIFORM`,
`DS4_POOL_INTERLEAVE`, and the five `DS4_PREFILL_*` knobs. The lossy
`DS4_DENSE_Q4` travels as a typed ASSIGN field instead, together with its
cache file. I/O and scheduling knobs preserve the intended math, but fused
kernels can change floating-point reduction order and `DS4_PREFILL_MM`
explicitly changes accumulation order; therefore mixed settings are not always
bit-for-bit identical even when quality is unchanged. The coordinator also
reads `DS4ExpertCacheSlots`, `DS4DiskKV`,
`DS4DiskKVBudgetKTok`, `DS4ExpertBundle`, and `DS4DenseQ4` from the local
settings to build the ASSIGN, and persists the distributed chat's agent as
`DS4SelectedAgentDist`.

Frames are plaintext TCP with no authentication (protocol v10, magic `DS4D`):
run distributed mode only on trusted networks.

Minimal two-Mac setup: on the worker Mac open **Worker**, keep port `9100`,
press Start; on the coordinator Mac set Mode = Distributed, list
`<worker-ip>:9100` in Workers, press Connect, and chat.

### MCP Servers (MCP Tab)

External [Model Context Protocol](https://modelcontextprotocol.io) servers.
The list is persisted in `UserDefaults` under `DS4MCPServers`; import/export
uses the standard `mcpServers` JSON (Claude Desktop / Cursor / VS Code
compatible — a bare `{"<name>": {…}}` object also imports):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": { "MY_VAR": "value" }
    },
    "remote-api": {
      "url": "https://example.com/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

| Field | Transport | What it is / what it does |
|---|---|---|
| `command` | stdio | Executable spawned as a child process (resolved via PATH, including `/usr/local/bin` and `/opt/homebrew/bin`). Its presence selects the stdio transport. |
| `args` | stdio | Command-line arguments (default `[]`). |
| `env` | stdio | Extra environment variables merged over the inherited environment. |
| `url` | HTTP | Streamable-HTTP endpoint (`http://`/`https://`); its presence selects the HTTP transport. |
| `headers` | HTTP | Extra request headers sent verbatim (e.g. `Authorization`). |
| `enabled` | both | Persisted on/off switch per server (default `true`). |

Connected servers' tools appear as `mcp_<server>_<tool>` next to the
built-ins, in the chat Tool sheet and in agent tool lists.

### Model Downloads

The Download sheet drives the native resumable downloader (HTTP Range +
`.part` files) against `huggingface.co/antirez/deepseek-v4-gguf`, saving into
`<resources>/gguf/`. Authentication: token saved in **Settings → Hugging Face**
(stored in the macOS Keychain, shown only redacted, passed as the explicit
token) > `HF_TOKEN` env > `~/.cache/huggingface/token`. The sheet shows which
source is active; public models need none.

The catalog is broader than the executable backend: a successful download does
not imply that the current runtime can load the artifact.

| Target id | Contents | ~Size |
|---|---|---|
| `q2-imatrix` | Flash, 2-bit routed experts (imatrix) — the 16 GB streaming path | 81 GB |
| `q2-q4-imatrix` | Flash, q2 + last 6 expert layers q4 — higher quality | 98 GB |
| `q4-imatrix` | Flash, 4-bit routed experts — ≥256 GB RAM | 153 GB |
| `pro-q2-imatrix` | **Catalog/download only.** Pro, 2-bit imatrix; current local and distributed runtimes reject the Pro shape. | 430 GB |
| `mtp` | **Catalog/download only.** Flash MTP component; the current runtime does not load or consume it. `DS4_SPEC_K` is a separate self-speculative CLI experiment. | 4 GB |

## Packaging a `.app`

```sh
make app          # -> build/DwarfStar.app, release, ad-hoc signed
open build/DwarfStar.app
```

`make app` uses [`packaging/make_app.sh`](packaging/make_app.sh) and signs the
bundle **without App Sandbox entitlements**. This is intentional for the local,
ad-hoc package: it has normal filesystem access and `NSOpenPanel` can select a
GGUF without Powerbox/bookmark restrictions. Setting `DS4_SIGN_IDENTITY` changes
the signing identity but does not add sandbox entitlements.

For a sandboxed distribution build, generate/build the Xcode project instead;
`project.yml` applies `packaging/DwarfStar.entitlements` with network
client/server and user-selected file/bookmark capabilities. Sign that build
with Developer ID and notarize it.

## Status

Working and verified on a MacBook Pro M1 Pro 16 GB; the performance preset was
last benchmarked on 2026-07-13 with the DeepSeek-V4 Flash 2-bit imatrix model:

- model load and streaming chat on the 2-bit GGUF;
- thinking/reasoning handling;
- multi-turn KV reuse;
- text-file attachments;
- DSML tool calling with built-ins;
- agents and per-agent expert profiles;
- project library;
- tuning panel and expert cache controls;
- disk-KV cache, default on;
- fast low-RAM profile: direct expert pread, dense streaming, `mlock`, full Q4
  projection set, expert bundle + MetalIO, and 22-slot expert cache;
- native HTTP server, verified for OpenAI `chat/completions` and Anthropic
  `messages` streaming.

Implemented, still requiring broader on-device validation:

- `/v1/responses`;
- isolated-context sub-agents with main-KV snapshot/restore and per-file/project
  content-keyed KV cache;
- newer default agents: Orchestrator, LaTeX, Documentation;
- distributed inference protocol, worker/coordinator, UI, and benchmark;
- vertical expert-parallel chat/benchmark;
- broader numerical parity and multi-Mac distributed runs.

## Known Limits

On 16 GB systems, decode is I/O-bound. That is the physics of streaming a 284B
MoE model from SSD; distributed inference is the intended mitigation. KV memory
still scales with context, so keep context modest on low-RAM systems. The
raw-KV ring reduces only the raw cache, not every compressed KV row. Server,
distributed inference, diagnostics, and benchmark are native in-process panels;
no subprocess-driven UI panels remain.
