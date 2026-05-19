# Architecture

This document explains the overall design of the *engine* + the
*desktop app* built on top of it. It tells you what each file is for
and where it sits in the dependency graph.

For the **model itself** — what MLA / MoE / HC / Compressor / Indexer
do, the forward pass, the dtypes per component, the weight naming, the
KV cache lifecycle — open [`MODEL.md`](MODEL.md) (or
[`MODEL.it.md`](MODEL.it.md) in italiano). That's the canonical
reference; this file no longer duplicates it.

## What the model is (one paragraph)

DeepSeek-V4 is a Mixture-of-Experts transformer with several
non-standard parts: **MLA** (low-rank Q + shared KV + grouped O,
plus per-head attention sinks), **sliding-window sparse attention**
with optional per-layer KV compression (`compress_ratios`), a
**Compressor** that pools consecutive tokens into a single compressed
KV, an **Indexer** that learns which compressed positions to attend
to, **Hyper-Connections** (the residual stream is held as `hc_mult =
4` parallel copies with Sinkhorn-normalised mixing), an **MoE FFN**
with `sqrtsoftplus` top-2 routing + a shared expert, **YaRN-corrected
RoPE** on the trailing dims of each head, and a trailing **MTP** block
for speculative decoding. The Swift port mirrors the reference Python
in `Reference/inference/model.py` 1:1 — see
[`PYTHON-MAPPING.md`](PYTHON-MAPPING.md) for the line-by-line table.

## Data flow

```
                          HuggingFace V4 release
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │   converter CLI      │
                       │   (Sources/converter)│
                       └──────────┬───────────┘
                                  │  rename + fuse FP8/FP4 scales
                                  │  layer-aligned sharding
                                  │  model.safetensors.index.json
                                  ▼
                          BF16 (or kept) sharded
                          checkpoint on disk
                                  │
   prompt string ──────►  tokenizer.encode  ──── ids ──┐
                          (Sources/.../BPETokenizer)   │
                                                       ▼
                  ┌────────────────────────────────────────┐
                  │  Transformer.forward(ids, startPos)    │
                  │  (Sources/DeepSeekKit/Model.swift)     │
                  │                                        │
                  │   embedding lookup                     │
                  │     → HC expand to [B,S,hc,dim]        │
                  │     → for each Block:                  │
                  │         HC.pre → norm → MLA → HC.post  │
                  │         HC.pre → norm → MoE → HC.post  │
                  │     → ParallelHead (HC collapse + LM head)│
                  └──────────────┬─────────────────────────┘
                                 │ logits [B, vocab]
                                 ▼
                       Sampler.sample(opts)
                       (temperature, top-K, top-P, Gumbel-max)
                                 │
                                 ▼
                          next token id
                                 │
                                 │  loop until EOS or maxTokens
                                 ▼
                          tokenizer.decode → stdout
```

Inside each Block, MLA reads/writes the per-layer KV cache, and the
Compressor (if present) writes compressed tokens into the trailing
slice of the same cache.

## Memory model

See [MEMORY.md](MEMORY.md) for the full mmap walkthrough, KV cache
lifecycle, and per-phase footprint.

The checkpoint is **memory-mapped**, never copied to RAM in bulk.

- `SafeTensorsFile.init(url:)` does `mmap(MAP_PRIVATE)` on each shard
  and wraps the result as an `MTLBuffer` via
  `makeBuffer(bytesNoCopy:length:options:deallocator:)`. The OS pages
  weight pages in on first access and evicts cold pages under memory
  pressure.
- On Apple Silicon (unified memory), the GPU reads the mmapped pages
  directly with zero CPU→GPU copy.
- `Tensor` is `MTLBuffer` + offset + shape + dtype. Tensors built from
  safetensors share the underlying mmap; tensors built from computation
  (`Tensor.empty(...)`) get their own fresh `MTLBuffer`.
- KV cache is allocated per-layer as a single `Tensor` of shape
  `[maxBatch, windowSize + maxSeqLen/ratio, headDim]`. Sliding-window
  writes are ring-buffer updates; compressor writes go into the
  trailing slice.

This is the only way a 140 GB (V4-Flash) or 600 GB (V4-Flash BF16) /
~900 GB (V4-Pro) checkpoint becomes touchable on a Mac with 192–512 GB
of unified memory.

## Module dependency graph

Read top-down: lower modules depend on upper.

```
Foundation, Metal (system)
        │
        ▼
Device           ← MTLDevice + queue + lazy default.metallib
        │
        ▼
Tensor           ← MTLBuffer + dtype + shape; built on Device.shared
        │
        ▼
SafeTensorsFile  ← mmap-backed reader; returns Tensors
SafeTensorsWriter ← streaming writer (used only by converter)
        │
        ▼
WeightLoader     ← indexes a directory of *.safetensors shards
        │
        ▼
Layers/          ← Linear, RMSNorm, RoPE, Hadamard, ActQuant, …
                   each a thin Swift wrapper over a Metal kernel
                   in Kernels/, plus a CPU reference in the same file
                   for testing
        │
        ▼
Layers/Compressor, Indexer, MLA, MoEFFN, MTPBlock, HyperConnections
                 ← composition of multiple kernels
        │
        ▼
Model.swift      ← ParallelEmbedding, ParallelHead, Transformer
                   = whole forward pass
        │
        ▼
Generation,      ← inference loop
Sampling,
Encoding/
        │
        ▼
Sources/deepseek (CLI)
Sources/converter (CLI)
```

`Reference/` is read-only context; `Tests/` validates the Layers/Kernels
against pure-Swift CPU references.

## Where Metal kernels sit vs Swift wrappers

For every non-trivial kernel there is a pair:

```
Sources/DeepSeekKit/Kernels/foo.metal        ← MSL kernel
Sources/DeepSeekKit/Layers/Foo.swift         ← Swift wrapper:
                                                - constructs the MTLComputePipelineState
                                                - exposes an `apply(...)` or `callAsFunction(...)`
                                                - includes `referenceCPU(...)` pure-Swift
                                                  for testing
Tests/DeepSeekKitTests/FooTests.swift         ← XCTest comparing Metal vs CPU
```

This pattern is the convention, look for it in every new kernel.

## Forward pass at decode time

The single-token decode path is documented in
[`MODEL.md` §5](MODEL.md#5-the-full-data-flow-at-decode). Short
version: `embed.lookup` → `hc_expand` → for each layer (HC.pre →
attn_norm → MLA → HC.post → HC.pre → ffn_norm → MoE → HC.post) →
`ParallelHead` (HC collapse + LM head) → sampler picks the next id.

Layers are run **sequentially** with one command buffer per layer
(commit + wait between blocks). OS page-cache behaviour matters: when
the converter shards by layer boundary (the default), the kernel
prefetches the next shard while the GPU is reading the current one,
and the streaming-pool loader rotates pages with `MADV_DONTNEED` after
each block.

## Dtypes and conversions

See [`DTYPES.md`](DTYPES.md) for full bit layouts, conversion math,
and fusion details; [`MODEL.md` §6](MODEL.md#6-numeric-data-types-per-component)
for the per-component dtype map.

Apple Silicon natively supports `F32`, `F16`, `BF16` (Metal 3+), and
integer types. It does **not** have native FP8 / FP4 / E8M0 arithmetic.

The converter has three modes:

| `--target-dtype` | Effect |
|---|---|
| `keep` | Leave non-native dtypes as-is. Inference dispatches to `fp8_gemm` / `fp4_gemm` (with shader-side dequant). Smaller disk. |
| `bf16` (default) | Fuse FP8/FP4 + E8M0 scale → BF16 single tensor. Inference uses `gemm_bf16` (native simdgroup matrix). ~4× disk for FP4 experts. |
| `f16` | Same as bf16 but with F16. |

See [`USAGE.md`](USAGE.md) for trade-offs, and `Sources/converter/main.swift`
for the implementation.

## Desktop app architecture

`DeepSeekKit` is the inference library. The macOS app
(`Sources/DeepSeekUI/`) sits on top of it and adds: chat history,
multiple backends (local + remote OpenAI-compatible), MCP tool
servers, agent presets, sub-agent delegation, and the SwiftUI
surface.

### State graph

Three long-lived `@StateObject` / `let` singletons + a handful of
libraries hang off the App scene:

```
DeepSeekUIApp
 │
 ├── InferenceService            (non-Observable, internal serial queue)
 │     ▲ owns Transformer + Tokenizer + CacheImage shadow
 │
 ├── ModelLibrary                (recents persistence)
 │     ▲
 │     └── ModelState            (load lifecycle, Published status)
 │
 ├── DocumentLibrary
 ├── ProjectLibrary
 ├── AgentLibrary
 ├── MCPServerLibrary            (mcp.json on disk)
 │     ▲
 │     └── MCPClientPool         (live JSON-RPC clients, Published per-server status)
 │
 ├── NativeToolHost              (DeepSeekTools registry + PlanStore)
 │     ▲ owns ToolRegistry + bridges PermissionDelegate to a SwiftUI sheet
 │     └── PermissionStore       (durable always-allow/deny defaults)
 │
 ├── SkillLibrary                (skill catalogue, built-in + custom)
 ├── SlashCommandLibrary         (composer-intercepted commands)
 ├── ThemeStore                  (active theme + appearance pref)
 ├── KeybindingStore             (per-user keyboard-shortcut overrides)
 │
 └── OpenRouterCatalog           (cached /models response)
```

`ChatStore` is constructed inside `ChatContainer` (one per app run)
and holds **references** to every singleton above plus
`ModelState`. It owns the conversations list + selection + the
per-conversation `phases` map the chat surface reads. There's no
explicit dependency injection container — the App scene's `init` is
the wiring point.

### Backend dispatch

The chat surface treats inference as an event stream
(`AsyncThrowingStream<GenerationEvent, Error>`), regardless of where
the events come from. `ChatStore.send` branches at the top:

```
ChatStore.send(text, mode, options, maxTokens)
 │
 ├── modelState.loadedEndpoint == .openRouter(modelID) ?
 │      │
 │      └─ YES → sendRemote(text, modelID, mode, options, maxTokens)
 │                │
 │                ▼
 │            runRemoteLoop:
 │              • OpenAI body { messages, tools, sampler }
 │              • OpenRouterClient.streamChatCompletion (SSE)
 │              • accumulate delta.content / reasoning_content / tool_calls
 │              • finalize on .done
 │              • if final.toolCalls non-empty: invoke via mcpPool,
 │                splice tool messages, re-fire HTTP, loop (≤ 8)
 │
 └─ NO (local or none) →
        buildPromptTokens
          ▼
        service.generateForConversation(promptTokens, …)
          ▼
        apply each event (token / done / progress …)
          ▼
        on .done with tool calls:
          • per-call: mcpPool.invokeQualified or
                      executeSubAgentDelegation
          • tokenizeToolOutputsDelta → fast-path delta append
          • re-issue generateForConversation
          • loop (≤ 21 roundtrips per turn)
```

Both branches surface the same `GenerationPhase` enum
(`.idle / .prefilling / .streaming(buffer, status, metrics) / .error`)
so the UI doesn't care which path produced the events.

### Local-only state that doesn't exist for remote

| Local | Remote (OpenRouter) |
|---|---|
| `CacheImage` shadow → fast-delta tokenization | n/a (cache is server-side) |
| `Conversation.encodedTokens` | always `nil` |
| `Conversation.pendingTurn` (crash resume) | always `nil` (re-send to resume) |
| `runSubAgentToCompletion` (delegation loop) | n/a — delegate schema is not injected |
| KV snapshot/restore via `Transformer.snapshotKVCache` | n/a |

The chat UI reads these fields opportunistically — when they're
`nil` the relevant affordances (resume banner, delegation chain)
just collapse.

### Where MCP plumbing lives

`MCPClient` is one persistent JSON-RPC connection over stdio to a
child process spawned through `/usr/bin/env`. `MCPClientPool` keys
clients by `MCPServerConfig.id`, syncing the live set against the
library every time the library is mutated. The pool exposes:

- `allTools()` — flattened catalogue across every connected server.
- `toolSchemasJSON(allowedNames:)` — the DSML JSON the local chat
  template expects (used by the agent filter to drop disallowed
  tools).
- `invokeQualified(_:argsJSON:)` — routes a `<server>__<tool>` call
  to the right client and returns the flattened textual output the
  chat splices back into the prompt.

Both backends call the same `invokeQualified`. The translation to
each backend's wire shape happens upstream — the local path emits
DSML tool blocks, the remote path emits the OpenAI `tools` array.

### Agents, delegation, KV snapshots

`AgentConfig` is a preset (system prompt + tool allowlist +
sampling defaults + thinking mode + cosmetics) that hides under
`Conversation.agentID` when attached. `ChatStore.send` resolves it
to prepend the agent's system prompt and filter the MCP tools
before composing the request.

When more than one agent is registered, the local chat injects a
synthetic `__delegate_to_agent` schema with a roster of every other
agent. The model can call it with `{ agent_name, task }`;
`ChatStore.executeSubAgentDelegation` dispatches through
`dispatchDelegation` → `runSubAgentToCompletion` →
`runSubAgentToCompletionInner`. Each level snapshots the live KV
cache via `InferenceService.beginDelegation` (a single-process map
of UUID → `KVCacheSnapshot`) and restores it on the way back up
through `endDelegation`. That's why the host doesn't pay a cold
re-prefill when the sub-agent returns.

Nesting is capped at 3 levels; cycle prevention is a `chain: [UUID]`
threaded through every nested call so an agent already on the stack
can't be re-entered.

### Native tool subsystem (DeepSeekTools)

Separate target. Owns the protocols + the built-in tool catalogue
the model can invoke directly without going through MCP.

```
ChatStore.send (local or remote)
    │
    │   tools available to the model =
    │     MCP catalogue (filtered by agent.allowedToolNames)
    │   ∪ NativeToolHost.registry.availableSchemas(mode:)        [TODO §8 wire]
    │
    ▼
Model emits tool call { name, arguments }
    │
    ▼
ChatStore tool-call branch
    │
    ├── name has MCP "server__tool" prefix → MCPClientPool.invokeQualified
    └── otherwise → NativeToolHost.dispatch
                       │
                       ▼
                  ToolRegistry.dispatch(name, input, context)
                       │
                       ▼
              ┌─────────────────────────────────────────┐
              │  Mode filter                            │
              │    plan + (mutating | dangerous) → DENY │
              └─────────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────────────────────────────┐
              │  PermissionStore durable default        │
              │    alwaysAllow → run                    │
              │    alwaysDeny  → return ToolError.denied│
              │    ask         → ↓                      │
              └─────────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────────────────────────────┐
              │  Session cache (in-actor)               │
              │    cached → run                         │
              │    miss   → ↓                           │
              └─────────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────────────────────────────┐
              │  PermissionPromptView (SwiftUI modal)   │
              │    Deny | Allow once | Always allow     │
              └─────────────────────────────────────────┘
                       │
                       ▼
              ToolResult { output, metadata }
                       │
                       ▼
ChatStore splices output into the next prompt and re-fires generate.
```

Today the `MCPClientPool.invokeQualified` branch is wired through
both backends; the `NativeToolHost.dispatch` branch is scaffolded
on the GUI side but the inference loop doesn't yet merge native
tool schemas into the request body — see TODO §8 "Wire tool
registry into InferenceService". Once that lands, both branches
share the same `phases[id]` lifecycle the chat surface already
reads.

`PlanStore` is the shared backing for the three planning tools
(`plan`, `task`, `todo`) — it lives inside `NativeToolHost` so
all three see the same actor-isolated state across calls.

### Slash commands

`/`-prefixed text in the composer is intercepted by
`SlashCommandLibrary` before the inference loop sees it. Built-in
commands include `/mode plan|build`, `/tools`, `/permissions`,
`/skill <name>`, `/theme`, `/clear`. The slash command palette
(`SlashCommandPaletteView`) opens inline; selecting an entry
either replaces the draft with an expansion or fires an action
(toggle mode, open a Settings sheet, etc.) without sending a
message to the model.

### Chat template dispatcher

V4 has a one-off chat format (`EncodingDSV4`). Every other model
ships its own Jinja2 template in `tokenizer_config.json.chat_template`
(Llama / Mistral / Qwen / ChatML / hundreds of HF checkpoints).
Hard-coding V4 everywhere would have made non-V4 local backends
impossible.

The `ChatTemplate` protocol resolves the split:

```
TokenizerLoader.load(tokenizerDir:)
   │
   ├── inspects the directory
   │     · tokenizer.json + DeepSeek vocab signature → BPETokenizer
   │     · tokenizer.model (SentencePiece protobuf) → SentencePieceTokenizer
   │     · vocab.txt                               → WordPieceTokenizer
   │
   └── inspects tokenizer_config.json.chat_template
         · DSV4 marker (or default for DeepSeek vocab)  → DSV4Template
         · Jinja string                                 → JinjaChatTemplate
   ▼
InferenceService.{ _tokenizer, chatTemplate }
```

`InferenceService.chatTemplate` is exposed publicly so future
callers (sub-agent renderers, remote prompt builders that mirror
the model's official format) can render messages without
assuming V4. The chat-flow side currently still calls
`EncodingDSV4.*` directly for V4 chats (faster path, no Jinja
interpretation cost); the dispatcher is the bridge for "another
model is loaded locally" scenarios.

`JinjaTemplate` is a hand-rolled Swift subset of Jinja2: variable
interpolation, control flow (`{% for %}` / `{% if %}` /
`{% elif %}` / `{% else %}`), the filters HF templates actually
use (`trim`, `length`, `tojson`, `default`, …), and the
`raise_exception` builtin templates use to abort on malformed
input. `{% macro %}` / `{% set %}` / template inheritance are
deferred — none of the chat templates in the wild need them.

### GGUF reader (MVP)

`GGUFFile` parses the GGUF v2/v3 metadata format used by
llama.cpp. Today it's read-only and only supports the
pass-through dtypes (`F32 / F16 / BF16 / I32 / I8`); quantized
tensors (`Q4_0 / Q4_K_M / Q8_0 / …`) raise
`GGUFError.unsupportedType` and are left to the caller as raw
bytes via `info(name:)`.

This exists for two reasons:

1. **Non-V4 local backends.** Combined with the chat template
   dispatcher + the tokenizer dispatcher, the engine has the
   pieces it needs to consume a Llama/Mistral GGUF — once the
   matching Metal dequant kernels land.
2. **Diagnostic / convert path.** The converter can already read
   non-quantized GGUF metadata. Writing a GGUF output (so a V4
   checkpoint becomes loadable by llama.cpp) is the symmetric
   direction and lives on the roadmap.

Full status + roadmap in [`GGUF.md`](GGUF.md).

### Crash recovery (local only)

Every local `Send` writes a `PendingTurn { promptTokens,
generatedTokens, mode }` snapshot onto the active `Conversation`.
The chat UI shows a Resume banner whenever that field is non-nil
and the chat is idle. Clicking Resume feeds the same prompt +
already-sampled ids back through `service.generateForConversation`
so the bubble keeps extending exactly where it left off.

The snapshot is also re-armed during a tool-call continuation
(`runToolCallsAndContinue`) so a crash between two iterations of the
tool loop is still resumable.

## What's not implemented

See [`ROADMAP.md`](ROADMAP.md) for the feature roadmap and
[`PERFORMANCE.md`](PERFORMANCE.md) for the planned perf optimizations.
Headline items:

- **Multi-token decode** works end-to-end (Tier 1 complete).
- **act_quant QAT noise** on non-rope KV dims is skipped (Tier 3 deferred:
  needs a strided Tensor view).
- **`cast_e2m1fn_to_e4m3fn`** in the converter is deferred (Tier 3:
  unused with the default BF16 path).
- **Numerical validation vs Python** has a hand-tested infrastructure
  but not an automated harness (Tier 3: requires PyTorch + CUDA).
