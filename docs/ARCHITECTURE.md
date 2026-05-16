# Architecture

This document explains the overall design. Read it before opening any
specific source file — it'll tell you what each file is for and where
it sits in the dependency graph.

## What the model is

DeepSeek-V4 is a Mixture-of-Experts transformer with several non-standard
parts. The Swift port mirrors the reference 1:1 (see
[`PYTHON-MAPPING.md`](PYTHON-MAPPING.md)).

Notable architecture choices:

- **MLA**: Multi-head Latent Attention. Low-rank Q (`wq_a → q_norm → wq_b`),
  shared KV projection + `kv_norm`, grouped low-rank O (`wo_a → wo_b`,
  with `n_groups = 8`). Per-head `attn_sink` scalar.
- **Sliding window + sparse attention**: window size 128, plus optional
  compressed-token attention via per-layer `compress_ratio` (0 = pure
  window, 4 = indexed compression, 128 = heavy compression).
- **Compressor**: learned gated softmax pooling over consecutive tokens,
  with overlap when `ratio == 4`. Maintains state buffers
  (`kv_state`, `score_state`) across decode steps.
- **Indexer**: top-k learned KV-position selector. Runs its own Compressor
  + Hadamard rotation + FP4 quant before scoring. Used by `ratio == 4`
  layers.
- **Hyper-Connections (HC)**: hidden state is expanded into `hc_mult = 4`
  parallel copies. Each block wraps the attention and FFN with `hc_pre`
  (Sinkhorn-normalized collapse to 1) and `hc_post` (expand back to 4).
- **MoE FFN**: `sqrtsoftplus` gating with bias, top-2 routed experts +
  one shared expert. Hash routing for `n_hash_layers` early layers.
- **YaRN RoPE**: applied only to the last `rope_head_dim = 64` of each
  head, with frequency-scaling correction for long context.
- **MTP** (Multi-Token Prediction): trailing speculative blocks for
  speed-up. One MTP block by default.

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

## Forward pass at decode time (single-token)

For each generated token the CLI calls `model.forward(inputIds:startPos:)`
once. Inside:

1. `ParallelEmbedding.lookup` reads one row of the embed table → `[1, dim]`.
2. `hc_expand` tiles it to `[1, hc, dim]`.
3. For each layer `i` (run sequentially, layers are not parallelized):
   - `HyperConnections.pre` collapses `[1, hc, dim] → [1, dim]` for the
     attention sublayer (Sinkhorn-normalized weighted sum).
   - `attnNorm` → `MLA.forward` (which writes one row into the
     sliding-window KV cache at slot `startPos % win` and may emit one
     compressed token).
   - `HyperConnections.post` expands the attention output back to
     `[1, hc, dim]`.
   - Same `pre`/`post` chain around `MoEFFN.forward`.
4. `ParallelHead` applies the final HC collapse and the LM head matmul,
   producing `[1, vocab]` logits.

`Sampler.sample(_:history:options:)` then picks one token id.

OS page-cache behaviour matters: layers are touched in strict order, so
shards aligned to layer boundaries (which the converter does by default)
let the kernel prefetch the next shard while we're still on the current
one.

## Dtypes and conversions

See [DTYPES.md](DTYPES.md) for full bit layouts, conversion math, and
fusion details.

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
          • loop (≤ 8 roundtrips per turn)
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
