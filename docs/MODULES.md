# Modules reference

Per-file index of `Sources/`. Each entry: purpose, key public API,
dependencies, and any cross-references. Use as a directory index when
deciding which file to open.

For Metal kernels see [`KERNELS.md`](KERNELS.md). For the
Python correspondence see [`PYTHON-MAPPING.md`](PYTHON-MAPPING.md).
For contributor recipes (how to add a kernel/layer/test) see
[`DEVELOPING.md`](DEVELOPING.md). For the test inventory see
[`TESTING.md`](TESTING.md).

---

## `Sources/DeepSeekKit/` (top-level)

### `Device.swift`
Singleton wrapping `MTLDevice`, command queue, and the lazy-loaded
`default.metallib` (compiled by `MetalLibPlugin`). The converter only
touches `Device.shared.mtl` for `MTLBuffer` creation — the library is
loaded only on first `makePipeline` call.

Public: `Device.shared`, `Device.shared.mtl`, `Device.shared.queue`,
`Device.shared.library`, `Device.shared.makePipeline(_:)`.

### `Config.swift`
`ModelConfig`: Codable struct mirroring `ModelArgs` from
`Reference/inference/model.py:34`. Field names use Python snake_case via
`CodingKeys`. `ModelConfig.load(from:)` reads `config.json`.

Used by: every layer constructor, the assembly loader, the CLI.

### `Tensor.swift`
`Tensor` = shape + dtype + `MTLBuffer` + offset. Row-major. No autograd,
no broadcasting. `Tensor.empty(shape:dtype:)` allocates a fresh
`MTLBuffer`; `Tensor.from(bytes:shape:dtype:)` copies host bytes into a
new `MTLBuffer`; safetensors return tensors that share an mmapped
buffer.

`DType` enum: `f32`, `f16`, `bf16`, `i32`, `i8`, `fp8E4M3`, `fp4E2M1`,
`e8m0`. `Tensor.reshape(_:)` returns a same-buffer view.

`Tensor.toFloatArray()` is a host-side debug helper; supports
`f32`/`f16`/`bf16` only.

### `Quantization.swift`
Constants for the FP8/FP4/E8M0 block layouts plus scalar host-side
`dequantE4M3` / `dequantE2M1` / `dequantE8M0` helpers used by tests and
the converter. Documents the exact safetensors layout the loader
expects.

### `YaRN.swift`
`YaRN.precomputeFreqsCis(...)` returns a flat `[seqlen][rope_dim/2][2]`
(cos, sin) array. Pure Swift port of
`Reference/inference/model.py:199-229`. Called once per layer by `MLA`.

### `SafeTensors.swift`
`SafeTensorsFile`: mmap-backed reader. `init(url:)` opens the file with
`MAP_PRIVATE`, wraps the whole mapping as one `MTLBuffer` via
`makeBuffer(bytesNoCopy:)`. `load(_:)` returns a `Tensor` referencing
the shared buffer with the right offset.

The mmap is deallocated when the buffer is released (closure passed as
`makeBuffer` deallocator).

### `SafeTensorsWriter.swift`
Streaming writer used by the converter. Three source types:
- `.data(Data)` — already-in-memory bytes
- `.file(url:offset:byteCount:)` — copy directly from another file in
  64 MB chunks
- `.compute(byteCount:closure)` — lazy producer, called when the writer
  reaches that tensor

### `WeightLoader.swift`
Walks a directory of `.safetensors` shards, builds a flat name → shard
index, exposes `load(_:)` and `tryLoad(_:[fallbackNames])`. Skips LFS
pointer files (< 1 KB). Used by `Transformer.load(from:)`.

### `Tokenizer.swift`
Protocol `Tokenizer { encode/decode/bosId/eosId }` and
`TokenizerLoader.load(from:)` that constructs a `BPETokenizer` from a
`tokenizer.json` file.

### `BPETokenizer.swift`
Byte-level BPE compatible with HuggingFace `tokenizer.json`. Parses
`model.vocab`, `model.merges`, `added_tokens`, plus the pre-tokenizer
regex. GPT-2 byte-to-unicode map for UTF-8 round-trip. Greedy
lowest-rank merge BPE.

### `KVCache.swift`
Allocators for per-layer KV caches and a `CacheBank` aggregating them.
Currently unused at runtime — MLA holds its `kvCache` field directly,
populated in assembly.

### `Sampling.swift`
- `Sampler.argmax(_:)` — GPU-side reduction, returns a single Int.
- `Sampler.applyTemperature(_:)` — in-place GPU scaling.
- `Sampler.sample(_:history:options:)` — full pipeline (temperature →
  repetition penalty → top-K → top-P → Gumbel-max multinomial), done
  host-side after a single CPU-GPU sync.
- `SamplingOptions` struct carries the parameters + RNG state across
  decode steps.

### `Generation.swift`
`Generator` + `GenerationOptions` are the OO wrapper around the
generation loop. **Currently unused** — the CLI in
`Sources/deepseek/main.swift` open-codes the prefill+decode loop. Kept
as a future API surface.

### `Model.swift`
Three classes:

- `ParallelEmbedding` — embed table + `lookup([Int32])` returning
  `[N, dim]`.
- `ParallelHead` — final RMSNorm + HC head collapse + LM head matmul,
  producing logits on the LAST sequence position only.
- `Transformer` — assembled model; `forward(inputIds:startPos:)` runs
  embed → HC expand → all blocks → head, returning `[B, vocab]` logits.

### `Assembly.swift`
Two factory methods on `Transformer`:

- `Transformer.randomInit(config:)` — builds every module with small
  random F32 weights. End-to-end smoke testing without weights on
  disk.
- `Transformer.load(config:from:)` — walks the canonical V4 weight
  name tree, pulls each tensor via `WeightLoader`, builds the
  modules. Falls back to random init for missing names and prints a
  summary.

The canonical weight name list lives in the docstring.

---

## `Sources/DeepSeekKit/Layers/`

Each Layer file is a Swift wrapper around one or more Metal kernels in
`Kernels/`. Convention: every layer has a `callAsFunction(...)` or
similar public method, and (where it makes sense) a `referenceCPU(...)`
pure-Swift implementation used by tests.

### `Linear.swift`
`Linear(in:out:weight:scale:)`. Dispatches based on `weight.dtype`:
- `.bf16` → `gemm_bf16_to_f32` (or `gemm_f32_bf16_to_f32` if input is f32)
- `.f32`  → `gemm_f32_to_f32`
- `.fp8E4M3` → act_quant + `gemm_fp8_to_f32`
- `.fp4E2M1` → act_quant + `gemm_fp8_fp4_to_f32`

Output is always `f32`.

### `RMSNorm.swift`
`RMSNorm(weight:eps:)` + `callAsFunction(_:in:)`. One kernel
(`rmsnorm_f32`), one threadgroup per row.

### `RoPE.swift`
`RoPE(ropeHeadDim:freqs:)`. `apply(_:startPos:inverse:in:)` does
in-place rotary on the trailing `rope_head_dim` of each head. Freqs are
pre-baked by `YaRN.precomputeFreqsCis` at model construction. The
`inverse: true` path is used for the output projection's de-rotation.

### `Hadamard.swift`
`Hadamard.apply(_:in:)` — in-place Walsh-Hadamard transform on the last
axis. Requires power-of-2 dim. Used by Indexer and Compressor (when
`rotate == true`).

### `ActQuant.swift`
`ActQuant(format: .fp8 | .fp4)`. `quant(_:inplace:in:)`. Block-wise
activation quantization, with `inplace=true` performing a round-trip
(dequant immediately) so the buffer stays f32 — used for QAT noise
injection.

### `Elementwise.swift`
Bundle of small elementwise ops as static methods: `siluMul`, `axpy`,
`scale`, `addInPlace`. Used everywhere by Compressor / MoE / MLA /
HyperConnections.

### `SparseAttention.swift`
`SparseAttention.apply(q:kv:sink:topkIdxs:scale:in:)` — sparse multi-
head attention with FlashAttention-style online softmax and KV gather
by topk index. One thread per `(b, m, h)`.

### `HCSinkhorn.swift`
`HCSinkhorn(hcMult:sinkhornIters:hcEps:)`. `split(mixes:hcScale:hcBase:in:)`
produces `(pre, post, comb)` from a `[N, mix_hc]` mixing tensor.
Used inside `HyperConnections.pre`.

### `HyperConnections.swift`
`HyperConnections(config:dim:)`. Two methods:
- `pre(x:hcFn:hcScale:hcBase:in:)` → `(y[N, dim], post, comb)`. Collapses
  `[N, hc, dim]` to `[N, dim]` for the sublayer input.
- `post(x:residual:post:comb:in:)` → `[N, hc, dim]`. Re-expands.

### `SoftmaxAxis.swift`
`SoftmaxAxis.apply(_:axis:in:)` — softmax along any axis of an N-D
tensor. Used by Compressor's per-window softmax and indirectly by
Indexer.

### `TopK.swift`
`TopK.apply(_:k:in:)` → `(values, indices)`. Top-K along last axis.
In-register heap; max k = 32.

### `MoEDispatch.swift`
`MoEDispatch.prepare(...)` builds the host-side permutation tables from
gate output. `MoEDispatch.gather(...)` groups tokens by expert.
`MoEDispatch.scatter(y:outs:plan:in:)` does the weighted sum of
expert outputs back into the dense `[N, dim]` output.

### `OverlapTransform.swift`
`OverlapTransform.apply(_:padValue:in:)` — Compressor's overlap shuffle
(`[B, S, ratio, 2D] → [B, S, 2*ratio, D]`).

### `Einsum.swift`
Two specialized einsums used by V4 attention:
- `Einsum.bshdBtd(q:kv:in:)` → `[B, S, H, T]` (Indexer score)
- `Einsum.bsgdGrd(o:woA:in:)` → `[B, S, G, R]` (MLA grouped output)

### `AttentionIndices.swift`
Pure host-side helpers (no kernel):
- `slidingWindow(windowSize:batch:seqlen:startPos:)` → `[Int32]` window
  topk indices for both prefill (`startPos == 0`) and decode (wrap).
- `compressed(ratio:batch:seqlen:startPos:offset:)` → compressed-token
  topk indices for layers without an Indexer (`ratio == 128`).

### `Compressor.swift`
`Compressor(config:compressRatio:...)`. `callAsFunction(_:startPos:in:)`
returns the compressed KV when one is emitted, else nil.

Prefill (`startPos == 0`) processes the whole sequence in one shot.
Decode accumulates per-token into `kvState`/`scoreState` buffers and
emits one compressed token every `compressRatio` steps. Overlap
(`compressRatio == 4`) maintains a double-window state via
`compressor_overlap_concat_f32` + `compressor_state_shift_copy_f32`.

### `Indexer.swift`
`Indexer(config:compressRatio:wqB:weightsProj:compressor:kvCache:)`.
`callAsFunction(_:qr:startPos:offset:in:)` → `[B, S, K]` Int32 topk
indices into the compressed-KV cache. Used only by layers with
`compress_ratio == 4`.

### `Attention.swift`
`MLA` struct + `callAsFunction(_:startPos:in:)`. The full attention
forward (prefill or decode):

1. Low-rank Q (`wq_a → q_norm → wq_b → rsqrt re-norm`) + RoPE on rope tail
2. KV (`wkv → kv_norm`) + RoPE on rope tail
3. Window topk + optional compressed topk (Indexer or
   `AttentionIndices.compressed`)
4. KV cache write (full prefill, cutoff/wrap for `seqlen > window`, or
   ring-buffer single row for decode)
5. Compressor call (writes compressed tokens into the trailing KV cache slice)
6. `SparseAttention.apply`
7. Inverse RoPE on output
8. Grouped output via `Einsum.bsgdGrd` + `wo_b`

### `DecoderLayer.swift`
`Block(layerId:config:attn:ffn:...)`.
`callAsFunction(_:startPos:inputIds:in:)`:

```
residual = x
(y, post, comb) = HC.pre(x, hc_attn_fn, ...)
y = attn_norm(y)
out = attn(y, startPos)
x = HC.post(out, residual, post, comb)

residual = x
(y, post, comb) = HC.pre(x, hc_ffn_fn, ...)
y = ffn_norm(y)
out = ffn(y, inputIds)
x = HC.post(out, residual, post, comb)
return x
```

### `MoE.swift`
Three types in one file:

- `Gate(config:layerId:weight:bias:tid2eid:)` — top-k gating with
  `sqrtsoftplus` / `sigmoid` / `softmax` score func via function
  constants, plus the hash-routing branch for early layers.
- `Expert(w1:w2:w3:swigluLimit:)` — SwiGLU FFN expert.
- `MoEFFN(config:gate:experts:shared:)` — orchestrator. Builds the
  dispatch plan via `MoEDispatch.prepare`, runs each active expert on
  its assigned tokens, scatters back, adds the shared expert.

### `MTPBlock.swift`
`MTPBlock(block:eProj:hProj:...)`. Forward fuses the next-token
embedding with the previous block's hidden state, runs `Block.forward`,
then routes through `ParallelHead` for a speculative prediction.

Not yet wired into the CLI (Tier 2 building block, integration deferred).

---

## `Sources/DeepSeekKit/Encoding/`

### `Message.swift`
`Role`, `Message`, `ToolCall`, `ThinkingMode` data types.

### `EncodingDSV4.swift`
Port of `Reference/encoding/encoding_dsv4.py`. Covers:

- BOS + per-role markers (`<｜User｜>`, `<｜Assistant｜>`) + EOS.
- `<think>...</think>` reasoning blocks for `.high` / `.max` modes.
- Tool calls in DSML format (`<｜DSML｜tool_calls>`, `<｜DSML｜invoke ...>`,
  `<｜DSML｜parameter ...>`).
- Optional `REASONING_EFFORT_MAX` system prompt and `TOOLS_TEMPLATE`
  block, auto-merged into the first system message.

`encodeMessages(_:mode:toolSchemasJSON:)` produces the prompt string.
`parseCompletion(_:mode:)` parses the model output back into a
`Message`, extracting any `<think>` and tool-call blocks.

Deferred: task tokens, response_format injection, latest_reminder
(simple string concatenation; do it caller-side until needed).

---

## `Sources/DeepSeekUI/` (desktop app)

The SwiftUI app that drives `DeepSeekKit` locally **and** dispatches
to remote OpenAI-compatible providers (OpenRouter today). Top-level
groups:

- `State/` — observable state owned by the App scene.
- `Views/` — SwiftUI views, grouped by surface area.
- `Utility/` — small helpers used across the surface.

### State

#### `InferenceService.swift`
Long-lived, non-ObservableObject (its mutating fields are guarded by
an internal serial `q`). Holds the loaded `Transformer` + `Tokenizer`
+ a `CacheImage` shadow of what's in the GPU KV cache so the
next turn's prompt-prefix match enables the fast-delta path.

Public surface used by the chat flow:
`loadModel(at:strategyOverride:forceLoad:onPlan:)`,
`unloadModel()`, `tokenizeFullHistory(...)`,
`tokenizeFirstTurnWithProject(...)`, `tokenizeUserTurnDelta(...)`,
`tokenizeToolOutputsDelta(...)`, `generateForConversation(...)`,
`beginDelegation()`, `endDelegation(_:)`,
`currentModelDir()`, `currentTokenizer()`.

#### `ModelEndpoint.swift`
`enum ModelEndpoint: Codable, Hashable` with cases `localDirectory(path)`
and `openRouter(modelID)`. Carries `displayName` / `subtitle` /
`iconName` / `isRemote` so picker rows can render uniformly across
backends.

#### `ModelLibrary.swift`
`@MainActor ObservableObject` persisting `[ConfiguredModelEntry]`
under `Application Support/.../models.json`. Powers the picker's
**Recent** submenu (`recents()`), bumped by `touch(_:)` after every
successful load, drained by `forget(_:)`.

#### `ModelState.swift`
`@MainActor ObservableObject` façade over `InferenceService`'s load
lifecycle. Single source of truth for "what model is the chat
talking to right now". Status enum:
`.idle / .loading(endpoint, plan?) / .loaded(endpoint, config) /
.error(endpoint, message)`. Drives `load(_:)`, `unload()`,
`retryWithForce()`. Dispatches between `loadLocal` and
`loadRemoteOpenRouter` per endpoint kind.

#### `KeychainStore.swift`
Generic-password wrapper over Security. Service
`com.deepseek.v4pro`; entries keyed by `account` string.
`KeychainAccount.openRouterAPIKey` is the canonical slot today.
API: `set / get / delete / exists`.

#### `OpenRouterAPI.swift`
DTOs (`OpenAIMessage`, `OpenAIToolCall`, `OpenAIStreamChunk` +
deltas, `OpenAIUsage`) + `OpenRouterClient` (URLSession wrapper).
`streamChatCompletion(apiKey:body:)` returns an
`AsyncThrowingStream<OpenAIStreamChunk, Error>` driven by
`URLSession.bytes(for:).lines` so the chat sees the first token
without buffering the whole reply. `fetchModels` + `validateKey`
round out the surface.

#### `OpenRouterCatalog.swift`
`@MainActor ObservableObject` cache of `GET /models`. Persisted
under `Application Support/.../openrouter-catalog.json`. 24 h
stale-after; `refresh(apiKey:force:)` for the picker's **Reload**
button. `model(for:)` lookup so the cost banner / picker can pull
pricing without re-fetching.

#### `ChatStore.swift`
The big one — owns the chat history, dispatches sends to the right
backend, runs the tool-call loop, drives the delegation chain.
Conversations live as `[Conversation]` (with selection). Phase
machinery (`@Published phases: [UUID: GenerationPhase]`) is what
the chat surface reads to render banners / throughput / streaming
buffer. `send(text:mode:options:maxTokens:)` is the entry point;
it branches on `modelState.loadedEndpoint`:

- Local → the existing token pipeline (`buildPromptTokens` ▸
  `service.generateForConversation` ▸ `apply` ▸ on `.done` with
  tool calls, `runToolCallsAndContinue`).
- Remote (`.openRouter`) → `sendRemote` ▸ `runRemoteLoop`, an
  OpenAI/SSE driver that mirrors the local lifecycle into the
  same phase types but stores no encodedTokens.

Also owns: delegation dispatch (`executeSubAgentDelegation` ▸
`dispatchDelegation` ▸ `runSubAgentToCompletion` + Inner), with
KV-cache snapshot/restore around each level via
`service.beginDelegation` / `endDelegation`; cost tracking
(`Conversation.cumulativeCostUSD` + `GenerationMetrics.turnCostUSD`).

#### `Conversation.swift`
`Conversation` struct: id, title, messages, optional `projectID` /
`agentID`, `cumulativeCostUSD`, `encodedTokens` (local only),
`pendingTurn` (crash recovery, local only). `StoredMessage` / `StoredToolCall` /
`StoredRole` / `PendingTurn` mirror the on-disk Codable shape;
`asKitMessage()` / `.from(Message:)` round-trip to the
`DeepSeekKit.Message` the engine speaks.

#### `AgentLibrary.swift`
`@MainActor ObservableObject` over `[AgentConfig]`. Each agent is a
preset of (name, summary, systemPrompt, allowedToolNames,
samplingDefaults, defaultMode, iconName, tint). Persisted to
`Application Support/.../agents.json`. CRUD + `agent(id:)`. `AgentTint`
enum + `AgentTint.color(for:)` map the on-disk tint identifiers
("blue", "purple", …) to SwiftUI Colors.

#### `MCPClient.swift`
Two classes:

- `MCPClient` — one live JSON-RPC client over stdio. Spawns the
  server through `/usr/bin/env` with a PATH-extended environment,
  handles the `initialize / initialized / tools/list` handshake,
  newline-framed JSON-RPC, pending-request map guarded by
  `NSLock.withLock`. `callTool(_:arguments:)` for invocation.
- `MCPClientPool` — `@MainActor ObservableObject` keyed by server
  UUID. Re-syncs with `MCPServerLibrary` on every mutation
  (`librarySynced(_:)`), spawning newly-enabled / bouncing
  edited / disconnecting removed clients. `allTools()` flattens
  every connected server's catalogue;
  `toolSchemasJSON(allowedNames:)` builds the DSML system-block
  JSON the local model expects; `invokeQualified(_:argsJSON:)`
  routes a `<server>__<tool>` call to the matching client.

#### `MCPServerLibrary.swift`
`@MainActor ObservableObject` storage for `[MCPServerConfig]`
(name, command, args, env, enabled, createdAt). Persisted to
`mcp.json`. Includes `importClaudeDesktopJSON(_:)` so users can
paste their existing Claude Desktop config.

#### `DelegationFrame.swift`
Per-in-flight-sub-agent UI frame: agent identity (id, name, icon,
tint), task text, streaming buffer, depth. Pushed/popped by
`ChatStore.runSubAgentToCompletionInner`; rendered by
`DelegationStackView`.

#### `DocumentLibrary.swift`
Persists the user-imported documents (each tokenised once against
the loaded model's tokenizer). `ModelFingerprint.of(modelDirPath:)`
detects when a document was tokenised against a different model and
needs re-import.

#### `ProjectLibrary.swift`
Groups documents into named "projects" the chat can attach to its
first turn for context injection. Persisted under
`Application Support/.../projects/`.

#### `ConvertViewModel.swift`
Drives the offline converter binary from the Convert sheet — argv
assembly, child process spawn, stdout/stderr streaming back into
the SwiftUI sheet.

#### `AppSettings.swift`
`@AppStorage`-backed user preferences (key names in
`AppSettingsKey`). Sampler defaults, loading strategy, last + recent
model dirs.

### Views/

#### `ContentView.swift`
Top-level router. Always renders `ChatContainer` — the model load
step no longer gates the surface. `ChatContainer` lays out the
sidebar + chat detail + four toolbar pickers (`ModelPicker`,
agent picker, project picker, Convert). `ModelPicker` is a Menu
showing the current load status, recents (auto-load on click),
**Choose model folder…**, **Add OpenRouter model…**, **Unload**,
and **Retry with Force Load** when in `.error`.

#### `Chat/ChatView.swift`
The single-conversation chat surface. Owns the draft text +
`@AppStorage` sampling defaults. Layers (top → bottom): scrolling
transcript with `MessageView`s, `ThroughputBar` while streaming,
`modelStateBanner`, cumulative-cost banner, `DelegationStackView`,
resume banner, `thinkingPicker` (segmented control above composer),
`ComposerView`. `resolveSampling()` merges the agent override (when
attached) with the global sliders to produce per-send
`SamplingOptions` + mode + maxTokens.

#### `Chat/ComposerView.swift`
Text field + Send/Stop button. `canSend: Bool` gates Send and
rewrites the placeholder ("Load a model from the toolbar…") so a
chat with no model loaded is visibly inert.

#### `Chat/MessageView.swift`
One bubble. Renders user / assistant / system roles distinctly;
assistant grows a `ReasoningDisclosure` for `<think>` /
`reasoning_content` content and a wrench-icon disclosure for
`toolCalls + toolOutputs`. Delegation calls (recognised by
`call.name == EncodingDSV4.delegateToolName`) get a special row
showing the target agent's icon + name + the bare task text
instead of the JSON envelope.

#### `Chat/DelegationStackView.swift`
Live card stack of in-flight sub-agents (`ChatStore.activeDelegations`).
Each frame shows the agent identity, its task as a quote, and the
tail of the streaming buffer (clamped to 6 lines so a chatty
sub-agent doesn't push the composer off-screen). Depth indents
each level by 14 px.

#### `Chat/ReasoningDisclosure.swift`
Collapsible brain-icon disclosure for `Message.reasoningContent`.

#### `Sidebar/ConversationListView.swift`
The chats sidebar — list + selection + delete + new-chat trigger.

#### `AddOpenRouterModelSheet.swift`
Modal sheet from the toolbar's **Model → Add OpenRouter model…**.
Search-filters the `OpenRouterCatalog`, renders pricing + context
length per row, kicks off `ModelState.load(.openRouter(…))` on
selection. Shows an inline warning when no Keychain key is
configured.

#### `Settings/SettingsScene.swift`
The `Cmd+,` scene. Threads every library + service into nine tabs:
Generation, Loading, Model Config, Agents, Documents, Projects,
MCP, API Keys, Storage.

#### `Settings/APIKeysSettingsTab.swift`
SecureField for the OpenRouter key → KeychainStore. Save / Delete /
Test (hits `/auth/key`).

#### `Settings/GenerationSettingsTab.swift` · `LoadingSettingsTab.swift` · `ModelConfigSettingsTab.swift` · `StorageSettingsTab.swift`
Tab content for the existing engine-side settings (sampler,
loader, model overrides, on-disk history). Unchanged surfaces.

#### `Agents/AgentsView.swift` · `Agents/AgentEditSheet.swift`
Master-detail for the agent registry and the sheet that edits one.
Tool-policy is a segmented control with three modes (all / none /
explicit allowlist).

#### `MCP/MCPServersView.swift` · `MCP/MCPServerEditSheet.swift`
Same pattern for MCP servers. Sidebar status footer is a live
`StatusRow` observing the matching `MCPClient`.

#### `Projects/ProjectsView.swift` · `Projects/ProjectDetailView.swift` · `Projects/CreateProjectSheet.swift`
Project CRUD + per-project document picker.

#### `Documents/DocumentsView.swift` · `Documents/ImportDocumentSheet.swift`
Single-document import: pick a file, choose splitting, tokenise
against the loaded model.

#### `Convert/ConvertSheet.swift`
Offline weight quantisation. Driver: `ConvertViewModel`. Streams the
converter binary's stdout into the sheet's log pane.

#### `Loading/PreflightSummaryView.swift`
Read-only summary of a `LoadPlan` (shard count, RAM budget, chosen
strategy). Currently unmounted — kept around for a future
"Show preflight detail" toggle from the model picker.

### Utility/

#### `PersistencePaths.swift`
Single source of truth for every Application Support path the app
writes to (`mcp.json`, `agents.json`, `models.json`,
`openrouter-catalog.json`, the projects/documents directories, the
chat history root).

#### `MarkdownText.swift`
Markdown renderer used by `MessageView` for finalised assistant
content. Promotes fenced code blocks into separate artifact cards.

#### `ProjectIndexer.swift`
Helper that tokenises a project's files against the current model
and writes the per-document token index.

---

## `Sources/deepseek/main.swift`

CLI for token generation. Flags: `<model-dir> <prompt> [--mode raw|chat]
[--max-tokens N] [--temperature T]`.

Loads config, tokenizer, and model (via `Transformer.load`). Encodes
prompt with `EncodingDSV4` (chat mode) or as-is (raw). Runs prefill,
then decode loop using `Sampler.sample` with `SamplingOptions`. Streams
in raw mode; buffers + parses in chat mode.

## `Sources/converter/main.swift`

CLI that ports `Reference/inference/convert.py`. Reads HF safetensors,
applies the rename mapping (`self_attn → attn`, `mlp → ffn`,
`weight_scale_inv → scale`, etc.), fuses FP8/FP4 + scale into BF16/F16
where requested, packs the output into layer-aligned shards capped at
`--shard-size-gb`, writes `model.safetensors.index.json`, copies the
tokenizer files. Resume-safe: scans the output dir before writing and
skips shards already present at the right size.

Helpers used by the converter (defined in the same file): `renameKey`,
`shouldSkip`, `floatToBF16`, `floatToF16`, `fuseFP8ToNative`,
`fuseFP4ToNative`, `isFP8DType` / `isFP4DType` / `isE8M0DType`.

---

## `Tests/DeepSeekKitTests/`

One XCTest per kernel / module that has a GPU implementation paired
with a CPU reference. Convention: `*Tests.swift` files compare
Metal output vs `referenceCPU(...)` on small randomized inputs.

| Test file | What it covers |
|---|---|
| `HadamardTests.swift` | FWHT correctness + involution (H∘H = I) |
| `HCSinkhornTests.swift` | Sinkhorn-normalized comb output + doubly-stochastic |
| `ActQuantTests.swift` | FP8/FP4 round-trip + scale match |
| `SoftmaxAxisTests.swift` | Softmax along arbitrary axis |
| `TopKTests.swift` | Top-K values + indices, k = 1 special case |
| `MoEDispatchTests.swift` | gather + scatter round-trip identity |
| `OverlapTransformTests.swift` | Compressor shuffle, pad behavior on s=0 |
| `EinsumTests.swift` | `bshd,btd→bsht` + `bsgd,grd→bsgr` |
| `AttentionIndicesTests.swift` | sliding window + compressed indices |
| `LinearTests.swift` | f32 GEMM + FP8 GEMM relative bound |
| `SparseAttentionTests.swift` | full forward + all-padding zero case |
| `HyperConnectionsTests.swift` | pre + post round-trip |
| `CompressorTests.swift` | prefill no-overlap match vs CPU |
| `BPETokenizerTests.swift` | encode/decode round-trip on UTF-8 + special tokens |
| `MoEHashRoutingTests.swift` | hash routing emits correct expert ids from tid2eid |
