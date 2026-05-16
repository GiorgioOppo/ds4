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

### `GGUF.swift`
Parser for the GGUF v2/v3 metadata format used by llama.cpp.
Reads the magic + version, the key/value metadata block (all
GGUF value types incl. arrays), the tensor info table
(`name / n_dims / shape / ggml_type / offset`), and the global
alignment padding. `GGUFHeader` exposes the parsed result;
`GGUFTensorInfo` carries enough to compute the absolute offset
+ byte count of each tensor.

### `GGUFLoader.swift`
mmap-backed wrapper over a `.gguf` file. Same pattern as
`SafeTensorsFile`: `init(url:)` parses the header, `info(name:)`
inspects, `load(name:) -> Tensor` returns a zero-copy `Tensor`
view for the supported pass-through dtypes (`F32`, `F16`, `BF16`,
`I32`, `I8`). Quantized dtypes (`Q4_0 / Q4_K_M / …`) raise
`GGUFError.unsupportedType` — see [`GGUF.md`](GGUF.md) for the
roadmap.

### `WeightLoader.swift`
Walks a directory of `.safetensors` shards, builds a flat name → shard
index, exposes `load(_:)` and `tryLoad(_:[fallbackNames])`. Skips LFS
pointer files (< 1 KB). Used by `Transformer.load(from:)`.

### `Tokenizer.swift`
Protocol `Tokenizer { encode/decode/bosId/eosId }` plus
`TokenizerLoader.load(tokenizerDir:)` — a dispatcher that
detects the format on disk and picks the matching impl:

- `tokenizer.json` (HuggingFace) → `BPETokenizer`.
- `tokenizer.model` (SentencePiece protobuf) →
  `SentencePieceTokenizer`.
- `vocab.txt` (+ optional `tokenizer.json`) → `WordPieceTokenizer`.

Also resolves the active `ChatTemplate` (DSV4 vs Jinja) by
inspecting `tokenizer_config.json.chat_template`.

### `BPETokenizer.swift`
Byte-level BPE compatible with HuggingFace `tokenizer.json`. Parses
`model.vocab`, `model.merges`, `added_tokens`, plus the pre-tokenizer
regex. GPT-2 byte-to-unicode map for UTF-8 round-trip. Greedy
lowest-rank merge BPE.

### `SentencePieceTokenizer.swift`
Reads the binary `tokenizer.model` protobuf used by Llama /
Mistral / Gemma. Unigram language model with `▁` whitespace
encoding; supports SP normalization (NFC / NFD / NMT) + BOS/EOS
control tokens.

### `WordPieceTokenizer.swift`
BERT-style WordPiece (`vocab.txt` + greedy longest-match
sub-token resolution with `##` continuation prefix). Used by the
embedding-only / classification models the GGUF reader will
eventually load.

### `KVCache.swift`
Allocators for per-layer KV caches and a `CacheBank` aggregating them.
Currently unused at runtime — MLA holds its `kvCache` field directly,
populated in assembly.

### `Sampling.swift`
- `Sampler.argmax(_:)` — GPU-side reduction, returns a single Int.
- `Sampler.applyTemperature(_:)` — in-place GPU scaling.
- `Sampler.sample(_:history:options:)` — full pipeline applied
  host-side after a single CPU-GPU sync. Order:
  1. temperature (skipped at `T == 0`, falls to argmax).
  2. repetition penalty (HF-style) on history-id logits.
  3. frequency + presence penalties (OpenAI-style).
  4. top-K truncation.
  5. top-P (nucleus) cutoff.
  6. min-p — drop tokens with `p < min_p × max_p`.
  7. tail-free sampling — keep the mass that flattens the
     second derivative.
  8. locally-typical sampling — keep tokens close to the
     average per-token surprise.
  9. Mirostat v2 (when enabled) — replaces the (top-K /
     top-P / min-p / tfs / typical) layer with an adaptive
     surprise-target loop driven by `mirostatTau` + `mirostatEta`.
  10. Gumbel-max multinomial.
- `SamplingOptions` struct carries every parameter + the RNG
  state + Mirostat's running estimate across decode steps. Tests
  in `Tests/DeepSeekKitTests/SamplerTests.swift`.

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

### `ChatTemplate.swift`
`ChatTemplate` protocol — render a `[Message]` into the prompt
string the model expects. Two implementations:

- **`DSV4Template.swift`** — wraps `EncodingDSV4.encodeMessages`.
  Used for every DeepSeek-V4 checkpoint (the only one with
  native MLA + MoE the engine actually runs today).
- **`JinjaChatTemplate.swift`** — wraps the Jinja2 subset
  driver below. Used when the loaded model's
  `tokenizer_config.json` carries a `chat_template` field
  (Llama / Mistral / Qwen / ChatML / any HuggingFace model).

`ChatTemplateOptions` carries the mode + add-generation-prompt
flag + tool-schema JSON. `ChatTemplateError` reports unsupported
features or template parse failures.

### `JinjaTemplate.swift`
Pure-Swift implementation of the Jinja2 subset HuggingFace chat
templates actually use. ~900 LOC. Supports: variable
interpolation (`{{ var }}`), `{% for %}` / `{% if %}` /
`{% elif %}` / `{% else %}`, filters (`trim`, `length`, `lower`,
`upper`, `default`, `tojson`, …), the `raise_exception` builtin,
nested context scopes. Not a full Jinja2 — `{% macro %}` /
`{% set %}` / `include` / inheritance are deferred until
something needs them.

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

## `Sources/DeepSeekTools/` (native code-agent toolbox)

Separate Swift target that owns the tools the model can invoke
directly (no MCP round-trip): file ops, shell, web fetch, repo
overview, planning notes, and the protocols + registry around
them. Imported by `DeepSeekUI/State/Tools/NativeToolHost.swift`
(GUI side) and intended to be reusable by a future headless CLI
server.

### Core protocols

#### `Tool.swift`
The base protocol every tool conforms to: a `schema: ToolSchema`,
a `category: ToolCategory`, and an `async throws run(input:context:)`
that returns a `ToolResult { output, metadata }`. Tools are value
types where possible; long-lived state lives on the registry's
stores (e.g. `PlanStore`).

#### `ToolSchema.swift`
JSON-schema fragment describing one tool's input. Includes the
`SchemaBuilder` helpers (`string`, `number`, `boolean`, `array`,
`object`, `oneOf`, with `required: [...]`) so tool authors don't
hand-write the JSON.

#### `ToolCategory.swift`
`enum ToolCategory: readOnly | planning | mutating | dangerous |
network`. Drives both the plan-mode filter and the permission
policy.

#### `ToolContext.swift`
Per-call context: `rootDirectory` (the project / cwd root),
`mode: AgentMode`, `permissions: PermissionDelegate`, `sandbox`
flags. Tools never reach outside this.

#### `ToolError.swift`
Structured failures (`notImplemented`, `denied`, `invalidInput`,
`runtime`, `cancelled`, …). The dispatcher converts these to a
short string the model sees as the tool output.

#### `ToolRegistry.swift`
Actor that owns the active `[Tool]`, the plan-mode filter
(`availableSchemas(mode:)`), and the per-`(tool, category)`
session permission cache. `dispatch(name:input:context:)` is the
single entry point both backends call. Re-emits the
`PermissionDelegate.request(...)` for any uncached
mutating/dangerous/network call.

#### `Permission.swift`
`PermissionDecision` enum (`allowOnce / alwaysAllow / alwaysDeny`)
+ `PermissionDelegate` protocol. The GUI host wires this to a
SwiftUI sheet; the CLI default (`AutoPermissionDelegate`)
auto-allows everything except `.dangerous` unless flagged.

#### `AgentMode.swift`
`enum AgentMode: build | plan`. Plan-mode hides `.mutating` +
`.dangerous` tools from the schema list the model sees, which is
the structural enforcement; the registry also rejects late-
arriving calls with `denied` if the mode changes mid-stream.

#### `SlashCommand.swift`
Built-in slash commands the composer intercepts before the
inference loop sees the text: `/mode`, `/tools`, `/permissions`,
`/skill`, `/theme`, `/clear`, `/help`. The library is extendable
at runtime (user-defined commands come in a follow-up step).

#### `Theme.swift`
Theme descriptor (light/dark/system + accent + bubble tints +
appearance). Built-in catalogue plus a `decode(_:)` for custom
JSON themes.

#### `Keybinding.swift`
Configurable keyboard shortcuts addressable by stable action ids
(`composer.send`, `composer.stop`, `mode.toggle`,
`palette.open`, …). The library returns a default mapping; the
UI's `KeybindingStore` overlays per-user overrides.

#### `DefaultTools.swift`
`DefaultTools.standard(...)` returns the built-in tool set
populated into a fresh `ToolRegistry`. Single source of truth for
which tools ship today; new tools land here.

### `Tools/` — built-in tools

Each file is one `Tool` implementation. Category in brackets.

- `ReadTool.swift` (readOnly) — line-numbered UTF-8 file read with
  optional offset/limit.
- `WriteTool.swift` (mutating) — atomic create-or-overwrite.
- `EditTool.swift` (mutating) — exact-match string replace; refuses
  on non-unique match unless `replaceAll`.
- `ApplyPatchTool.swift` (mutating) — minimal unified-diff applier
  (create / delete / in-place, reverse-order hunks, exact context).
- `GlobTool.swift` (readOnly) — walk the agent root, match by glob,
  sort by recent mtime.
- `GrepTool.swift` (readOnly) — `NSRegularExpression` search across
  files matched by an inner glob.
- `ShellTool.swift` (dangerous) — subprocess via `/bin/zsh -c`,
  combined output, 32 KB cap, watchdog timeout, optional
  `sandbox-exec` wrap.
- `WebFetchTool.swift` (network) — `URLSession` GET; HTML → plain
  text unless `raw=true`. 1 MB cap.
- `WebSearchTool.swift` (network) — DuckDuckGo lite scraper as the
  default backend. Fragile by design; replace with a real API for
  serious use.
- `RepoCloneTool.swift` (dangerous) — shallow `git clone` via
  subprocess. Uses the user's local git auth.
- `RepoOverviewTool.swift` (readOnly) — tree (depth-capped) +
  extension histogram + content of conventional manifests
  (`Package.swift`, `Cargo.toml`, `package.json`, …).
- `LSPTool.swift` (readOnly) — stub; registers a schema but
  `run` throws `.notImplemented`. Pending: spawn `sourcekit-lsp`,
  JSON-RPC framing, definition / hover / references / diagnostics.
- `TaskTool.swift` (planning) — list / set / update the active
  task list. Backed by `PlanStore`.
- `TodoTool.swift` (planning) — cross-task TODO bag.
- `PlanTool.swift` (planning) — read / replace the high-level
  plan note.
- `PlanStore.swift` — actor that owns plan + tasks + todos. Held
  inside `NativeToolHost` so the three planning tools share the
  same data.

### `Skills/Skill.swift`
A reusable bundle of (system prompt addendum, suggested tool
allowlist, optional default mode). Built-in skills declared at
stable UUIDs in `BuiltInSkills`; user can layer custom skills on
top.

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

Carries the active `chatTemplate: ChatTemplate` (defaults to
`DSV4Template()`; the loader swaps in `JinjaChatTemplate` when
the model directory ships a `tokenizer_config.json` with a
`chat_template`). The V4 prompt path still calls
`EncodingDSV4.*` directly — the dispatcher lets future
non-DeepSeek local backends speak their own template through the
same `InferenceService` surface.

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

#### `Tools/NativeToolHost.swift`
Singleton that owns the `ToolRegistry` + `PlanStore` for the GUI.
Bridges the registry's `PermissionDelegate` to a SwiftUI sheet
(`PermissionPromptView`) so the model's mutating / dangerous /
network calls produce a modal instead of being auto-denied.
`dispatch(name:input:mode:rootDirectory:)` is the entry point the
chat-flow side calls when wiring through `InferenceService` lands
(TODO §8). Today the host is constructed at app launch and
plumbed into the relevant Settings views.

#### `Tools/PermissionStore.swift`
`@MainActor ObservableObject` over the durable
`<tool>:<category> → ask | alwaysAllow | alwaysDeny` map.
Persisted to `permissions.json`. Read by the registry before the
session cache so a previously-granted "Always allow" skips the
modal.

#### `SkillLibrary.swift`
`@MainActor ObservableObject` over the catalogue of skills
(`Skill` from DeepSeekTools). Built-in entries from
`BuiltInSkills` are merged with the user's custom ones; the
Agents tab uses this for the per-agent `allowedSkillIDs` editor.

#### `SlashCommandLibrary.swift`
Library of `SlashCommand`s the composer intercepts. Today only
the built-in catalogue is exposed; custom command CRUD is
scaffolded for a future Settings tab.

#### `ThemeStore.swift`
`@MainActor ObservableObject` that owns the active theme +
appearance preference. Reads `themes.json` (custom themes) +
yields the built-in catalogue. Drives the Theme Settings tab.

#### `KeybindingStore.swift`
Per-user keybinding overrides on top of the `Keybinding`
defaults. Persisted to `keybindings.json`. Read by the
Keybindings tab and by the SwiftUI views that bind action ids to
`KeyboardShortcut`.

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
The `Cmd+,` scene. Threads every library + service through the
full tab set: Generation, Loading, Model Config, Agents, **Tools,
Permissions, Skills, Theme, Keybindings**, Documents, Projects,
MCP, API Keys, Storage.

#### `Settings/APIKeysSettingsTab.swift`
SecureField for the OpenRouter key → KeychainStore. Save / Delete /
Test (hits `/auth/key`).

#### `Settings/ToolsSettingsTab.swift`
Read-only inventory of every native tool the registry knows about.
Groups by `ToolCategory`, shows the JSON schema's name + summary
+ whether the tool is reachable under Plan and Build modes. No
edit — tools are code-defined, not user-configured.

#### `Settings/PermissionsSettingsTab.swift`
Editor for `PermissionStore`. List of every (tool, category)
default with a segmented control `ask | alwaysAllow | alwaysDeny`.
Reset-all button for session grants.

#### `Settings/SkillsSettingsTab.swift`
CRUD for `SkillLibrary`. Built-in skills appear as read-only
rows; custom skills get the full edit sheet (name, instructions,
suggested tools, default mode).

#### `Settings/ThemeSettingsTab.swift`
Theme picker + appearance toggle (light / dark / system). Custom
themes from `ThemeStore` listed below the built-ins; a future
editor (TODO §8) will add inline ColorPicker rows.

#### `Settings/KeybindingsSettingsTab.swift`
Read-only list of every bindable action with its current
shortcut. Reset-to-defaults button. The inline rebind widget is
on the roadmap.

#### `Settings/GenerationSettingsTab.swift` · `LoadingSettingsTab.swift` · `ModelConfigSettingsTab.swift` · `StorageSettingsTab.swift`
Tab content for the existing engine-side settings (sampler,
loader, model overrides, on-disk history). Unchanged surfaces.

#### `Tools/ModePickerView.swift`
Segmented `Build / Plan` control pinned in the chat above the
composer. Reads the current conversation's effective mode (agent
default + per-chat override) and writes it back on change.
Disabled with a 🔒 hint when the attached agent locks the mode.

#### `Tools/PermissionPromptView.swift`
Modal sheet presented by `NativeToolHost` when the registry hits
an unresolved mutating / dangerous / network call. Three actions:
**Deny**, **Allow once**, **Always allow**. The Always-allow
choice writes through to `PermissionStore`.

#### `Tools/SlashCommandPaletteView.swift`
Inline picker that opens when the user types `/` in the composer.
Shows filtered matches from `SlashCommandLibrary`; on selection,
replaces the draft with the command's expansion (or fires the
command directly when there are no args).

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
writes to. Today's slots: chat history root (`conversations/`),
projects + documents directories, `mcp.json`, `agents.json`,
`models.json`, `openrouter-catalog.json`, `permissions.json`,
`skills.json`, `themes.json`, `keybindings.json`, per-conversation
KV-cache snapshot (`<id>.kvcache`, B3 reserved).

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
