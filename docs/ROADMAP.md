# Roadmap & known limitations

What's implemented, what's stubbed, and what's deferred. When
opening a file expecting a specific feature, check here first.

The project has two roughly-independent stacks: the **inference
engine** (`DeepSeekKit`) and the **desktop app** (`DeepSeekUI`).
Their roadmaps run in parallel — engine work is per-kernel /
per-layer correctness vs performance, app work is feature surface.

---

## Engine

### ✅ Tier 1 — Multi-token chat (done)

The CLI generates token-by-token end-to-end. All the blockers from
the "single token per invocation" prototype are removed.

- `MLA.forward` — prefill (any seqlen, with cutoff/wrap when >
  window) and decode (single-token ring-buffer write).
- `Compressor.forward` — prefill (with and without overlap), decode
  (state machine that emits one compressed token every `ratio`
  steps, including overlap with state-shift).
- `Transformer.forward` — full layer chain, embed → HC expand →
  blocks → head.
- Multi-token CLI loop with streaming output (raw mode) and
  `<think>` buffering (chat mode).

### ✅ Tier 2 — UX completeness (done)

- `Sampler.sample(_:history:options:)` — temperature, repetition
  penalty, top-K, top-P, Gumbel-max multinomial.
- `EncodingDSV4` — port of the practical chat-with-tools surface:
  BOS/role markers, EOS, `<think>...</think>` reasoning blocks,
  REASONING_EFFORT_MAX system prompt, tool_calls DSML emit + parse,
  `__delegate_to_agent` synthetic schema, native tool_outputs
  block (`<｜tool▁outputs▁begin｜>…<｜tool▁outputs▁end｜>`).
- `MTPBlock.callAsFunction` — forward pass implemented;
  speculative decoding integration in the CLI is the next step.
- `Transformer.snapshotKVCache` / `restoreKVCache` — value-typed
  KV snapshots that ChatStore uses around sub-agent delegations
  so the host doesn't pay a cold re-prefill on return.

### ⏳ Tier 3 — Parity with Python reference (partial)

#### Done

- `MoEHashRoutingTests` — verifies hash-routing layers route via
  the `tid2eid` lookup, exercising the `model.py:577` branch.

#### Deferred — structural prerequisite needed

- **act_quant noise on non-rope KV dims** (`MLA`, `Compressor`).
  The reference applies `act_quant(kv[..., :-rope_dim], ...)` to
  inject QAT noise. Our port skips it because `Tensor` has no
  strided view — that slice along the last axis is not contiguous.
  Adding strided views is a refactor across every kernel; impact
  on the forward result is < 1 % typical.

#### Deferred — not on critical path

- **`cast_e2m1fn_to_e4m3fn` in converter** (`--expert-dtype fp8`
  lossless re-encode). Unused with the default BF16 path since
  FP4 experts are fully dequantized to BF16 at convert time.

#### Deferred — needs external environment

- **End-to-end numerical validation vs Python reference**. Requires
  PyTorch + CUDA to dump activations from
  `Reference/inference/generate.py` on a toy config, then compare
  with the Swift forward. Plan documented in earlier roadmap
  iterations.

### Performance — correctness-first, not yet optimized

See [PERFORMANCE.md](PERFORMANCE.md) for the full breakdown.
Headline opportunities, none pursued yet:

- **MLA multi-token forward with `startPos > 0`** → would collapse
  the post-tool-call delta loop (N single-token forwards) into one
  multi-token forward, ~5-10× speedup on tool-heavy turns.
- simdgroup_matrix BF16 GEMM → ~5-10× on every Linear.
- FlashAttention tiling for sparse_attn → ~3-5× on attention.
- Persistent MoE dispatch kernel → ~2× per layer.
- Pipeline state caching → ~10-50 ms saved per inference call.
- KV cache pool → matters for multi-session serving.
- **KV cache persistence to disk** (B3) → reopen the same project
  chat after a restart without paying the cold prefill again.

---

## Desktop app

### ✅ Done

#### Model lifecycle

- **In-chat model picker** (toolbar). The app launches on the chat
  surface unconditionally; the load step is reachable from the
  Model menu (Browse / Recents / Add OpenRouter model / Unload /
  Retry with Force Load).
- `ModelEndpoint` enum with `.localDirectory` + `.openRouter`
  cases; `ModelLibrary` persists recents.
- `ModelState` published lifecycle (`idle / loading / loaded /
  error`) that the chat surface reads to render the load banner +
  composer gating.

#### Remote inference (OpenRouter)

- Keychain-backed API key (`KeychainStore` + Settings → API Keys).
- Full SSE streaming via `OpenRouterClient.streamChatCompletion`.
- `OpenRouterCatalog` with 24 h disk cache + manual refresh, drives
  the `AddOpenRouterModelSheet` autocomplete.
- `ChatStore.sendRemote` parallel dispatch with the same
  `GenerationPhase` lifecycle as the local path.
- Tool calling: MCP tools translated to OpenAI `tools` array;
  multi-iteration HTTP loop with `runRemoteLoop` (cap 8).
- Reasoning content (`delta.reasoning_content`) captured and
  rendered through the existing brain-icon disclosure.
- Per-turn cost (`usage.total_cost` in `GenerationMetrics`) +
  cumulative chat cost (`Conversation.cumulativeCostUSD`,
  persisted to disk). Banners under the bubble and above the
  composer.
- Reasoning-mode picker → `reasoning: { effort }` hint.

#### MCP

- `MCPClient` (stdio JSON-RPC, newline framing) + `MCPClientPool`
  reactive to `MCPServerLibrary`.
- Settings → MCP tab: CRUD + live status footer + reconnect +
  Claude Desktop config importer.
- Tool schemas exposed to both backends (DSML for local, OpenAI
  `tools` for remote) with agent allowlist filtering.

#### Agents

- `AgentConfig` (system prompt, allowedToolNames, sampling
  defaults, default mode, icon, tint) + `AgentLibrary`.
- Settings → Agents tab: CRUD + tool-policy segmented control +
  per-agent sampling editor.
- Chat toolbar Agent picker; per-chat attach/detach via
  `Conversation.agentID`.
- Sampling overrides on send (Generation tab sliders are bypassed
  when an agent is attached).
- Thinking-mode picker above the composer; locks to the agent's
  `defaultMode` when one is attached.

#### Sub-agent delegation

- Synthetic `__delegate_to_agent` tool with roster of every other
  registered agent (auto-injected when ≥ 2 agents exist).
- `ChatStore.runSubAgentToCompletion` drives the sub-agent
  through its own tool-call loop (cap 8).
- Nested delegation up to 3 levels, with structural
  prevention at the depth cap (schema not injected) and chain-
  membership cycle refusal at the dispatch site.
- KV-cache snapshot/restore via
  `InferenceService.beginDelegation` / `endDelegation` so the
  host doesn't pay a cold re-prefill on return.
- Live `DelegationStackView` card above the composer shows every
  in-flight sub-agent (icon, task, streaming buffer, indented by
  depth).

#### Native tools (DeepSeekTools)

- Separate target with the `Tool` protocol, `ToolSchema` builder,
  `ToolCategory`, `ToolContext`, `ToolError`, an actor-isolated
  `ToolRegistry` with plan-mode filter + session permission cache,
  and a `PermissionDelegate` boundary.
- 14 built-in tools shipped: `read`, `write`, `edit`, `glob`,
  `grep`, `shell`, `apply_patch`, `webfetch`, `websearch`,
  `repo_clone`, `repo_overview`, `plan`, `task`, `todo`. `lsp`
  registered as a stub (throws `.notImplemented`).
- Smoke tests under `Tests/DeepSeekToolsTests/` for the registry
  + slash-command parser.

#### Agent operating modes + permission system

- `AgentMode.build` / `.plan` on `AgentConfig`, with the
  plan-mode filter enforced structurally by
  `ToolRegistry.availableSchemas(mode:)` (the model never sees
  the disallowed tools).
- `PermissionStore` durable map (`<tool>:<category> → ask |
  alwaysAllow | alwaysDeny`) persisted to `permissions.json`.
- `PermissionPromptView` SwiftUI modal with three actions
  (Deny / Allow once / Always allow); "Always allow" writes
  through to `PermissionStore`.
- Settings → Permissions tab for editing the durable map +
  resetting session grants.

#### Skills, slash commands, themes, keybindings

- `SkillLibrary` holds the catalogue (built-ins + custom).
  `AgentConfig.allowedSkillIDs` restricts which skills the agent
  can activate.
- `SlashCommandLibrary` intercepts `/`-prefixed composer input
  via `SlashCommandPaletteView`. Built-ins: `/mode`, `/tools`,
  `/permissions`, `/skill`, `/theme`, `/clear`, `/help`.
- `ThemeStore` drives appearance (light/dark/system + accent +
  bubble tints) — Settings → Theme tab.
- `KeybindingStore` overlays per-user shortcuts on top of the
  built-in `Keybinding` defaults — Settings → Keybindings tab.

#### Projects + Documents

- Per-document tokenisation against the active model's tokenizer
  (`ProjectIndexer`).
- Project-attached chats splice the project's files into the
  first turn with native repo/file delimiter tokens.
- Settings tabs for Documents (single-file import) and Projects
  (grouping + per-project file picker).

#### Crash recovery

- `Conversation.pendingTurn` snapshots every local Send (prompt +
  generated-so-far ids + mode).
- Resume banner above the composer offers a one-click continue
  when a chat has a non-nil `pendingTurn` and the phase is
  `.idle`. Re-armed during tool-call continuations so a crash
  mid-loop is still resumable.

### ⏳ Deferred / not yet implemented

#### Remote backend

- **Cross-agent delegation on OpenRouter chats**. The synthetic
  `__delegate_to_agent` schema is intentionally not injected on
  remote chats — supporting it would need a remote sub-agent loop
  that doesn't exist. Local fully works.
- **Remote crash recovery**. `pendingTurn` is not armed for
  `sendRemote`; a crash mid-stream loses the turn. The workaround
  is to re-send the user message (idempotent on the provider
  side modulo cost).
- **Anthropic / OpenAI prompt-caching** via OpenRouter. The
  request body doesn't yet pass the cache-control headers
  Anthropic accepts. Cheap to add, hasn't been a felt need.

- **Wire native tool registry into `InferenceService`.**
  `NativeToolHost.dispatch` exists but `InferenceService` still
  only assembles the tools block from MCP. Needs (a) merge of
  native schemas into the system block / OpenAI `tools` array,
  (b) routing native names to `NativeToolHost.dispatch`, (c)
  resolution of `ToolContext.rootDirectory` from the attached
  project (or the user's home as fallback). Tracked as TODO §8
  first item.

- **Real `lsp` tool.** Stub registered today; needs to spawn
  `sourcekit-lsp` for Swift (plus a `pyright` / TS server for the
  obvious follow-ups), JSON-RPC framing the same way `MCPClient`
  does it, and operations for `definition` / `hover` /
  `references` / `diagnostics`.

- **`websearch` real provider.** The default DuckDuckGo lite
  scraper works but is fragile. Plug Tavily / Brave / Serper /
  Bing behind a Keychain-stored key.

- **`ShellTool` sandbox profile.** Today the tool can wrap calls
  in `sandbox-exec` if a profile is provided, but the bundled
  default profile is deliberately strict. Tuning it for typical
  dev workflows + flipping the Settings toggle is open.

- **Custom slash commands UI** + per-project `.deepseek/` config
  (agents / skills / slash commands tracked in the user's repo
  instead of the global Application Support).

- **Inline keybinding rebind widget** — today Settings → Keys
  is read-only + reset. Add the key-grab + conflict detection
  + system-shortcut overwrite confirmation.

- **Custom theme editor** with inline ColorPicker rows for the
  six theme slots. `ThemeStore` already accepts custom themes
  via JSON; UI to author them is missing.

#### Engine performance optimisations

- **MLA multi-token forward with `startPos > 0`**. The biggest
  single perf win for tool-heavy local chats — the tool-output
  delta currently loops N single-token forwards instead of one
  multi-token forward. Blocked by a precondition on
  `MLA.callAsFunction` and a buffer wrap on `Compressor`. Could
  land behind an opt-in flag with low risk.
- **KV cache persistence to disk** (B3). Would let a project-
  attached chat reopen after a restart without re-prefilling the
  project context. Big design space — file format, eviction,
  cross-model invalidation. Not started.

#### Other

- **MTPBlock speculative decoding** in the CLI / GUI. Forward pass
  exists; the integration that runs speculative tokens + verifies
  against the next forward step is unwritten.
- **Multi-batch CLI / batched serving**. The model code is shaped
  for `[B, S]` input but the CLI never exercises B > 1. Batched
  serving is a CLI rework, not a model change.
- **Single-rank only** for the converter and loader
  (`model_parallel == 1`). Multi-rank sharding is unsupported.
- **Encoding stubs**: task tokens (`<｜action｜>`, `<｜query｜>`),
  `response_format` schema injection, `latest_reminder` token.
  Caller can prepend manually.
- **`.high` thinking mode in local**: `.chat` and `.max` produce
  different prompts; `.high` currently behaves like `.chat`. The
  reference's `.high` mode adds a less-extreme reasoning prompt
  that hasn't been ported. (On remote the picker maps `.high` to
  `reasoning: { effort: "medium" }`.)

### Known limitations on existing surfaces

- **`bf16` ParallelEmbedding**: precondition is
  `weight.dtype == .f32`. After `--target-dtype bf16` the embed
  table lands as BF16 and gets random init. Workaround: relax
  the precondition (~20 LOC) or use `--target-dtype keep`.
- **`wo_a` always fused** by the converter (uses `Einsum.bsgdGrd`
  internally). Leaving it FP8 would need an FP8-aware einsum.
- **`Conversation.modelDirPath` is metadata**, not a load key.
  Created chats stamp it as an audit trail; switching the loaded
  model doesn't migrate old chats.

---

## How to extend

When adding new features, follow the conventions:

1. **New kernel**: see
   [`KERNELS.md`](KERNELS.md#how-to-add-a-new-kernel). Always pair
   with a Swift wrapper + CPU reference + XCTest.
2. **New layer composition**: put it in
   `Sources/DeepSeekKit/Layers/`. Document the corresponding Python
   source line range in the file header.
3. **New CLI flag**: in `Sources/deepseek/main.swift` (inference)
   or `Sources/converter/main.swift` (converter). Update
   [`USAGE.md`](USAGE.md).
4. **New weight name convention**: update the canonical map in
   `Assembly.swift`'s `Transformer.load`. Add a fallback name
   list via `loader.tryLoad([...])` so old and new naming both
   work.
5. **New remote backend / Settings tab / MCP transport / chat
   feature**: see
   [`DEVELOPING.md` §9-§12](DEVELOPING.md#9-recipe-add-a-new-modelendpoint-backend).
