# TODO

Living checklist of outstanding work. Each item links to where the
corresponding background lives (mostly `docs/ROADMAP.md` and
`docs/PERFORMANCE.md`); read those for context before picking one
up.

Legenda: `[ ]` open · `[~]` partial · `[x]` done · `[!]` blocked

---

## 0. Quantizzazione

- [x] **`--target-dtype int8`** — INT8 W8A16 (peso INT8 simmetrico
  per-riga, gruppo K=128, scala F16). Quantizza solo i pesi
  `Linear` (whitelist in `shouldQuantizeToInt8`); embed/head/norm
  restano BF16. Footprint ≈ ½ × BF16 sui pesi Linear. Sorgenti:
  BF16, F32, FP8+scala, FP4+scala. Kernel: `int8_gemm.metal`. Test:
  `Int8GemmTests.swift`, `Int8ConverterTests.swift`.

- [x] **`--target-dtype int4`** — INT4 sui pesi Linear. Kernel
  `int4_gemm.metal` con unpacking nibble + scala. `Int4Quant.swift`
  per la conversione offline.

- [x] **`--target-dtype int2`** — INT2 sperimentale. `Int2Quant.swift`
  + scelte di scala più conservative.

- [x] **W8A8 (activations INT8)**. Follow-up di W8A16, opt-in per
  layer via `Linear(useW8A8Activations: true)`. Implementato:
  - `ActQuant.Format.int8` + kernel `act_quant_int8` in
    `Sources/DeepSeekKit/Kernels/act_quant.metal` (per-row,
    per-128, simmetrico `[-127, 127]`, scale f32 = amax/127).
  - Coppia di GEMM in `Sources/DeepSeekKit/Kernels/int8_gemm_w8a8.metal`:
    `gemm_int8_w8a8_to_f32` (naive, int32 accumulator,
    rescale per-block) e `gemm_int8_w8a8_to_f32_sg`
    (simdgroup_matrix<bfloat> dopo dequant-on-stage di entrambi
    i lati int8 → bfloat in TGM — Apple Silicon non espone int8
    matrix MMA).
  - `Linear.int8W8A8Forward` con stessi gating SG/naive di W8A16.
  - Test: `Int8ActQuantTests` (round-trip + scale exact),
    `Int8W8A8GemmTests` (naive + SG vs CPU dequant reference).
  Trade-off: throughput memory-bound ~2× (metà bandwidth letta
  per le activations); quantization-noise aggiuntiva — opt-in
  per layer dove la perdita è accettabile (default off).

- [x] **Quantizzazione calibrata (GPTQ / AWQ / SmoothQuant)** sui
  pesi INT8/INT4. Tutti i 5 follow-up chiusi:
  1. ✅ AWQ + SmoothQuant + GPTQ (full OBS via Accelerate LAPACK)
     in `Sources/DeepSeekKit/CalibratedQuant.swift`.
  2. ✅ `inverseChannelScale: Tensor?` su `Linear` + kernel
     `channel_scale_f32` per il pre-mul (TODO §12 di AWQ chiuso).
  3. ✅ `ActivationObserver` + `HessianObserver` con accumulazione
     via `cblas_dgemm`.
  4. ✅ `LlamaCalibrationRunner` + `V4CalibrationRunner` che
     installano hook su `Block.preAttnObserver` / `preFfnObserver`
     per intercettare gli input MLA / MoE in mid-forward.
  5. ✅ `deepseek_calibrate <model> <corpus> <out> --architecture
     llama|v4 --collect-hessian` dumpa `stats.json` + `hessians/<n>.f64`.
  6. ✅ Converter CLI `--quant-method rtn|awq|smoothQuant|gptq` +
     `--calib-stats <path>` con dispatch per-tensor.
  7. ✅ `CalibrationDir` loader pubblico per i consumer esterni.

  Gap residui documentati nei commit (non bloccanti):
  - `inverseChannelScale` round-trip su safetensors (writer in
    converter + reader in `WeightLoader` → consumed by `Linear`
    at load time). Oggi il converter calcola lo scale ma lo
    scarta; AWQ/SmoothQuant restano "QAT warm-start" finché il
    round-trip non chiude.
  - MLA-internal sites non osservati (`wq_b`, `wo_a`, `wo_b`,
    expert `w2`). Servono hook simili in `MLA.callAsFunction` /
    `MoEFFN.callAsFunction`.
  - Router-aware MoE calibration: oggi tutti gli expert ricevono
    le stesse stats (pre-routing); una calibrazione per-routed-token
    serve instrumentation in `MoEFFN`.

---

## 1. Parità con il reference Python

- [ ] **act_quant noise sulle dimensioni non-rope di KV** in `MLA`
  e `Compressor`. Bloccato dall'assenza di viste strided in
  `Tensor`. Impatto stimato: < 1 % di differenza nel forward.

- [ ] **`cast_e2m1fn_to_e4m3fn` nel converter** (`--expert-dtype
  fp8` lossless re-encode). Non sul critical path.

- [!] **Validazione numerica end-to-end vs Python**. Richiede
  PyTorch + CUDA. Piano in `docs/ROADMAP.md`.

---

## 2. Funzionalità di runtime mancanti (engine)

- [ ] **Batched serving** nel CLI. `Transformer.forward` è già
  shaped per `[B, S]` ma `Sources/deepseek/main.swift` itera una
  singola prompt. Rework CLI-level.

- [ ] **Speculative decoding via MTP**. `MTPBlock.callAsFunction`
  è implementato, il CLI non lo invoca.

- [ ] **Multi-rank loader**. Converter e loader assumono
  `model_parallel == 1`.

- [ ] **`bf16` ParallelEmbedding**. Precondition
  `weight.dtype == .f32`. Fix: rilassare + ramo lookup BF16
  (~20 LOC).

- [ ] **Restore `wo_a` non fuso**. Per tenere `wo_a` in FP8 serve
  un einsum FP8-aware.

---

## 3. Encoding (chat / DSML)

- [x] **Tool calls native** (`<｜DSML｜tool_calls>`).
- [x] **Tool outputs native** (`<｜tool▁outputs▁begin｜>…`).
- [x] **Synthetic `__delegate_to_agent` schema** (auto-injected
  quando ≥ 2 agenti registrati).

- [ ] **Task tokens** (`<｜action｜>`, `<｜query｜>`, ecc. da
  `DS_TASK_SP_TOKENS`) non emessi.

- [~] **`response_format` schema injection** (encoding_dsv4.py:49)
  — la versione "constrained decoding via token mask" è chiusa
  (T3 / `SchemaCompiler` / `SchemaMask` / `LocalServer` binding
  / `--json-schema` CLI flag). L'iniezione del JSON Schema come
  hint testuale nel system prompt resta da fare.

- [ ] **`latest_reminder` token** (encoding_dsv4.py:25) non emesso.

- [ ] **Thinking mode `.high`** distinto da `.chat` per i modelli
  locali. Oggi `.high` si comporta come `.chat`. (Su remoto il
  picker mappa già `.high` → `reasoning: { effort: "medium" }`.)

---

## 4. Desktop app

### Stack feature completo

- [x] **In-chat model picker**: toolbar Model menu, browse,
  recents, unload, retry. App si apre senza modello.
- [x] **`ModelEndpoint` enum** con `.localDirectory` + `.openRouter`.
- [x] **OpenRouter R1**: foundation (Keychain + HTTP/SSE client
  + catalog cache).
- [x] **OpenRouter R2**: `ChatStore.sendRemote` parallelo al path
  locale; streaming SSE + reasoning_content + tool_calls accum.
- [x] **OpenRouter R3**: tool dispatch loop (MCP tools come
  OpenAI `tools[]`, cap 8 round-trip HTTP); per-turn + cumulative
  cost dal campo `usage.total_cost`.
- [x] **OpenRouter R4**: Settings → API Keys tab; Add OpenRouter
  model… sheet con autocomplete dal catalogo.
- [x] **Agents A1**: AgentConfig + AgentLibrary + Settings →
  Agents tab.
- [x] **Agents A2**: `Conversation.agentID` + picker toolbar +
  iniezione system prompt + filtro tool MCP per allowlist.
- [x] **Agents A2.5**: sampling defaults dell'agente vincono sui
  slider globali.
- [x] **Agents A3**: sub-agent delegation single-shot, prose-only.
- [x] **Agents A4**: sub-agent con tool MCP propri + loop interno
  bound a 8 iter.
- [x] **Agents A5**: snapshot/restore KV cache (`beginDelegation`/
  `endDelegation`) attorno alle delegation, host non paga il cold
  re-prefill.
- [x] **Agents A6**: tool call rows con identità dell'agente
  delegato + task plain.
- [x] **Agents A7**: nesting fino a 3 livelli con cycle prevention.
- [x] **Agents A8**: live delegation chain UI sopra il composer
  con streaming buffer.
- [x] **MCP M1**: server registry + Settings → MCP tab + import
  Claude Desktop config.
- [x] **MCP M2**: client JSON-RPC over stdio + pool + status live.
- [x] **MCP M3a**: tool schemas iniettati nel system block.
- [x] **MCP M3b**: tool execution + auto-continue loop locale.
- [x] **MCP M4**: rendering tool call/output nei message bubble.
- [x] **Thinking picker** segmented control sopra il composer,
  bound a `@AppStorage`, locked dall'agente quando attached.

### Open

- [x] **Cross-agent delegation on remote chats**. `composeOpenAITools`
  ora emette lo schema `__delegate_to_agent` quando esistono altri
  agenti delegabili; il dispatch in `runRemoteLoop` chiama
  `executeSubAgentDelegation` come sul path locale. Il sub-agent
  spawnato gira sul backend del suo `AgentConfig` (local / OpenRouter
  / Anthropic). Commit `3bfc079`.

- [~] **Remote crash recovery**. `Conversation.remotePendingTurn`
  (`RemotePendingTurn` struct) ora viene popolato in `sendRemote`
  e cleared in `finalizeRemoteIteration`. Manca la UI banner di
  retry su next launch (legge il campo + pulsante reinvia). Commit
  `e0abd40`.

- [x] **Prompt-caching Anthropic** — chiuso via T4 native driver
  (`AnthropicMessageBuilder` auto-inietta `cache_control` su system
  block e ultimo `tool_result`). Commit `f43188c`.

- [ ] **Sub-agent cross-delega reuse**. Una sub-agent invocata due
  volte nella stessa turn paga il cold prefill ogni volta. Cache
  per `(agentID, promptHash)` aggiuntiva (di nicchia). Richiede
  refactor di `InferenceService` cache layer; perf optimization
  separata.

- [~] **Conversation.modelDirPath → endpointID** migration. Il
  campo `endpoint: ModelEndpoint?` è stato aggiunto a `Conversation`
  con `effectiveEndpoint` resolver. I call site del codice esistente
  continuano a usare `modelDirPath` per compatibilità; sweep dei
  consumer è follow-up. Commit `e0abd40`.

- [x] **Streaming reasoning content visibile in tempo reale**. La
  `.streaming` phase ora carry `reasoningBuffer: String`; il remote
  loop chiama `updateRemoteReasoningBuffer` su ogni delta;
  `MessageView` preferisce il live buffer al persisted finché il
  `.done` non finalizza. Path locale ancora limitato (split
  `<think>` solo a `.done`). Commit `e0abd40`.

- [~] **Stop di un singolo sub-agent**. API surface chiusa:
  `cancelDelegation(frameID:)`, storage in `delegationCancellations`
  + `cancelledDelegations`. La runner-side (wrap di
  `runSubAgentToCompletionInner` in un Task + check `isCancelled`
  nel loop interno) è documentata come follow-up — la UI può già
  chiamare il metodo, marcando il frame come user-cancelled. Commit
  `3bfc079`.

---

## 5. Performance

Stime di speedup — vedi `docs/PERFORMANCE.md` per le metriche.

- [ ] **MLA multi-token forward con `startPos > 0`** → ~5-10× sui
  turn locali tool-heavy. Collapsa il delta post-tool da N
  single-token forwards a uno solo multi-token. Bloccato da
  `precondition(S == 1)` su `MLA.callAsFunction` e dal blit KV
  per `S > 1, startPos > 0`. Realizzabile dietro flag opt-in.

- [ ] **simdgroup_matrix BF16 GEMM** → ~5-10× su ogni `Linear`.

- [ ] **FlashAttention tiling per `sparse_attn`** → ~3-5×.

- [ ] **Persistent MoE dispatch kernel** → ~2× per layer MoE.

- [x] **Pipeline state caching** → ~10-50 ms / inference call.
  `Device.shared.makePipeline(_:)` ora cacha le pipeline e
  espone una variante `makePipeline(_:constants:)` per i kernel
  con function constants (vedi
  `Sources/DeepSeekKit/PipelineTuning.swift`). Hit identico
  cross-istanza per ActQuant/MoE.Gate/HCSinkhorn.

- [~] **Threadgroup sizing dinamico** ai siti `dispatchThreads`
  "free" (no SLM, no contratto kernel). Implementato via
  `MTLComputePipelineState.tunedThreadgroup(forGrid:)` su
  Linear (FP8/FP4/INT8/INT4/INT2), Elementwise (`dispatch1D`),
  HyperConnections (broadcast/collapse/compose), Compressor
  (5 siti), Indexer (2 siti), AttentionIndicesGPU.window,
  Attention.broadcast_row_mul. I siti a contratto (simdgroup
  32×1, riduzioni 256×1, SLM-bound) restano hardcoded — vedi
  commento di testata in `PipelineTuning.swift`.

- [ ] **KV cache pool** → multi-session serving.

- [~] **Cold-start prefetch**. Sfogliare sequenzialmente i shard
  per pre-popolare il page cache OS. Il path GGUF (`GGUFFile.init`)
  ora supporta `warmup: true` che applica `POSIX_MADV_WILLNEED`
  sull'intera mmap; safetensors path ancora pending. Commit `41385eb`.

- [ ] **B3 — KV cache persistence to disk**. Riprende una chat
  project-attached dopo un riavvio senza ri-prefilare il contesto
  del project.

- [ ] **Shader precompilation warmup (low priority)**. Le pipeline
  Metal sono già cache singleton (`Device.pipelineCache`) e
  always-in-memory una volta create, quindi questo NON sblocca
  use case nuovi — è solo una smoothing del primo turn. Idea:
  un `pipelineWarmupTask` background lanciato post-`loadModel`
  che precompila tutte le pipeline note (RoPE, SoftmaxAxis, TopK
  con tutti i `k` usati, GEMM bf16→f32, ecc.) **mentre l'utente
  sta scrivendo il prompt**. Al primo forward tutte le PSO sono
  già nella cache, niente JIT compile pause da 10-50ms su
  specializzazione mai vista. Stima: ~50 LOC in `Device.swift`
  + chiamata in `InferenceService.loadModel` post-success. Da
  fare con `Task.detached(priority: .background)` per non
  ostacolare l'UI.

---

## 6. Documentazione

- [x] **`README.md` riscritto** per coprire local + remote.
- [x] **`README.it.md` aggiornato** (traduzione del nuovo README).
- [x] **`docs/USAGE.md` aggiornato** per la triple-track (local
  GUI / remote GUI / local CLI).
- [x] **`docs/ISTRUZIONI.md` aggiornato** come tutorial italiano
  multi-track.
- [x] **`docs/README.md` aggiornato** come index multi-backend.
- [x] **`docs/MODULES.md` esteso** con tutta la mappa
  `Sources/DeepSeekUI/`.
- [x] **`docs/ARCHITECTURE.md` esteso** con sezione "Desktop app
  architecture".
- [x] **`docs/EXAMPLES.md` esteso** con ricette OpenRouter / MCP /
  agenti.
- [x] **`docs/DEVELOPING.md` esteso** con ricette §9-§12
  (backend / Settings tab / MCP transport / InferenceService
  method).
- [x] **`docs/ROADMAP.md` riscritto** in due stack (engine +
  desktop app).

- [ ] **Diagramma mermaid del decode pass** in
  `docs/ARCHITECTURE.md`.

- [ ] **Aggiornare `docs/PYTHON-MAPPING.md`** dopo i refactor di
  `Tensor` o quando vengono portati i task tokens / response_format.

- [ ] **`docs/PERFORMANCE.md` revisione** con i nuovi numeri post
  KV snapshot/restore e l'effetto MLA multi-token quando lander.

---

## 7. Testing

- [!] **`EndToEndForwardTests.swift`** — vedi §1, bloccato.

- [ ] **Test per Sampler** (top-K, top-P, repetition penalty,
  temperature).

- [ ] **Test per `EncodingDSV4`** sul golden corpus di
  `Reference/encoding/`.

- [ ] **Test per il converter** su un toy checkpoint sintetico.

- [ ] **Test per `OpenRouterClient`** con un mock URLSession
  (validateKey, fetchModels, streamChatCompletion).

- [ ] **Test per `MCPClient`** con un finto stdio server (Swift
  inline) per coprire l'handshake JSON-RPC + tools/list + un
  tools/call round-trip.

- [ ] **Test per `ChatStore.runRemoteLoop`** con mock backend per
  verificare il tool-call loop, il cap iterazioni, la
  accumulazione di cost.

---

## 8. Code-agent tooling (DeepSeekTools target)

Toolbox nativo per agire su codice. Storia: vedi
`docs/GAP-ANALYSIS-OPENCODE.md` §6 e `docs/TOOLS.md`.

- [x] **Target `DeepSeekTools`** — protocollo `Tool`, `ToolRegistry`
  (actor, plan-mode filter, session permission cache), `ToolContext`,
  permission delegate, agent mode.

- [x] **Tool nativi** read / write / edit / glob / grep / shell /
  apply_patch / webfetch / repo_clone / repo_overview / plan / task /
  todo. Test smoke in `Tests/DeepSeekToolsTests/`.

- [x] **Wire tool registry into `InferenceService`**. `ChatStore`
  ora include gli schemi nativi (prefisso `native__`) in
  `composeToolSchemasJSON` (path locale) e `composeOpenAITools` (path
  remoto); `dispatchNativeTool` instrada le chiamate
  `native__<tool>` a `NativeToolHost.dispatch` con il root resolved
  dal project attaccato (fallback `$HOME`). Commit `52832ef`.

  Gap residuo: `AgentConfig.allowedToolNames` oggi filtra solo MCP.
  Aggiungere filtering anche per nativi serve uno sweep delle
  configurazioni esistenti (le allowlist conoscono solo nomi MCP).

- [x] **`websearch`** — Tavily / Brave / Serper aggiunti come
  `WebSearchProvider` conformances. `NativeToolHost.init` legge
  `AppSettingsKey.webSearchProvider` + la chiave matching
  (`KeychainAccount.{tavily,braveSearch,serper}APIKey`) e costruisce
  il provider; fallback su DuckDuckGo se la chiave manca. Commit
  `cfe94ee`. UI per il picker è follow-up.

- [ ] **`lsp` tool**. Stub registrato che ritorna `notImplemented`.
  Necessita: spawn `sourcekit-lsp` (per Swift) / `pyright` (Python)
  / `typescript-language-server` per file `.ts`/`.tsx`. Framing
  JSON-RPC simile a `MCPClient`. Operazioni minime: `definition`,
  `hover`, `references`, `diagnostics`.

- [~] **Sandbox `ShellTool`**. Toggle `AppSettingsKey.useShellSandbox`
  + Settings → Tools → "Initialize default profile" button + auto-
  wiring di `shellUsesSandbox:` su `DefaultTools.standard`. Il
  profilo default rimane strict; tuning per workflow specifici
  resta lavoro del singolo utente. Commit `4964470`.

- [x] **HTTP recorder wiring**. `HTTPRecorderURLProtocol`
  intercetta richieste sulle `URLSession` di `OpenRouterClient` e
  `AnthropicClient`. Modi `.off` (default, no-op), `.record`
  (forward + persist), `.replay` (sintetizza response da file).
  `HTTPRecorder.shared.configure(directory:mode:)` flippa lo stato.
  Commit `758a6cc`. Limitazione: streaming SSE catturato come
  single chunk (replay batch invece di event-by-event).

- [x] **Server mode / headless CLI**. T1 chiuso — vedi §10.1.
  `LocalServer` actor + routes `/v1/models` + `/v1/chat/completions`
  (stream + buffered) + tools[] passthrough + Settings → Server.

- [ ] **Slack bot completo**. `Sources/DeepSeekIntegrations/Slack/`
  ha solo un webhook one-shot. Mancano: Events API listener, OAuth,
  session keyed by `(team_id, channel_id)`. Dipendenza dal server
  mode (T1) ora soddisfatta — manca il bot vero. ~500+ LOC.

- [~] **Per-project `.deepseek/`**. Loader `ProjectOverlayLoader` +
  `ProjectOverlay` struct + `ChatStore.effectiveAgents()` /
  `currentProjectOverlay()` helpers. Carica
  `<projectRoot>/.deepseek/{agents,skills,slash}.json` e fa merge
  con i globali (project-local vince su name collision). Le view
  (agent picker, slash palette, skills tab) non chiamano ancora
  `effectiveAgents()` — sweep follow-up. Commit `a2f9dd9`.

- [x] **Inline rebind di keybinding**. Settings → Keys ora ha
  pulsante "Rebind" per riga + sheet `KeybindingRebindSheet` che
  cattura chord via `.onKeyPress(phases: .down)` (macOS 14+), con
  toggle modificatori, named-key recognition (return / escape /
  tab / space / delete / arrows), e conflict detection inline che
  marca il binding rivale come empty su overwrite. Commit `86b6583`.

- [x] **Custom theme editor**. Settings → Theme ha "Create custom
  theme…" button + sheet con 5 `ColorPicker` (round-trip via
  NSColor → #RRGGBB), TextField name / summary, segmented
  Appearance picker. `ThemeStore.addCustomTheme` /
  `removeCustomTheme`; built-ins protetti. Commit `86b6583`.

---

## 9. Multi-format / GGUF / chat-template dispatcher

Pezzi scaffoldati nel merge di llama.cpp-gap. Storia: vedi
`docs/GAP-ANALYSIS-LLAMACPP.md` e `docs/GGUF.md`.

- [x] **`ChatTemplate` protocol + DSV4 + Jinja2 subset
  (`JinjaChatTemplate`)**. Driver Jinja2 puro Swift ~900 LOC
  (variabili, for/if/elif/else, filtri base, `raise_exception`).
  Macro / set / include / inheritance deferred.

- [x] **Tokenizer dispatcher** in `TokenizerLoader.load(tokenizerDir:)`
  con BPE / SentencePiece / WordPiece.

- [x] **Sampler suite estesa**: min-p, mirostat v2, tail-free
  sampling, locally-typical, frequency penalty, presence penalty
  (oltre al precedente top-K / top-P / repetition). CLI espone
  i flag corrispondenti. Test: `SamplerTests.swift`.

- [x] **GGUF reader MVP**: header parser v2/v3 + tensor info
  table + zero-copy load per dtype pass-through (`F32 / F16 /
  BF16 / I32 / I8`). Test: `GGUFTests.swift`.

- [x] **Kernel dequant per GGUF quantizzati**. `Q8_0`, `Q4_0`,
  `Q4_K` (= `Q4_K_M`), `Q5_K`, `Q6_K` con variante F32; `Q8_0` /
  `Q4_0` / `Q4_K` anche BF16 (Q5_K / Q6_K fallback automatico a F32).
  Vedi `Sources/DeepSeekKit/Kernels/dequant_gguf.metal`. `GGUFFile.load`
  ora dispatcha kernel-side invece di sollevare `unsupportedType`.
  Commit `3fb5bef` + `aadbd12`.

- [x] **Kernel transformer Llama-style** — `StandardMHA` (GQA +
  causal mask + KV cache), `SwiGLU`, `LlamaDecoderLayer`,
  `LlamaModel`, `LlamaConfig`, `ModelArchitecture` dispatcher,
  `LlamaModel.fromGGUF` factory, `TokenizerLoader.loadFromGGUF`,
  `LlamaStreamingModel` per il path `.streaming`. CLI
  `deepseek_gguf`. Commit `9024ce2` + `02571a7` + `41385eb`.

  Gap residuo: SDPA kernel oggi è `sdpa_naive_causal_gqa_f32`
  (streaming softmax ma no tiling). Una variante FlashAttention
  con `simdgroup_matrix` MMA darebbe ~5× su prefill. Drop-in nello
  stesso entry point Swift.

- [ ] **GGUF writer**. Simmetrico al reader; permetterebbe al
  converter di produrre un GGUF leggibile da llama.cpp. Low
  priority — interop bidirezionale.

- [ ] **`{% macro %}` / `{% set %}` / `include` / inheritance**
  in `JinjaTemplate.swift`. Nessuno dei template in circolazione
  ne ha bisogno per ora; il giorno che inciampiamo in uno che li
  usa, le aggiungiamo.

---

## 10. Standard LLM compatibility — piano di implementazione

Piano consolidato dopo la review critica sugli standard LLM (branch
`claude/review-llm-standards-UuG3Y`). Cinque track ordinati per ROI:
T1 (server) sblocca testabilità esterna di tutti gli altri; T2 (GGUF
forward) è il più grosso ed è parallelizzabile. Effort stimati a
"uomo-settimana" single developer focused.

### 10.1 Server HTTP OpenAI-compatible — T1 ✅ LANDED

Blocker per integrazione esterna (VS Code / Zed / TUI / Slack /
GitHub Actions). Sostituisce e dettaglia la voce "Server mode" in §8.

- [x] **`LocalServer` actor** via `Network.framework` `NWListener`
  — no SwiftNIO dep. Commit `932639f`.
- [x] **`POST /v1/chat/completions` non-streaming + SSE streaming**
  via `OpenAIRequest`. Commit `7539cf3`.
- [x] **`GET /v1/models`** — modello locale caricato come catalogo
  singleton. Commit `7539cf3`.
- [x] **`tools[]` passthrough** + tool-call loop (cap 8) +
  fallback graceful. Commit `7421e54`.
- [x] **Settings → Server tab** con port / bind addr / toggle /
  optional bearer token in Keychain + auto-start on launch
  se enabled. Commit `6410fb4`.
- [x] **curl recipe** in `docs/EXAMPLES.md` §19. Test integrazione
  esplicitamente saltato per preferenza del committer; manca
  `LocalServerTests.swift`. Commit `7421e54`.

### 10.2 GGUF run-forward (dequant + Llama) — T2 ✅ LANDED

Estende §9 "Kernel dequant" + "Kernel transformer Llama-style" con
piano sequenziato.

- [x] **Q8_0 / Q4_0 / Q4_K_M dequant kernels** + varianti BF16 per
  i tre (output dtype configurabile via `outputDtype:`). Commit
  `3fb5bef` + `aadbd12`.
- [x] **Q5_K + Q6_K dequant kernels** (F32 output; BF16 fallback su
  F32 per now). Commit `aadbd12`.
- [x] **`GGUFFile.load(outputDtype:)`** dispatcha al kernel matching.
- [x] **`LlamaDecoderLayer`** + `StandardMHA` (con GQA + causal +
  KV cache) + `SwiGLU`. Commit `9024ce2`.
- [x] **`LlamaModel` + `LlamaConfig` + `ModelArchitecture`
  dispatcher** + `LlamaModel.fromGGUF` factory. Commit `02571a7`.
- [x] **`LlamaStreamingModel`** per `--load-strategy streaming`:
  weights dequantati per-forward + `madvise(DONTNEED)` per layer.
  Commit `41385eb`.
- [x] **Tokenizer auto-detect** da GGUF metadata via
  `TokenizerLoader.loadFromGGUF` (gpt2/llama BPE; bert WordPiece
  rimane stub). Commit `9024ce2`.
- [x] **`deepseek_gguf` CLI** end-to-end con flag sampler completi
  + `--load-strategy mmap|preload|streaming` + `--weight-dtype
  f32|bf16` + `--use-map-shared` + `--warmup`. Commit `ddb6253` +
  `41385eb`.
- [ ] **End-to-end test TinyLlama Q4_K_M** vs llama.cpp reference.
  Skip per preferenza del committer.

### 10.3 Constrained decoding (JSON schema) — T3 ✅ LANDED

Output JSON garantito — gap killer dopo "no server". Subset
`response_format: {type:"json_schema"}` di OpenAI Structured Outputs.

- [x] **`SchemaCompiler` + `SchemaMask`** in `Sources/DeepSeekKit/`.
  Cartesian-product enumeration di `enum` / `const` / `oneOf` /
  `anyOf` / `type:"object"` (con caps su properties × values ≤ 8 ×
  32, total ≤ 4096) / `type:"array"` con `items` + `minItems` +
  `maxItems`. Commit `193682b` + `aadbd12` + `68095e9` + `2169e8d`.
- [x] **Mask injection in `Sampling.swift`** come stage 0a (prima
  di logitBias / temperature). `sample()` split in `sampleCore` +
  wrapper che chiama `mask.advance(token:)` dopo ogni pick.
- [x] **API binding `response_format`** su `LocalServer` via
  `extractResponseSchema(body:)`; resolve tokenizer + vocabSize da
  `InferenceService.snapshotTokenizerAndConfig()`.
- [x] **CLI flag `--json-schema <path>`** su `deepseek` +
  `deepseek_gguf`.
- [ ] **Regex `pattern` su `type:"string"`** — fuori scope cartesian
  product. Servirebbe un automaton character-class. ~200 LOC parser
  + automaton.
- [ ] **5 test golden** (object, array, enum, oneOf, pattern). Skip
  per preferenza del committer.

Stretch: **GBNF parser** stile llama.cpp per grammar arbitrarie. Non
bloccante — JSON Schema copre ~90 % dei casi d'uso.

### 10.4 Driver provider nativi — T4 ✅ LANDED

Smette di pagare il margine OpenRouter sui modelli più usati e abilita
prompt caching Anthropic (chiude la voce §4).

- [x] **`AnthropicClient`** Messages API + SSE streaming → traduce
  in `OpenAIStreamChunk` (text_delta / input_json_delta /
  thinking_delta / message_delta) così `ChatStore.runRemoteLoop`
  consuma entrambi i provider uniformemente. Commit `f43188c`.
- [x] **`ModelEndpoint.anthropic(modelID:)`** + `KeychainAccount
  .anthropicAPIKey` + `ModelState.loadRemoteAnthropic` +
  `ChatStore.RemoteProvider` enum + Settings → API Keys
  ProviderKeySection generica.
- [x] **Format translation** in `AnthropicMessageBuilder`
  (system separato; tool_use/tool_result content-block array;
  role="user" per tool results).
- [x] **`cache_control` auto-injection** su system block (quando
  ≥ 3500 chars) + ultimo tool_result block.
- [x] **"Add Anthropic model…" sheet** con 4 modelli suggeriti +
  custom id.

Stretch: driver **Ollama localhost** (`/api/chat`). Non implementato.

### 10.5 Sampler residui — T5 ✅ LANDED

Tutti in `Sources/DeepSeekKit/Sampling.swift` + CLI flag su
`deepseek` + `deepseek_gguf`. Commit `5a80d4b`.

- [x] **DRY sampler** — bounded scan O(H × L_max), history cap 1024,
  match cap 32 token. Stage 2c dopo frequency/presence.
- [x] **`logitBias: [Int32: Float]`** stage 0a (additivo, prima di
  temperature; `-100` è hard block regardless of T).
- [x] **Mirostat v1** — variante smoothed con window-average di `m`
  surprise recenti vs single-step di v2.
- [x] **CLI flag**: `--logit-bias '<JSON>'`, `--dry-multiplier`,
  `--dry-base`, `--dry-allowed-length`, `--mirostat-v1`,
  `--mirostat-m`. Test golden saltati per preferenza del committer.

### 10.6 Sequenziamento (storico) — tutte le 5 track landed

Pianificazione originale conservata per riferimento; l'effettivo
landing è avvenuto in un'unica burst su `claude/read-todo-C9gW9`
(commit `932639f` → `4964470`).

```
Settimana 1-2:  T1 (server)        — landed in 4 commit
Settimana 3:    T4 + T5 in parallelo (cheap, indipendenti)
Settimana 4-6:  T2 (GGUF + Llama)  — landed in 5 commit + factory
                 in `LlamaModel.fromGGUF`
Settimana 7-8:  T3 (constrained)   — enum/const/oneOf/anyOf/object/array
```

### 10.6.bis Vocab + Expert pruning italiano-only (in scope)

- [x] **Vocab pruning** per use case italiano-only / latino-only:
  riduce le matrici `embed.weight` e `head.weight` da ~129k a
  ~32-50k token senza fine-tuning. Implementato come target SPM
  separato `DeepSeekVocabPruner` + CLI `vocab_pruner` + sheet UI
  `VocabPrunerSheet` (bottone toolbar "Prune vocab…" accanto a
  Convert/Fine-tune). Vedi `docs/VOCAB-PRUNING.md`. Risparmio
  atteso: ~1-1.5 GB per checkpoint V4-Flash a bf16.
  **Fine-tuning resta fuori scope** (vedi §10.7); il pruning non
  lo richiede.

- [x] **Expert pruning** sopra il vocab pruning: droppa esperti
  MoE raramente attivati su un corpus IT di calibrazione. Stesso
  CLI `vocab_pruner` con flag `--prune-experts` + `--calib-corpus`
  + `--expert-coverage 0.99`. Le due fasi sono indipendenti
  (entrambe opt-in via flag): si possono pipelinare in un singolo
  comando o eseguire separatamente. Per resume da un vocab-prune
  esistente, basta passare la directory già pruned come `--input-dir`
  con solo le flag dell'expert phase. Implementazione: hook
  `routingObserver` in `MoEFFN.callAsFunction` (costo zero
  sull'inference path), nuovo facade `ExpertPruner` + checkpoint
  `expert_pruner.json` sibling. Vedi `docs/VOCAB-PRUNING.md`.
  Risparmio atteso: ~5-30 GB su V4-Flash production a seconda
  di `--expert-coverage`. UI non ancora wired (l'expert phase
  richiede `Transformer.load`, che pesa nello sheet — follow-up).

### 10.7 Out-of-scope deliberato

Standard identificati nella review ma fuori scope per scelta
strategica. Da rivalutare solo se cambia il target del progetto:

- **Multimodale** (vision/audio in/out). V4-Flash non ha encoder.
- **LoRA / QLoRA / PEFT**. Fine-tuning + serving non è il target.
- **Embedding / reranker**. Modelli architetturalmente diversi.
- **Mamba / RWKV / SSM**. State-space, non transformer.
- **Driver Gemini / Cohere / Bedrock nativi**. OpenRouter li copre.
- **GGUF writer**. Solo se serve interop bidirezionale.
- **A2A protocol** (Google). MCP è sufficiente.
- **EXL2/EXL3, MLX, ONNX, AWQ/GPTQ packed**. Non target
  Apple-Silicon-Metal o richiedono runtime separato.
- **Continuous batching / paged attention**. Single-user offline è
  l'attuale modo d'uso; rivalutare se §10.1 prende traffico serio.
- **DRY / XTC** in versione completa con tutte le tunable di
  koboldcpp. Subset minimo in T5.

---

## 11. Follow-up impegnativi rimasti

Cose esplicitamente lasciate aperte dopo il giro di lavoro sul
branch `claude/read-todo-C9gW9`. Ognuna è un'unità di lavoro
focused (multi-day) che non rientrava in una sessione conversazionale.

### Engine performance

- **SDPA tiled (FlashAttention-style)** per `StandardMHA`. Il kernel
  `sdpa_naive_causal_gqa_f32` è corretto ma 1-thread-per-output-row;
  una variante con threadgroup memory + `simdgroup_matrix<bfloat>`
  MMA + softmax rescale darebbe ~5× su prefill Llama lungo. ~300-400
  LOC Metal. Drop-in nello stesso entry point Swift.

- **MLA multi-token forward con `startPos > 0`** (§5). Rimuove la
  `precondition(S == 1)` in `Attention.swift:180` + adatta il blit
  KV per `S > 1, startPos > 0`. ~5-10× sui turn tool-heavy.

- **`simdgroup_matrix` BF16 GEMM** (§5). Sostituisce il path BF16
  in `Linear.callAsFunction` con MMA tile-based. ~5-10× per ogni
  Linear V4.

- **Sub-agent KV cache reuse** per `(agentID, promptHash)` (§4).
  Refactor di `InferenceService` cache layer per evitare cold
  prefill ad ogni rievocazione dello stesso sub-agent nella stessa
  turn.

### Integrazioni

- **LSP tool** (§8). Spawn `sourcekit-lsp` / `pyright` /
  `typescript-language-server` + framing JSON-RPC simile a
  `MCPClient` + operazioni `definition` / `hover` / `references` /
  `diagnostics`. ~400-500 LOC.

- **Slack bot completo** (§8). OAuth + Events API listener + session
  keyed by `(team_id, channel_id)`. Dipendenza T1 server soddisfatta.
  ~500+ LOC.

- **GGUF writer** (§9, §10.7). Simmetrico al reader. Tradeoff
  esplicito in §10.7 — solo se servirà interop bidirezionale.

### Round-trip configurazione calibrata

- **`inverseChannelScale` round-trip safetensors**. AWQ/SmoothQuant
  oggi calcolano il vettore di scale per-canale e lo scartano. Per
  rendere il quant matematicamente esatto a inference (e non solo
  "QAT warm-start") serve:
  1. Writer in `converter/main.swift` che emette
     `<layer>.inv_channel_scale` come tensor F32.
  2. Reader in `WeightLoader` / `Assembly.swift` che lo legge e
     popola `Linear.inverseChannelScale` al load.
  ~200 LOC across the two sides.

- **Router-aware MoE calibration**. Oggi tutti gli expert ricevono
  le stesse stats `yNorm2` (pre-routing); per quant calibrato
  per-expert serve instrumentation in `MoEFFN.callAsFunction` che
  emetta gli input per-routed-token. Sotanziale.

### Test (esplicitamente saltati per scelta del committer)

Tutti i 7 item di §7 — Sampler, EncodingDSV4, converter,
OpenRouterClient, MCPClient, ChatStore.runRemoteLoop, e
`EndToEndForwardTests` (quest'ultimo bloccato dalla mancanza di
PyTorch reference).

### UI sweep necessari (data layer pronto, view ancora cablano i vecchi accessor)

- Agent picker / slash palette / skills tab → migrare da
  `agents.agents` a `ChatStore.effectiveAgents()` per pickare la
  overlay `.deepseek/` quando il chat ha un Project.
- Remote crash recovery banner: legge `Conversation.remotePendingTurn`
  + pulsante "Retry" che reinvia il `userText` salvato.
- Chain UI per-frame Stop button: legge `delegationCancellations` +
  pulsante che chiama `ChatStore.cancelDelegation(frameID:)`.
- Settings → API Keys: aggiungere campi per Tavily / Brave / Serper
  + picker per `webSearchProvider`. Keys già scrivibili via Keychain
  programmaticamente.

---

## Come contribuire

1. Apri un item, controlla `docs/ROADMAP.md` per il contesto
   completo e `docs/PERFORMANCE.md` per le metriche di baseline.
2. Le convenzioni sono in `docs/DEVELOPING.md` (engine §1-§8,
   desktop app §9-§12).
3. Per kernel nuovi: aggiungi sempre `referenceCPU` + XCTest.
4. Per nuove feature UI: aggiungi anche una ricetta in
   `docs/EXAMPLES.md` se è scriptable da codice.
5. Aggiorna questo file: sposta l'item a `[~]` con un branch link,
   o a `[x]` con un commit hash quando chiuso.
