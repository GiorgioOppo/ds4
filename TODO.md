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

- [~] **Quantizzazione calibrata (GPTQ / AWQ / SmoothQuant)** sui
  pesi INT8/INT4. RTN attuale è il baseline; calibrazione recupera
  ~1-2 punti di perplexity. Scaffold in
  `Sources/DeepSeekKit/CalibratedQuant.swift`:
  - `QuantMethod` enum (`.rtn` / `.awq` / `.smoothQuant` / `.gptq`).
  - `CalibrationStats` struct (per-channel absmax + opzionalmente mean).
  - `ActivationObserver` (hook accumulatore per il forward pass —
    NON ancora wirato in `Linear.swift`).
  - `quantizeBF16ToInt8Calibrated(method:stats:)` entry point:
    - `.rtn` → delega a `quantizeBF16ToInt8` esistente (no-op).
    - `.awq` → implementazione preview (smoothing per-canale
      `s = (act_amax^α · w_amax^(1-α))/geomMean` → moltiplica i
      pesi → RTN). **Limite documentato**: manca l'inverse-scale
      sulle activations a runtime — vedi nota in
      `awqQuantizeBF16ToInt8`.
    - `.smoothQuant` / `.gptq` → `throw QuantNotImplemented` stub.
  Follow-up necessari per la chiusura completa:
  1. Wire `ActivationObserver` in `Linear.forward` (capture
     opzionale durante calibration runs).
  2. Aggiungere `inverseChannelScale: Tensor?` a `Linear` + pre-mul
     nel forward, così l'AWQ smoothing si bilancia esattamente.
  3. Implementare SmoothQuant (più semplice di GPTQ — solo
     smoothing per-canale come AWQ ma con `α` fisso 0.5 e nessun
     activation re-scale runtime). Stub già in posto.
  4. Implementare GPTQ (algoritmo OBS layer-by-layer con Hessian
     approssimata — il più costoso). Stub già in posto.
  5. Wire `--quant-method awq|gptq|...` nel converter CLI.

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

- [ ] **`response_format` schema injection** (encoding_dsv4.py:49)
  non portato.

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

- [ ] **Cross-agent delegation on remote chats**. Lo schema
  `__delegate_to_agent` non è injected su remoto. Una versione
  remota richiede un loop sub-agent remoto. ~200 LOC se serve.

- [ ] **Remote crash recovery**. `pendingTurn` non viene riarmato
  per `sendRemote`. Workaround: l'utente reinvia il messaggio
  (cheap, idempotente lato provider).

- [ ] **Prompt-caching Anthropic via OpenRouter**. Il body non
  passa headers cache-control. Cheap da aggiungere.

- [ ] **Sub-agent cross-delega reuse**. Una sub-agent invocata due
  volte nella stessa turn paga il cold prefill ogni volta. Cache
  per `(agentID, promptHash)` aggiuntiva (di nicchia).

- [ ] **Conversation.modelDirPath → endpointID** migration. Oggi
  per chat remote `modelDirPath = ""`. Cosmetico.

- [ ] **Streaming reasoning content visibile in tempo reale** nel
  buffer della bolla (oggi solo a `.done`).

- [ ] **Stop di un singolo sub-agent** dalla chain UI (oggi solo
  Stop globale).

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

- [ ] **Cold-start prefetch**. Sfogliare sequenzialmente i shard
  per pre-popolare il page cache OS.

- [ ] **B3 — KV cache persistence to disk**. Riprende una chat
  project-attached dopo un riavvio senza ri-prefilare il contesto
  del project.

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

- [~] **Wire tool registry into `InferenceService`**. Il registry
  esiste in `NativeToolHost`, ma `InferenceService` ancora costruisce
  il blocco `tools` solo dagli MCP. Mancano:
  1. Merge degli schemi nativi con quelli MCP nel system block /
     OpenAI `tools` array (key prefix `native__<name>`?).
  2. Routing del `tools/call` al `NativeToolHost.dispatch` quando il
     nome è nativo.
  3. Resolve `rootDirectory` dal Project attaccato, o dalla home
     dell'utente come fallback.

- [~] **`websearch`** — il backend di default (DuckDuckGo lite scraper)
  funziona ma è fragile. Aggiungere provider configurabili (Tavily /
  Brave / Serper) con API key in `Keychain`.

- [ ] **`lsp` tool**. Stub registrato che ritorna `notImplemented`.
  Necessita: spawn `sourcekit-lsp` (per Swift) / `pyright` (Python)
  / `typescript-language-server` per file `.ts`/`.tsx`. Framing
  JSON-RPC simile a `MCPClient`. Operazioni minime: `definition`,
  `hover`, `references`, `diagnostics`.

- [ ] **Sandbox `ShellTool`**. `Sources/DeepSeekIntegrations/Sandbox/`
  scrive un profilo `sandbox-exec` base; `ShellTool(useSandbox:true)`
  lo cerca a `<root>/sandbox/default.sb`. Il profilo è
  deliberatamente strict (deny default); tunarlo per workflow di dev
  e abilitare il toggle nelle Settings.

- [ ] **HTTP recorder wiring**. `Sources/DeepSeekIntegrations/HTTPRecorder/`
  ha l'API ma non è collegato a `OpenRouterAPI`. Implementare come
  `URLProtocol` su una sessione opt-in.

- [ ] **Server mode / headless CLI**. Esporre `InferenceService` su
  `localhost:PORT` con un'API OpenAI-compatible. Sblocca:
  TUI client esterni, plugin VS Code / Zed, GitHub Actions
  `agent-review.yml` (oggi placeholder), Slack bot completo.
  **Piano dettagliato in §10.1 (T1).**

- [ ] **Slack bot completo**. `Sources/DeepSeekIntegrations/Slack/`
  ha solo un webhook one-shot. Mancano: Events API listener, OAuth,
  session keyed by `(team_id, channel_id)`, dipendenza dal server
  mode sopra.

- [ ] **Per-project `.deepseek/`**. Carica agent / skill / slash
  command da un percorso versionabile nel repo target, sovrascrivendo
  i default globali. Pattern di `.opencode/` e `CLAUDE.md`.

- [ ] **Inline rebind di keybinding**. La tab `Keys` è oggi
  read-only + reset; aggiungere un widget di key-grab + detection
  conflitti + conferma overwrite delle scorciatoie di sistema.

- [ ] **Custom theme editor**. Oggi `ThemeStore` accetta temi custom
  via JSON ma non c'è UI per crearli. Aggiungere editor con
  ColorPicker per i sei slot (accent / background / foreground /
  bubble assistant / bubble user / appearance).

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

- [ ] **Kernel dequant per GGUF quantizzati** (`Q4_0`, `Q4_K`,
  `Q4_K_M`, `Q5_K`, `Q6_K`, `Q8_0`). Senza questi `GGUFFile.load`
  solleva `unsupportedType`. Pattern di partenza:
  `int4_gemm.metal` + `int8_gemm.metal`. Ogni formato ~2-3 giorni.
  **Piano dettagliato in §10.2 (T2).**

- [ ] **Kernel transformer Llama-style** (MHA + SwiGLU + RMSNorm,
  no MLA, no MoE, no HC). Pre-requisito perché il dispatcher di
  template + il reader GGUF facciano *girare* un modello Llama,
  non solo leggerlo. **Piano dettagliato in §10.2 (T2).**

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

### 10.1 Server HTTP OpenAI-compatible — T1 (~2 settimane)

Blocker per integrazione esterna (VS Code / Zed / TUI / Slack /
GitHub Actions). Sostituisce e dettaglia la voce "Server mode" in §8.

- [ ] **`LocalServer` actor**. SwiftNIO o `URLSession` server APIs;
  bind su `localhost:PORT`, configurabile. ~3d. Nuovo file:
  `Sources/DeepSeekUI/State/LocalServer.swift`.
- [ ] **`POST /v1/chat/completions` non-streaming**. Mapping
  `OpenAIRequest → InferenceService.generateForConversation`.
  Tokenizer attivo dal modello caricato. ~2d.
- [ ] **SSE streaming (`stream: true`)**. Delta chunks
  `data: {choices:[{delta:{...}}]}\n\n`, terminatore `data: [DONE]`.
  ~2d.
- [ ] **`GET /v1/models`**. Catalogo dal modello locale caricato +
  modelli OpenRouter conosciuti (opt-in). ~0.5d.
- [ ] **`tools[]` passthrough**. Array OpenAI → MCP/native registry;
  emissione `tool_calls` nella response. Riusa logica `ChatStore`.
  ~2d.
- [ ] **Settings → Server tab**. Port, bind addr, enable toggle,
  optional bearer token. ~1d. Nuovo file:
  `Sources/DeepSeekUI/Views/Settings/ServerSettingsTab.swift`.
- [ ] **Test integrazione + curl recipe**. `LocalServerTests.swift`
  + sezione in `docs/EXAMPLES.md`. ~1d.

### 10.2 GGUF run-forward (dequant + Llama) — T2 (~3 settimane)

Estende §9 "Kernel dequant" + "Kernel transformer Llama-style" con
piano sequenziato. Senza questo, il lavoro post-merge llama.cpp-gap
(GGUF reader, Jinja, multi-tokenizer) è inerte.

- [ ] **Q8_0 dequant kernel**. Simmetrico a `int8_gemm.metal` ma su
  blocchi GGUF da 32 elementi + scala F16. ~2d. Nuovo:
  `Sources/DeepSeekKit/Kernels/dequant_gguf.metal`.
- [ ] **Q4_0 + Q4_K_M dequant kernels**. Q4_K_M è super-block 256
  con 8 scale F16 + min F16. ~4d.
- [ ] **`GGUFFile.load` → BF16 staging tensor**. Sostituisce
  `unsupportedType` in `GGUFLoader.swift`. ~1d.
- [ ] **`LlamaDecoderLayer`** (MHA + RoPE + SwiGLU + RMSNorm). ~3d.
  Nuovo: `Sources/DeepSeekKit/Layers/LlamaDecoderLayer.swift`.
- [ ] **`LlamaModel` wrapper + `ModelArchitecture` dispatcher**.
  ~2d. Nuovi: `Sources/DeepSeekKit/{LlamaModel,ModelArchitecture}.swift`.
- [ ] **Tokenizer auto-detect** da GGUF metadata
  (`tokenizer.ggml.model` = "llama" / "gpt2" / "bert"). ~1d.
- [ ] **End-to-end test con TinyLlama-1.1B Q4_K_M**. Greedy decoding
  match vs llama.cpp reference. ~2d. Nuovo:
  `Tests/DeepSeekKitTests/LlamaForwardTests.swift`.

### 10.3 Constrained decoding (JSON schema) — T3 (~1.5 settimane)

Output JSON garantito — gap killer dopo "no server". Subset
`response_format: {type:"json_schema"}` di OpenAI Structured Outputs.

- [ ] **JSON-Schema → token-mask compiler**. Subset: `type`,
  `enum`, `oneOf`, `anyOf`, `properties`, `items`, `pattern`
  (regex semplice). ~4d. Nuovo:
  `Sources/DeepSeekKit/Sampling/SchemaCompiler.swift`.
- [ ] **Mask injection in `Sampling.swift`**. Nuovo campo
  `Sampler.schemaMask: SchemaMask?`, applicato stage 0 prima di
  temperature/top-K. Stato automa avanzato per token. ~2d. Nuovo:
  `Sources/DeepSeekKit/Sampling/SchemaMask.swift`.
- [ ] **API binding `response_format: {type:"json_schema"}`**. Sia
  su `LocalServer` (§10.1) che su CLI. ~1d.
- [ ] **CLI flag `--json-schema <path>`** in
  `Sources/deepseek/main.swift`. ~0.5d.
- [ ] **Test golden con 5 schemi** (object, array, enum, oneOf,
  pattern). ~1d.

Stretch: **GBNF parser** stile llama.cpp per grammar arbitrarie. Non
bloccante — JSON Schema copre ~90 % dei casi d'uso.

### 10.4 Driver provider nativi — T4 (~3 giorni)

Smette di pagare il margine OpenRouter sui modelli più usati e abilita
prompt caching Anthropic (oggi non passabile via OpenRouter, vedi
§4 "Prompt-caching Anthropic via OpenRouter").

- [ ] **`AnthropicAPI`** client (Messages API + SSE streaming). ~1d.
  Nuovo: `Sources/DeepSeekUI/State/AnthropicAPI.swift`.
- [ ] **`ModelEndpoint.anthropic` case** + Settings → API Keys
  Anthropic field. ~0.5d.
- [ ] **Format translation** OpenAI `tools[]` ↔ Anthropic
  `tool_use`/`tool_result` blocks. ~1d.
- [ ] **`cache_control` auto-injection** sul system block + ultimo
  tool result. Chiude voce §4 "Prompt-caching Anthropic via
  OpenRouter". ~0.5d.

Stretch: driver **Ollama localhost** (`/api/chat`). Utile per chi
gira llama.cpp via Ollama in parallelo al nostro engine. Stessa
struttura, no auth.

### 10.5 Sampler residui — T5 (~2 giorni)

- [ ] **DRY sampler** — penalty moltiplicativa su token che
  estenderebbero n-gram già visti. Parametri standard: `multiplier`,
  `base`, `allowed_length`. ~1d. In `Sources/DeepSeekKit/Sampling.swift`.
- [ ] **`logitBias: [Int32: Float]`** in `Sampler`. Stage 0
  (additivo prima di softmax). ~0.3d.
- [ ] **Mirostat v1** — algoritmo originale (oggi solo v2). ~0.5d.
- [ ] **CLI flag + test golden**. ~0.2d.

### 10.6 Sequenziamento consigliato

```
Settimana 1-2:  T1 (server)
Settimana 3:    T4 + T5 in parallelo (cheap, indipendenti)
Settimana 4-6:  T2 (GGUF + Llama) — può forkare in parallelo a T1
                 se 2 dev
Settimana 7-8:  T3 (constrained) — più valore quando T1 è on
```

### 10.6.bis Vocab pruning italiano-only (in scope)

- [x] **Vocab pruning** per use case italiano-only / latino-only:
  riduce le matrici `embed.weight` e `head.weight` da ~129k a
  ~32-50k token senza fine-tuning. Implementato come target SPM
  separato `DeepSeekVocabPruner` + CLI `vocab_pruner` + sheet UI
  `VocabPrunerSheet` (bottone toolbar "Prune vocab…" accanto a
  Convert/Fine-tune). Vedi `docs/VOCAB-PRUNING.md`. Risparmio
  atteso: ~1-1.5 GB per checkpoint V4-Flash a bf16.
  **Fine-tuning resta fuori scope** (vedi §10.7); il pruning non
  lo richiede.

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
