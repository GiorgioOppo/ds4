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

- [ ] **W8A8 (activations INT8)**. Follow-up di W8A16. Richiede:
  1. nuovo formato in `ActQuant` per quantizzare le attivazioni a
     INT8 con scala per-token o per-128;
  2. GEMM `int8 × int8 → int32` con rescale finale (Apple Silicon
     ha `simd_matrix<char,...>`, sfruttabile);
  3. branch alternativo in `Linear.int8Forward`.
  Beneficio atteso: throughput memory-bound dimezzato.

- [ ] **Quantizzazione calibrata (GPTQ / AWQ / SmoothQuant)** sui
  pesi INT8/INT4. RTN attuale è il baseline; calibrazione recupera
  ~1-2 punti di perplexity. Struttura di `Int8Quant.swift` lascia
  spazio per `quantizeBF16ToInt8Calibrated`.

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

- [ ] **Pipeline state caching** → ~10-50 ms / inference call.

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

## 7. Code-agent tooling (DeepSeekTools target)

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
