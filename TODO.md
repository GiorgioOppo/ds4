# TODO

Living checklist of outstanding work. Each item links to where the
corresponding background lives (mostly `docs/ROADMAP.md` and
`docs/PERFORMANCE.md`); read those for context before picking one up.

Legend: `[ ]` open · `[~]` partial · `[x]` done · `[!]` blocked

---

## 0. Quantizzazione

- [x] **`--target-dtype int8`** — INT8 W8A16 (peso INT8 simmetrico
  per-riga, gruppo K=128, scala F16; attivazioni F32/BF16 passano
  invariate). Quantizza solo i pesi `Linear` (whitelist in
  `shouldQuantizeToInt8`); embed/head/norm/attn_sink/hc_*/bias restano
  BF16. Footprint ≈ ½ × BF16 sui pesi Linear. Sorgenti supportate:
  BF16, F32, FP8+scala, FP4+scala. Kernel: `int8_gemm.metal`. Test:
  `Int8GemmTests.swift`, `Int8ConverterTests.swift`.

- [ ] **W8A8 (activations INT8)**. Follow-up di W8A16. Richiede:
  1. nuovo formato in `ActQuant` per quantizzare le attivazioni a
     INT8 con scala per-token o per-128;
  2. GEMM `int8 × int8 → int32` con rescale finale (Apple Silicon
     ha `simd_matrix<char,...>`, sfruttabile);
  3. branch alternativo in `Linear.int8Forward`.
  Beneficio atteso: throughput memory-bound dimezzato.

- [ ] **Quantizzazione calibrata (GPTQ / AWQ / SmoothQuant)** sui pesi
  INT8/INT4. RTN attuale è il baseline; calibrazione recupera ~1-2
  punti di perplexity. Struttura di `Int8Quant.swift` lascia spazio
  per `quantizeBF16ToInt8Calibrated`.

- [ ] **`--target-dtype int4`**. Branch
  `claude/convert-bf16-to-int4-7mO21`. Necessita un nuovo kernel
  `int4_gemm.metal` con unpacking nibble + scala per-32 o per-64.

---

## 1. Parità con il reference Python

- [ ] **act_quant noise sulle dimensioni non-rope di KV** in `MLA` e
  `Compressor`. Il reference applica `act_quant(kv[..., :-rope_dim], ...)`
  per iniettare rumore QAT. Bloccato dall'assenza di viste strided in
  `Tensor`: lo slice `kv[..., :-rope_dim]` lungo l'ultimo asse non è
  contiguo. Richiede refactor di `Tensor` (stride per dimensione +
  aggiornamento di tutti i kernel consumatori).
  Impatto stimato: < 1 % di differenza nel forward (rumore QAT è
  piccolo). Vedi `docs/ROADMAP.md` §"Tier 3 — Deferred / structural".

- [ ] **`cast_e2m1fn_to_e4m3fn` nel converter** (`--expert-dtype fp8`
  lossless re-encode). Non sul critical path: con
  `--target-dtype bf16` (default) gli esperti FP4 sono dequantizzati a
  BF16 e l'opzione non serve. Riattivare solo se si vogliono esperti
  FP8 su disco per risparmiare spazio rispetto a BF16.

- [!] **Validazione numerica end-to-end vs Python**. Richiede
  PyTorch + CUDA per dumpare attivazioni con
  `Reference/inference/generate.py` su un toy config (n_layers=2,
  dim=64) e confrontare con il forward Swift. Piano:
  1. `Reference/inference/dump_activations.py` (da scrivere) — dumpa
     ogni layer in JSON su un prompt fisso.
  2. `Tests/DeepSeekKitTests/EndToEndForwardTests.swift` (da scrivere)
     — carica il JSON, runna il forward Swift sullo stesso config,
     asserisce errore relativo < 1e-2 per layer.
  Bloccato da disponibilità di ambiente CUDA.

---

## 2. Funzionalità di runtime mancanti

- [ ] **Batched serving** nel CLI. `Transformer.forward` è già shaped
  per `[B, S]` ma `Sources/deepseek/main.swift` itera una singola
  prompt. Rework CLI-level, non del modello.

- [ ] **Speculative decoding via MTP**. `MTPBlock.callAsFunction` è
  implementato, il CLI non lo invoca. Wire-in:
  1. Dopo lo stack di block standard, runnare ogni `MTPBlock` per
     ottenere logits speculativi.
  2. Campionare N token candidati.
  3. Verificare con il forward successivo. Accettare il prefix che
     combacia, retry dal primo mismatch.

- [ ] **Multi-rank loader**. Converter e loader assumono
  `model_parallel == 1`. Il reference supporta sharding multi-rank
  via `model{i}-mp{N}.safetensors`. Non bloccante per Mac inference,
  blocca uso distribuito.

- [ ] **`bf16` ParallelEmbedding**. `ParallelEmbedding.init` ha
  precondition `weight.dtype == .f32`. Dopo `--target-dtype bf16`
  l'embed.weight è BF16 e `Transformer.load` lo salta (fallback a
  random init). Fix: rilassare la precondition + aggiungere ramo
  lookup BF16 (~20 LOC). Workaround attuale: usare
  `--target-dtype keep`.

- [ ] **Restore `wo_a` non fuso**. Il converter fonde sempre
  `wo_a.weight + scale → BF16` indipendentemente da `--target-dtype`,
  perché `MLA.forward` usa `Einsum.bsgdGrd` (FP32) invece di
  `Linear`. Per tenere wo_a in FP8 serve un einsum FP8-aware o
  ripristinare il path Linear.

---

## 3. Encoding (chat / DSML)

- [ ] **Task tokens** (`<｜action｜>`, `<｜query｜>`, ecc. da
  `DS_TASK_SP_TOKENS`) non emessi. Workaround: il caller prepende
  manualmente.

- [ ] **`response_format` schema injection** (encoding_dsv4.py:49)
  non portato. Stesso workaround: prepend nel system message.

- [ ] **`latest_reminder` token** (encoding_dsv4.py:25) non emesso.
  Stesso workaround.

- [ ] **Thinking mode `.high`** distinto da `.chat`. Attualmente
  `.high` si comporta come `.chat`. Il reference Python aggiunge un
  prompt di reasoning meno estremo del `.max`.

---

## 4. Performance (tutti i kernel sono "correctness-first")

Numeri di speedup atteso sono stime — vedi `docs/PERFORMANCE.md` per
le metriche dettagliate e i mini-spec per ogni voce.

- [ ] **simdgroup_matrix BF16 GEMM** → ~5-10× su ogni `Linear`.
  Tile 32×32 con `simdgroup_load_explicit` / `simdgroup_multiply`.
  Riscrittura di `gemm_bf16.metal`.

- [ ] **FlashAttention tiling per `sparse_attn`** → ~3-5×.
  Attualmente one thread per `(b, m, h)`, serial sui K topk. Goal:
  tile sui K, online softmax su tiled blocks, riduzione cooperativa.

- [ ] **Persistent MoE dispatch kernel** → ~2× per layer MoE.
  Attualmente `MoEDispatch.prepare` costruisce assignment tables su
  CPU. Mossa a GPU con un kernel persistente che produce gather/
  scatter indices in-place.

- [ ] **Pipeline state caching** → ~10-50 ms risparmiati per
  inference call. Cache di `MTLComputePipelineState` per
  (kernelName, functionConstants) chiavi. Vedi `Device.swift`.

- [ ] **KV cache pool** → conta per multi-session serving. Allocare
  KV cache buffers da un pool riutilizzabile invece di
  `device.makeBuffer` ogni init.

- [ ] **Cold-start prefetch**. Prima del primo forward, sfogliare
  sequenzialmente i shard nell'ordine layer per pre-popolare il page
  cache OS. Riduce la latency del primo token.

---

## 5. Documentazione

- [ ] **Diagramma mermaid del decode pass** in
  `docs/ARCHITECTURE.md`. Pending in piano docs (marcato opzionale).

- [ ] **Aggiornare `docs/PYTHON-MAPPING.md`** dopo i refactor di
  `Tensor` o quando vengono portati i task tokens / response_format.

- [ ] **Tradurre `ISTRUZIONI.md` in inglese** (`SETUP.md`?). Oggi è
  solo italiano.

---

## 6. Testing

- [ ] **`EndToEndForwardTests.swift`** — vedi §1, bloccato da
  validazione Python.

- [ ] **Test per Sampler** (top-K, top-P, repetition penalty,
  temperature). Oggi non c'è un `SamplingTests.swift` dedicato.

- [ ] **Test per `EncodingDSV4`** sul golden corpus di
  `Reference/encoding/`. Comparare token-by-token con la generazione
  del Python reference.

- [ ] **Test per il converter** su un toy checkpoint sintetico
  (n_layers=2). Verificare che il roundtrip
  `HF → convert → load → forward` non perda significativamente
  precisione.

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
2. Le convenzioni sono in `docs/DEVELOPING.md`.
3. Aggiungi sempre `referenceCPU` + test XCTest per ogni nuovo
   kernel.
4. Aggiorna questo file: sposta l'item a `[~]` con un branch link, o
   a `[x]` con un commit hash quando chiuso.
