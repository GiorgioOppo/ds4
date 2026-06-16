# Architettura del motore — DeepSeek V4 (DS4Core + DS4Metal)

Documentazione tecnica di dettaglio del **motore di inferenza in Swift puro**
che alimenta sia la demo CLI `DS4Demo` sia l'app `DwarfStar`. Spiega come
funzionano l'**encoder** (tokenizer + caricamento modello), il **decoder**
(forward pass token-per-token su Metal), e tutte le altre parti salienti
(runtime Metal, grafo di calcolo, compressore NSA, sampler, quantizzazione,
streaming dei pesi).

Il motore è un **port fedele in Swift** del motore C `ds4.c` / `ds4_metal.m`:
ogni componente è annotato nel sorgente con la funzione C da cui deriva e
validato contro l'output del motore C (vedi `Tests/DS4CoreTests/`). L'obiettivo
n.1 è la **correttezza numerica** bit-per-bit sulla stessa piattaforma.

> Documenti correlati: [`DOCUMENTAZIONE.md`](DOCUMENTAZIONE.md) (uso utente,
> demo e UI) · [`../README.md`](../README.md) (build & packaging).

---

## Indice

1. [Pipeline end-to-end](#1-pipeline-end-to-end)
2. [Encoder — caricamento del modello (GGUF)](#2-encoder--caricamento-del-modello-gguf)
3. [Encoder — il tokenizer](#3-encoder--il-tokenizer)
4. [Forma del modello e validazione (`ModelShape`)](#4-forma-del-modello-e-validazione-modelshape)
5. [Substrato di esecuzione Metal](#5-substrato-di-esecuzione-metal)
6. [Il decoder — forward pass completo](#6-il-decoder--forward-pass-completo)
7. [Il layer di decode in dettaglio](#7-il-layer-di-decode-in-dettaglio)
8. [Il compressore NSA](#8-il-compressore-nsa)
9. [Il MoE: router ed expert](#9-il-moe-router-ed-expert)
10. [StreamingDecoder e strategie di streaming](#10-streamingdecoder-e-strategie-di-streaming)
11. [Decoder — dai logit al token (`Sampler`)](#11-decoder--dai-logit-al-token-sampler)
12. [Quantizzazione](#12-quantizzazione)
13. [Riepilogo dei tensori per-layer](#13-riepilogo-dei-tensori-per-layer)
14. [Tool calling (function calling)](#14-tool-calling-function-calling)
15. [Inferenza distribuita](#15-inferenza-distribuita)

---

## 1. Pipeline end-to-end

```
 testo utente
     │
     ▼
┌──────────────┐   ids        ┌───────────────────────────────────────────┐
│  Tokenizer   │─────────────▶│  Decoder (per ogni token, posizione pos)  │
│ (encoder)    │              │                                           │
└──────────────┘              │  embed → N decode layer → output head     │
     ▲                         │                    │                      │
     │ tokenText()             │                    ▼                      │
     │                         │                 logits[vocab]             │
┌──────────────┐   token       └────────────────────┬──────────────────────┘
│   Sampler    │◀───────────────────────────────────┘
│ (decoder     │   campiona il prossimo token
│  dei logit)  │
└──────────────┘
     │  loop autoregressivo (prefill dei prompt, poi decode)
     ▼
 testo generato
```

I pesi del modello vivono in un file **GGUF** mappato in memoria (`mmap`); il
`GGUFModel` espone i tensori per offset assoluto. Il **decoder** li carica
nei `GPUTensor` (buffer Metal) e dispatcha i kernel.

Le quattro grandezze che attraversano tutto:

- **token** — id intero nel vocabolario (`vocab = 129 280`).
- **pos** — posizione assoluta del token nella sequenza (per RoPE e KV cache).
- **nKeys** — numero di righe KV valide (`pos+1`); per il flash-attention
  senza padding deve essere multiplo di 32.
- **logits** — vettore `[vocab]` di punteggi prodotto da un forward.

---

## 2. Encoder — caricamento del modello (GGUF)

**File:** `Sources/DS4Core/Format/GGUF.swift`

`GGUFModel` è un port fedele del loader GGUF di `ds4.c`:

- il file viene **`mmap`-pato una sola volta**; i byte dei tensori restano in
  posizione e i chiamanti raggiungono i pesi per **offset assoluto** dentro la
  mappatura (`model.mapBase + t.absOffset`);
- supporta **solo GGUF v3** (gli altri lanciano `GGUFError`);
- espone i **metadati** (KV store) con accessor tipizzati (`u32`, `f32Compat`,
  `intArray`, `stringArrayBytes`, …) e la **tabella dei tensori** (`findTensor`,
  che restituisce tipo, offset, numero di elementi e byte);
- `metalMapping: true` apre la mappatura in `MAP_SHARED` così i `GPUTensor`
  possono creare viste **no-copy** sui pesi (vedi §5 e §10).

A differenza del C, dove un errore chiama `ds4_die()` (abort del processo), qui
ogni errore viene lanciato come `GGUFError`; le chiavi dei metadati e i nomi dei
tensori sono decodificati in `String` al parse-time (le lookup restano
identiche).

### Caricamento dei pesi nei `GPUTensor`

**File:** `Sources/DS4Metal/Model/GGUFWeights.swift`

`GGUFWeights` è il ponte da GGUF → GPU:

- `tensor(rt, model, name)` carica i byte grezzi di un tensore in un `GPUTensor`;
- `layer(rt, model, il)` assembla **tutti** i tensori `blk.<il>.*` di un decode
  layer in una `LayerWeights` (vedi §13); con `loadExperts: false` lascia i tre
  tensori-expert da 256 come segnaposto (il percorso expert-cache raccoglie solo
  i 6 expert selezionati dopo il routing — risparmia ~3,6 GB per layer);
- `outputHead(...)` assembla embedding table + testa di output;
- `gatherExperts(...)` copia dalla `mmap` **solo le righe dei 6 expert
  selezionati**;
- `detectMoEQuant(model)` ispeziona il GGUF per dedurre lo schema di
  quantizzazione di gate/up/down e se il router è F16 (vedi §12); questo
  configura quali kernel MoE il decoder dispatcha.

Esistono varianti **mapped** (`layerMappedDense`, `outputHeadMapped`,
`layerMappedExperts`) che creano **viste no-copy** sui pesi invece di copiarli
in RAM — la base dello streaming SSD.

---

## 3. Encoder — il tokenizer

**File:** `Sources/DS4Core/Inference/Tokenizer.swift`

Port fedele del tokenizer di `ds4.c`: **BPE byte-level in stile GPT-2** con il
pre-tokenizer JoyAI/DeepSeek e la consapevolezza dei token speciali. Le tabelle
sono indicizzate per **byte esatti**, identiche al `str_i32_table` del C
(inclusa la fallback a singolo byte). Validato con `./ds4 --dump-tokens`.

### Componenti

1. **Mappatura byte ↔ codepoint (`ByteLevel`)** — `byteToCodepoint` /
   `codepointToByte` mappano i 256 byte su codepoint stampabili (il trucco
   GPT-2 per evitare byte di controllo), con utility UTF-8.
2. **Vocabolario e merge** — caricati dai metadati GGUF
   `tokenizer.ggml.tokens` (id → byte) e `tokenizer.ggml.merges` (coppia →
   rango). Vengono risolti gli id dei **token speciali**: `bos`, `eos`,
   `User`, `Assistant`, `<think>`, `</think>`, `｜DSML｜`.
3. **Pre-tokenizer (`preTokenize`)** — replica esattamente `bpe_tokenize_text`:
   spezza il testo in *pezzi* secondo regole su cifre (max 3), CJK,
   punteggiatura+lettera, sequenze di lettere (incluse non-ASCII),
   spazi/newline. Ogni pezzo passa a `emitPiece`.
4. **BPE (`emitPiece`)** — `byteEncode` mappa i byte in codepoint stampabili,
   poi merge ripetuti dalla coppia col **rango più basso** finché non ce ne
   sono più; ogni simbolo finale → id (con fallback per-byte se il simbolo non
   è in vocabolario).

### API pubblica

- `tokenize(text)` — testo semplice, nessun riconoscimento di special.
- `tokenizeRenderedChat(text)` — riconosce i token speciali *letterali* nel
  testo già renderizzato (porta di `tokenize_rendered_chat_vocab`).
- **`encodeChatPrompt(system:prompt:think:)`** — costruisce il prompt di chat
  (porta di `encode_chat_prompt`):

  ```
  [bos]
  ( prefisso "Reasoning Effort: …"  se think == .max )
  ( system prompt                   se presente       )
  [User]  <token del prompt utente>
  [Assistant]
  [<think>]   se think abilitato,  altrimenti  [</think>]
  ```

  È esattamente la sequenza che `DS4Demo` e `InferenceService` danno in pasto al
  decoder. Quando il **Thinking** è attivo il prompt termina con `<think>`, e il
  decoder emette `reasoning` finché non incontra `</think>` (`thinkEndId`).
- **`tokenText(id)`** — detokenizzazione (porta di `ds4_token_text`): ricostruisce
  i byte di output di un token, gestendo i token speciali *letterali* (quelli che
  contengono la barra fullwidth `U+FF5C`) che vengono emessi verbatim. È ciò che
  trasforma i token generati in testo visibile, byte per byte, in streaming.

`ThinkMode` ha tre livelli: `.none`, `.high`, `.max`. Solo `.max` inietta il
prefisso "Reasoning Effort" verbatim (`DS4ReasoningEffortMaxPrefix`). La GUI usa
`.none` / `.high`.

---

## 4. Forma del modello e validazione (`ModelShape`)

**File:** `Sources/DS4Core/Inference/ModelShape.swift`

Port di `ds4_shape` + `config_validate_model`. Definisce le due forme
architetturali supportate e valida i metadati GGUF al caricamento.

| Campo | **Flash** | **Pro** |
|---|---|---|
| `nLayer` | 43 | 61 |
| `nEmbd` | 4096 | 7168 |
| `nVocab` | 129 280 | 129 280 |
| `nHead` | 64 | 128 |
| `nHeadDim` (latent MLA) | 512 | 512 |
| `nRot` (dim RoPE) | 64 | 64 |
| `nOutGroup` | 8 | 16 |
| `nLoraQ` / `nLoraO` | 1024 / 1024 | 1536 / 1024 |
| `nExpert` / `nExpertUsed` | 256 / 6 | 384 / 6 |
| `nExpertShared` | 1 | 1 |
| `nFFExp` (FFN per-expert) | 2048 | 3072 |
| `nHC` (hyper-connection) | 4 | 4 |
| `nHCSinkhornIter` | 20 | 20 |
| `expertWeightScale` | 1.5 | 2.5 |

Punti chiave:

- la forma è **selezionata per match esatto** di tutti i campi-chiave contro i
  metadati GGUF; nessun match → `unsupportedShape`;
- **ratio di compressione per layer** (`expectedCompressRatio`): nel Flash i
  layer 0–1 hanno ratio 0 (attenzione densa), poi alternano **4** (layer pari) e
  **128** (layer dispari). Il valore letto dal GGUF è validato contro questa
  formula;
- vengono validati anche i parametri RoPE/YaRN (`freq_base = 10000`,
  `scale_factor = 16`, `beta_fast = 32`, `beta_slow = 1`, ctx originale 65536),
  la `compress_rope_freq_base = 160000` e il clamp SwiGLU (`= 10`) per ogni
  layer.

`DSV4Shape` (in `DS4Metal`) espone queste costanti al decoder come `DSV4Dims`,
oltre a `ropeParams(layer:)` e `compressRatio(layer:)` per-layer.

---

## 5. Substrato di esecuzione Metal

Tre livelli (corrispondono alle "Stage A/B" annotate nei file):

### `MetalRuntime` — il runtime

**File:** `Sources/DS4Metal/Runtime/MetalRuntime.swift`

- crea device + command queue e **compila la libreria Metal a runtime** dai
  **19 file kernel** `metal/*.metal`, concatenati nell'ordine canonico
  (`flash_attn, dense, moe, dsv4_hc, unary, dsv4_kv, dsv4_rope, dsv4_misc,
  argsort, cpy, concat, get_rows, sum_rows, softmax, repeat, glu, norm, bin,
  set_rows`) preceduti da un *prelude* (macro, `block_q8_0`, kernel di
  self-test). È identico a `ds4_gpu_full_source` in `ds4_metal.m`;
- **default = sorgenti incorporati nel binario** (`KernelSources.swift`,
  generato da `make embed-kernels`): nessuna cartella `metal/` a runtime. Esiste
  un `init(metalDir:)` esplicito usato dai test;
- `pipeline(name)` crea e **memoizza** i `MTLComputePipelineState`;
  `mulMVPipeline(...)` gestisce le pipeline specializzate per function-constant;
- `runTouchSelfTest()` esegue un kernel di gather e verifica il risultato:
  valida end-to-end library → pipeline → dispatch → readback (è il "GPU
  self-test PASSED" della demo).

### `GPUTensor` — l'unità dati

**File:** `Sources/DS4Metal/Runtime/GPUTensor.swift`

Wrappa un `MTLBuffer` (storage *shared*, hazard-tracked) + lunghezza in byte +
conteggio logico di elementi + un **`byteOffset`**. Costruttori:

- `zeros` / `zerosBytes` / `floats` / `bytes` — alloca e carica (scratch e pesi
  copiati);
- **`mappedNoCopy(ptr:byteLength:elementCount:)`** — crea un buffer Metal **senza
  copia** su una regione `mmap` del modello. Poiché il buffer deve iniziare su
  un confine di pagina, parte dalla pagina ≤ `ptr` e il `byteOffset` porta lo
  scostamento intra-pagina del dato reale. **Nessuna copia in RAM**: le pagine
  sono servite dalla page cache del SO (vero streaming SSD, come i
  `g_model_views` del C). Gli encode che leggono questo tensore devono usare
  `setBuffer(offset: byteOffset)`.

### `GraphContext` — il grafo di calcolo

**File:** `Sources/DS4Metal/Decode/GraphContext.swift` (+ estensioni in
`DecodeLayer.swift`, `GraphCompressor.swift`, e i file `Kernels/Metal*.swift`)

Possiede **un `MTLCommandBuffer` + encoder** per un'intera sequenza di
dispatch: *encode molti, commit una volta* (porta di
`ds4_gpu_command_buffer` / `ds4_gpu_finish_command_buffer`). I buffer sono
hazard-tracked, quindi dispatch concatenati che leggono l'output del precedente
**si serializzano automaticamente**.

`begin()` apre il command buffer; gli `encode*` (es. `rmsNorm`, `matmulF32/F16/
Q8_0`, `add`, `swiglu`, `flashAttnCore`, `moeMatvecID`, `hcExpand4`, …)
accodano dispatch; `commit()` chiude, esegue e **attende** (poi gli output sono
leggibili da CPU). Le primitive di matvec scelgono kernel e geometria di
threadgroup in base a dimensioni e quant (es. `kernel_mul_mv_q8_0_f32`,
`kernel_mul_mv_f16_f32_4`), e onorano `weight.byteOffset` per i pesi mmap
no-copy.

I singoli kernel sono wrappati nei file `Sources/DS4Metal/Kernels/Metal*.swift`
(uno per famiglia: norm, RoPE, MoE, flash-attn, softmax, argsort, GLU,
hyper-connection, KV-compress, ecc.) e ciascuno ha un test CPU di riferimento.

---

## 6. Il decoder — forward pass completo

**File:** `Sources/DS4Metal/Decode/DSV4Decoder.swift`

`DSV4Decoder.forward(token:pos:nKeys:)` esegue un forward completo (versione
*all-resident*, tutti i pesi in RAM). Composizione fedele di
`metal_graph_eval_token` + il loop dei layer + `metal_graph_encode_output_head`:

```
1. embed:        embedTokenHC(table, token) → embd, hc[A]   (4 stream HC × nEmbd)
2. layer loop:   for i in 0..<nLayer:
                     decodeLayer(cur, layers[i], …, rawCache[i], nKeys, pos) → other
                     swap(cur, other)        # ping-pong HC tra i due buffer
3. output head:  flat = rmsNorm(cur, hcDim)
                 pre  = matmulF16(hcFn, flat)            # collapse hyper-connection
                 owts = outputHCWeights(pre, scale, base)
                 oembd   = hcWeightedSum(cur, owts)      # 4 stream → 1 embedding
                 onormed = rmsNorm(oembd, out.norm)
                 logits  = matmulQ8_0(out.head, onormed) # proiezione su vocab
4. return logits[vocab]
```

Stato residente: una **KV cache per layer** (`rawCaches[i]`, `maxKeys × headDim`),
i due buffer HC ping-pong, e lo scratch riutilizzabile (`DecodeScratch`).

`generate(prompt:maxNew:)` fa il **prefill** (un `forward` per token del prompt,
popolando le KV cache una posizione alla volta), poi il **decode**
autoregressivo: campiona, si ferma su `eos`, altrimenti reinserisce il token e
avanza `pos`.

### Hyper-Connection (HC)

DeepSeek V4 mantiene **4 "stream" residui** (`nHC = 4`) invece di un singolo
residuo. Il vettore di stato è `[nHC × nEmbd]` e viene fatto **ping-pong** tra
due buffer (`hcA`/`hcB`) layer dopo layer. Ogni blocco (attenzione, FFN):

- **riduce** i 4 stream a 1 embedding (`hcReduce`: rmsNorm → matmul del *mixer*
  F16 → `hcSplitSinkhorn` con 20 iterazioni di Sinkhorn → somma pesata → rmsNorm);
- calcola il blocco su quell'embedding;
- **riespande** il risultato sui 4 stream (`hcExpand4`) con i pesi `post` e
  `comb` ricavati dallo split, sommando il residuo.

La testa di output infine **collassa** i 4 stream in un unico embedding prima
della proiezione sul vocabolario.

---

## 7. Il layer di decode in dettaglio

**File:** `Sources/DS4Metal/Decode/DecodeLayer.swift`

`decodeLayer` è diviso in due fasi (questo split è ciò che abilita
l'expert-cache: si committa dopo il routing, si leggono i 6 id, si raccolgono gli
expert, poi si esegue la fase 2):

### Fase 1 — `decodeRoute` (pre-attenzione → attenzione → pre-FFN → routing)

```
1)  HC-reduce pre-attn:  s.cur = attn_norm(curHc)
1.5) Compressore NSA (solo layer compressi, ratio≠0):
     aggiorna lo stato ricorrente da s.cur; ogni `ratio` token emette
     una riga KV compressa (vedi §8).  → nComp
2)  Q path (MLA):  q_a → rmsNorm(q_a_norm) → q_b → head-norm → RoPE
3)  KV path (MLA): kv → rmsNorm(kv_norm) → RoPE → store fp8 in rawCache[pos]
4)  Attenzione:    flashAttnCore(q, rawCache[0..nKeys] + comp.cache[0..nComp],
                                 sinks) → heads
5)  Proiezione output low-rank:
       RoPE inversa su heads
       attn_low = attnOutLowQ8(output_a, heads)     # grouped low-rank
       blockOut = matmulQ8_0(output_b, attn_low)
       afterAttn = hcExpand4(blockOut, residual=curHc, post, comb)
6)  HC-reduce pre-FFN:  s.cur = ffn_norm(afterAttn)
7)  Router:  logits = matmul(router, s.cur)          # Q8 o F16 per modello
             softplus → sqrt → top-6 → pesi di routing
```

#### Attenzione MLA (Multi-head Latent Attention)

Invece di proiettare K e V separati, il modello usa una **dimensione latente**
(`headDim = 512`) condivisa come K **e** V. Il path Q è **low-rank**: `q_a`
proietta `nEmbd → qRank`, poi `q_b` espande `qRank → qDim = nHead × headDim`,
con una *head-norm* per testa e RoPE solo sui primi `nRot = 64` elementi (il
resto è la parte "no-rope" del latente). La KV è una singola riga latente per
posizione, normata, RoPE-ata e **quantizzata fp8 (E4M3FN)** nella `rawCache`.

L'attenzione è un **flash-attention** con *attention sinks* per-testa (un logit
di sink aggiunto al denominatore della softmax) sul percorso senza padding
(`nKeys % 32 == 0`). Il risultato `heads` subisce una **RoPE inversa** e poi la
**proiezione di output low-rank raggruppata**: `output_a` mappa per gruppi
(`nOutGroup`) verso `nLoraO`, poi `output_b` torna a `nEmbd`.

### Fase 2 — `decodeExperts` (shared FFN + MoE + residuo)

```
shared FFN:  sgate = matmul(shared_gate, s.cur);  sup = matmul(shared_up, s.cur)
             smid  = swiglu(sgate, sup, clamp=10)
             sharedOut = matmul(shared_down, smid)
routed MoE:  gate6 = moeMatvecID(gate experts, ids, s.cur)     # 6 expert
             up6   = moeMatvecID(up experts,   ids, s.cur)
             mid6  = moeSwiGLUWeight(gate6, up6, route_weights, clamp=10)
             down6 = moeMatvecID(down experts, ids, mid6)
             routed = moeSum6(down6)                            # somma pesata dei 6
ffnOut    =  sharedOut + routed
outHc     =  hcExpand4(ffnOut, residual=afterAttn, post, comb)
```

Il modello ha **1 expert condiviso** (sempre attivo, il "shared FFN") più
**256 expert routed** di cui solo **6 selezionati** per token. La SwiGLU usa un
clamp (`limit = 10`) sia nello shared che nel routed.

`DecodeScratch` pre-alloca **una volta** tutti i buffer intermedi (Q, KV, heads,
mask flash, logit del router, i sei gate/up/mid/down, ecc.) e li riusa per ogni
layer e ogni token, così il forward non alloca nulla nel cammino caldo.

---

## 8. Il compressore NSA

**File:** `Sources/DS4Metal/Decode/GraphCompressor.swift`

I layer compressi (`ratio = 4` o `128`) usano un **compressore di attenzione
NSA** ricorrente: invece di far crescere la KV cache indefinitamente, mantengono
uno **stato ricorrente** e ogni `ratio` token emettono **una sola** riga KV
"compressa" (pooled). L'attenzione del layer gira quindi su **righe raw recenti
(finestra SWA) + tutte le righe compresse emesse**.

`CompressorState` (allocato solo per i layer compressi) vive per l'intera
generazione: `stateKv` / `stateScore` (finestra ricorrente, score inizializzato
a `-1e30`), `cache` (le righe emesse), e scratch. `reset()` lo azzera all'inizio
di una nuova sequenza (`pos == 0`).

`runCompressor` per ogni token su un layer compresso:

1. **proietta** `attn_norm` → `kv_cur` / `sc_cur` (matvec F16 con `compKv` /
   `compGate`);
2. **memorizza** nello stato ricorrente alla riga-finestra corrente, aggiungendo
   il bias posizionale **APE** allo score (`kernel_dsv4_compressor_store_one`);
3. se `(pos+1) % ratio == 0` **emette**: `softmax-pool` per-dimensione →
   `rmsNorm(comp_norm)` → RoPE (alla `comp_pos`) → **quantizzazione fp8** →
   scrive in `cache[count]`; per `ratio = 4` esegue anche uno *shift*
   `prev ← cur` dello stato. `count` (numero di righe compresse) avanza.

Il pooling è per-dimensione con stride sullo score; per `ratio = 4` (coff=2)
raccoglie due "corsie" in un buffer packed `8 × headDim` prima del pool, per
`ratio = 128` (coff=1) poola direttamente le `ratio` righe.

---

### L'indexer NSA (DSA, top-512)

Sui layer **ratio-4** un secondo compressore (stessa ricorrenza, pesi
`indexer_compressor_*`, head_dim 128, finalizzato con **Hadamard+FP4** invece
di fp8) mantiene la cache dell'indexer. Quando le righe compresse superano
`indexerTopK` (512 su Flash), per ogni token si calcolano gli score di
rilevanza (`q` da qr_norm via `indexer.attn_q_b` + rope + Hadamard; pesi-testa
da attn_norm via `indexer.proj`; kernel `indexer_score_one_direct`), si fa il
**top-K su CPU** (split del command buffer al confine degli score) e si scrive
una **maschera f16 0/-inf** sulle righe compresse che il flash-attention
consuma — il "dense top-k mask path" del C (`indexer_allowed_decode_one`).
Sotto i 2048 token (≤512 righe) il percorso resta quello denso, identico.
Lo stato dell'indexer fa parte dello snapshot KV su disco. Il decoder legacy
non-streaming (`DSV4Decoder`, demo) resta denso.

## 9. Il MoE: router ed expert

Il **router** (fase 1, passo 7) produce `nExpert` logit con un matvec (Q8_0 nel
modello Q4_K, **F16** nel modello IQ2_XXS — selezionato da `routerF16`), poi
`softplus → sqrt`, e infine `routerFinalizeTop6` seleziona i **6 expert** col
punteggio più alto e `routerWeights` ne calcola i pesi di combinazione.

Gli **expert** sono memorizzati come tre grandi tensori per layer
(`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`), ciascuno con tutte le righe
dei 256 expert. `moeMatvecID` (kernel `mul_mv_id`) esegue il matvec **solo per
gli id selezionati**. Questo è il fulcro dell'**expert-cache**: dopo il routing
si conoscono i 6 id, quindi basta avere in RAM/GPU solo le righe di quei 6
expert (≈ 6/256 ≈ 2,3% dell'I/O degli expert) — vedi §10.

Il quant di gate/up/down è **per-tensore** e configurato da `DSV4Dims`
(`gateQuant`, `upQuant`, `downQuant`), così lo stesso codice serve sia il
modello Q4_K sia quello a 2 bit (`IQ2_XXS` gate/up + `Q2_K` down).

### Kernel MoE fusi (decode)

Il percorso routed del decode usa, quando lo schema quant lo permette, i **kernel
fusi del percorso release del motore C** — 2 dispatch invece di 5:

- **`kernel_mul_mv_id_<q>_pair_swiglu_f32`** (`iq2_xxs`/`q4_K`): gate+up matvec +
  SwiGLU·peso di routing in un solo dispatch (`mid` scritto direttamente, niente
  re-lettura degli intermedi gate/up);
- **`kernel_mul_mv_id_<q>_sum6_f32`** (`q2_K`/`q4_K`): down-projection + somma dei
  **6** expert in un solo dispatch (scrive direttamente la riga routed). Il kernel
  ha i 6 slot cablati, quindi è usato solo a k pieno (`activeExperts == 6`).

Combinazioni senza kernel fuso (o `DS4_FUSED_MOE=0` per il confronto A/B) usano il
percorso validato a 5 dispatch. L'equivalenza numerica dei due percorsi è coperta
da `MetalMoEFusedDecodeTests`.

### Esperti attivi configurabili (`DSV4Dims.activeExperts`, env `DS4_ACTIVE_EXPERTS`)

Il router sceglie sempre i **top-6 di 256**. Se `activeExperts < 6`, il path di
streaming tiene in Swift i **top-K** di quei 6 (per peso di routing, **renormalizzati**
allo stesso totale), fa il **gather di solo K** esperti dall'mmap e calcola con
k=K (azzerando le righe `down6` inutilizzate, così la `moeSum6` cablata somma
zeri). **Nessuna modifica ai kernel.** Meno esperti = **meno I/O di esperti per
token** (la fase dominante su macchine con poca RAM), a scapito della qualità (il
modello è addestrato per 6). Si imposta con `DS4_ACTIVE_EXPERTS=2..6` (CLI o
scheme Xcode); onorato sia da `DS4Demo` sia da `InferenceService`.

---

## 10. StreamingDecoder e strategie di streaming

**File:** `Sources/DS4Metal/Decode/StreamingDecoder.swift`

Il `DSV4Decoder` tiene **tutti** i pesi residenti (≈164 GB): impossibile sotto i
64 GB. Lo `StreamingDecoder` risolve il problema caricando/evictando i pesi
**per-layer**, con due meccanismi combinati.

### Prefill layer-major (`prefill(tokens:chunk:)`)

Il decode (`forward`) è **token-major**: per ogni token ricarica i pesi di tutti
i layer. In streaming, dove il costo dominante è l'**I/O dei pesi** (vedi il
profilo per-fase: gather esperti + fault non-routed ≈ 85–90% del tempo), il
prefill del prompt è invece **layer-major**: per ogni layer i pesi si caricano
**una volta** e si applicano a **tutti** i token del chunk → l'I/O dei pesi si
ammortizza sull'intero prompt. È **numericamente identico** a chiamare `forward`
per i token 0..N-1 in ordine (stesse op, stesso ordine per token, stessa
evoluzione di KV cache e compressore NSA): solo il riordino dei loop (layer fuori,
token dentro) tiene i pesi mmap *caldi* tra i token. Il prompt è diviso in
**chunk** (default 512 token) per limitare la memoria delle attivazioni (≈ 2·chunk
buffer HC); la KV cache e lo stato del compressore proseguono tra i chunk. La
condivisione del corpo del layer tra decode e prefill è in `runLayer`; embed e
testa di output in `embedToken`/`outputHead`. Restituisce i logit dell'**ultimo**
token per avviare la generazione.

**Expert I/O batched (`batchedExpertLayer`).** Sul percorso expert-gather il
prefill deduplica anche l'I/O degli **expert**: dentro ogni layer, la fase A
esegue le **route in sequenza** per token (l'attenzione è causale: il token j
legge le KV scritte dai token 0..j nello stesso layer) salvando per token gli
input della FFN (cur normato, residuo, split HC) e la selezione del router; la
fase B raggruppa i token e fa il **gather dell'unione** degli expert selezionati
del gruppo **una volta sola**, eseguendo poi la FFN di ogni token con gli id
rimappati sull'unione. Numericamente identico (la FFN di un token non alimenta
gli altri token nello stesso layer); l'I/O scende da `6·token` a
`min(6·token, 256)` letture-expert per layer — su prompt lunghi fino a **~12×**
in meno nella fase dominante. L'unione per gruppo è limitata da
`DS4_PREFILL_UNION` (default 64 expert ≈ 450 MB transitori sul modello 2-bit).

### Cache expert a slot — "persistenti + che cambiano" (`ExpertSlotCache`)

Nel **decode**, opzionalmente (`DS4_EXPERT_CACHE_SLOTS=N`, default off), ogni layer
ha un **pool LRU di N slot** in buffer GPU condivisi (gate/up/down packed per
slot). Gli expert *caldi* restano **residenti** nel pool tra i token (hit = zero
copie); un *miss* evicta lo slot meno usato e copia **solo quell'expert**
dall'mmap. Il matvec gira sul pool usando gli **indici di slot come id**, quindi
funzionano sia i kernel validati sia i **fusi** — nessun kernel nuovo. Statistiche
hit/miss nel profilo decode (`cache expert X hit / Y miss`).

⚠️ Il pool è memoria **wired** (≈6,9 MB/slot sul modello 2-bit × 43 layer: N=8 ≈
2,4 GB) e compete con la page cache: su macchine con poca RAM parti basso (4–8)
e guarda l'hit-rate; con routing molto uniforme (load balancing) la cache può non
ripagare. Sul prefill resta la dedup a unione (più adatta al batch).

**"Imatrix d'uso" (`ExpertUsageStats`).** Il router alimenta statistiche di
frequenza per (layer, expert) — l'equivalente runtime di una importance matrix
per gli expert (nota: la *vera* imatrix di quantizzazione non è nel GGUF e serve
solo offline). Il servizio le **persiste** in Application Support
(`expert-usage-<modello>.json`, salvataggio automatico a fine generazione) e al
caricamento successivo **pre-scalda** il pool con gli expert storicamente più
caldi (entrano come voci "più vecchie": un prior sbagliato viene evictato in
fretta). `concentration(layer:n:)` misura quanto il routing è concentrato
(~n/256 = uniforme → cache inutile): è il segnale onesto esposto nella tab
**Tuning** della GUI, insieme a hit/miss live e alla configurazione slot.

**Agenti (ruoli) e profili per-agente (`AgentProfile`).** Un agente = system
prompt (ruolo) + tool esposti + **profilo d'uso expert separato**
(`expert-usage-<modello>-<agente>.json`): ruoli diversi instradano verso expert
diversi, quindi il prior di warm è molto più mirato di un profilo globale.
`InferenceService.setAgent` persiste il profilo dell'agente uscente, carica
quello del nuovo, **invalida i pool** della cache (`ExpertSlotCache.invalidate`,
re-warm lazy col nuovo profilo) e apre una conversazione fresca col ruolo. Nella
GUI: picker agente nell'header della chat (Generale / Coding / Matematica — con
i tool aritmetici — / Scrittura); il system prompt utente si concatena a quello
del ruolo.

### Pattern "split command buffer"

Ogni layer gira nel **proprio command buffer** (`commit + wait`, poi evict). I
buffer GPU di un layer devono restare vivi finché la GPU non ha finito di
leggerli: committando e attendendo prima di rilasciare la `LayerWeights`, i
buffer Metal del layer vengono liberati (eviction). Così il *working-set* è
**un layer alla volta** (pochi GB) invece dell'intero modello — il modello reale
gira in 16 GB. Embedding, testa di output, KV cache, buffer HC e scratch restano
residenti (sono piccoli).

### Variante con expert-cache

Quando è impostato `expertGather`, ogni layer è **spezzato al router**:

1. `decodeRoute` in un proprio cb → si legge `s.selected` (i 6 id);
2. `gather(layer, ids)` raccoglie **solo** quei 6 expert dalla `mmap`;
3. `decodeExperts` in un proprio cb, sui 6 expert packed.

### Costruttori (dal più al meno aggressivo)

| Factory | Pesi non-routed | Expert | Note |
|---|---|---|---|
| `fromGGUF` | copiati, evictati per-layer | tutti, per-layer | il più semplice; ricarica ogni layer ad ogni token (lento). |
| `fromGGUFExpertCached` | copiati una volta, **memoizzati** | **6 selezionati** gather/token | modello C `--ssd-streaming`. |
| `fromGGUFExpertCachedMapped` | **viste no-copy mmap** (page cache) | **6 selezionati** gather/token | **il percorso 16 GB più veloce**: nessuna copia, nessun buffer "dirty" che causa OOM. È quello usato da `InferenceService` e `DS4Demo`. |
| `fromGGUFMappedExperts` | copiati e memoizzati | viste no-copy su **tutti** i 256 | decode a cb singolo con id reali; la page cache trattiene gli expert toccati. |

`CachedLayerProvider` memoizza i pesi di un layer alla prima richiesta e li
riusa tra i token (i pesi sono read-only durante il decode).

> Il `fromGGUFExpertCached*` è il "modello `--ssd-streaming`" del motore C:
> i pesi non-routed sono residenti (copia singola o page cache), e per token si
> paga solo qualche *slab* di expert dall'SSD + il compute GPU, invece di
> ri-streamare l'intero modello.

Inoltre lo `StreamingDecoder` alloca **uno stato compressore per layer
compresso** (`CompressorState`) e dimensiona lo scratch dell'attenzione per
ospitare `maxKeys` righe raw + fino a `maxKeys/4` righe compresse (il ratio-4 è
il più denso). All'inizio di ogni sequenza (`pos == 0`) resetta tutti gli stati
compressore.

### Knob di streaming opt-in (default OFF)

- **Raw-KV ring** (`DS4_RAW_RING`, anche via toggle in Impostazioni): la raw cache
  `rawCaches[i]` passa da `maxKeys` righe a un **ring di `nSWA`** (l'attenzione vede
  solo l'ultima finestra SWA) → RAM della raw cache **costante** (da O(contesto) a
  ~11 MB). Implementato in modo trasparente: il modulo è implicito in
  `rawCache.count/headDim`, la scrittura va in `pos % nSWA`, lo staging
  dell'attenzione **de-ruota** il ring in ordine cronologico (kernel flash
  invariato), ed `exportKV`/`importKV` de-ruotano/ri-ruotano; con la cache piena
  (default) non c'è mai wrap → identico bit-a-bit. **Riallinea il port
  all'upstream**: `ds4.c`/`ds4_metal.m` usano già una finestra scorrevole
  (`raw_cap = n_swa`, scrittura `% raw_cap`, `raw_start` nell'attenzione). Le
  **righe compresse** restano ∝ contesto (candidate a un offload su disco).
- **Prefetch** (`DS4_PREFETCH`, +`DS4_PREFETCH_EXPERTS`): durante il calcolo del
  layer `i` si fa `madvise(WILLNEED)` sulle pagine mmap del layer `i+1` (pesi
  non-routed + esperti probabili dalla usage imatrix), su una coda di background →
  l'I/O da SSD si sovrappone al compute. Hint-only su mapping read-only: non può
  alterare i numeri.

---

## 11. Decoder — dai logit al token (`Sampler`)

**File:** `Sources/DS4Core/Inference/Sampler.swift`

Port fedele di `ds4_sample_logits`. Usa `expf` di libm così i risultati
combaciano col C bit-per-bit sulla stessa piattaforma. RNG = **xorshift64\***
(`rngNext`/`rngF32`, reseed non-zero come il C).

`sample(logits, temperature, topK, topP, minP, rng)`:

- `temperature <= 0` → **argmax** (greedy, deterministico — è ciò che usa la
  demo e il prefill);
- `topK <= 0` → **full-vocab** (`fullVocab`): softmax con temperatura su tutti i
  logit finiti, con filtri `top-p` e `min-p`; se `topP >= 1` evita
  l'ordinamento (campionamento diretto cumulativo);
- `topK > 0` → **insertion sort parziale** dei `topK` logit più grandi, poi
  softmax + `top-p`/`min-p` + estrazione cumulativa.

I default di `SamplingParams` nel `InferenceService` GUI sono
`temperature = 0.6, topP = 0.95, minP = 0.05`; la demo usa `temperature = 0`
(greedy). Il loop di decode (sia in `InferenceService` sia in `DSV4Decoder.
generate`) si ferma su `eosId` e, in modalità Thinking, separa il flusso prima/
dopo `thinkEndId` in `reasoning` vs `text`.

---

## 12. Quantizzazione

I pesi usano formati a blocchi in stile llama.cpp/GGUF. I principali:

- **Q8_0** — blocco di 32 valori int8 con uno scale F16 (`block_q8_0` nel
  prelude Metal: `half d; int8_t qs[32]` = 34 byte). Usato per la maggior parte
  dei pesi densi (q_a/q_b/kv/attn_out/shared/router nel modello Q4_K, testa di
  output).
- **Q4_K** — quant a 4 bit a super-blocchi: gli expert del modello "Flash Q4_K".
- **IQ2_XXS / Q2_K** — quant a 2 bit: il modello da ~2 bit usa `IQ2_XXS` per
  gate/up degli expert, `Q2_K` per down, e **router F16**. È il modello
  consigliato per macchine con poca RAM.
- **F16 / F32** — norm, mixer hyper-connection (`hc_*_fn` sono F16), scale/base,
  proiezioni del compressore (`compKv`/`compGate` F16).
- **fp8 (E4M3FN)** — la KV cache latente e le righe compresse NSA (la parte
  no-rope è quantizzata fp8 in cache).

`GGUFWeights.detectMoEQuant` legge i tipi reali dal GGUF e imposta
`DSV4Dims.{gateQuant,upQuant,downQuant,routerF16}`, così il decoder dispatcha i
kernel MoE corretti. **Un disallineamento tra il tipo reale e quello assunto
produce output spazzatura** — per questo `DS4Demo` offre l'audit `DS4_TYPES_ONLY=1`
che stampa i dtype effettivi dei tensori per-layer.

---

## 13. Riepilogo dei tensori per-layer

`LayerWeights` (in `DecodeLayer.swift`) raccoglie tutti i pesi di un layer:

| Gruppo | Tensori GGUF (`blk.<il>.…`) | Tipo | Ruolo |
|---|---|---|---|
| Hyper-conn. attn | `hc_attn_fn` / `hc_attn_scale` / `hc_attn_base` | F16/F32 | mixer + scale/base per la riduzione HC pre-attn |
| Norm attn | `attn_norm` | F32 | RMSNorm pre-attenzione |
| Q low-rank | `attn_q_a` / `attn_q_a_norm` / `attn_q_b` | Q8/F32 | proiezione query MLA |
| KV latente | `attn_kv` / `attn_kv_a_norm` | Q8/F32 | proiezione + norm KV latente |
| Sinks | `attn_sinks` | F32 | logit di sink per-testa (softmax) |
| Output attn | `attn_output_a` / `attn_output_b` | Q8 | proiezione output low-rank raggruppata |
| Hyper-conn. ffn | `hc_ffn_fn` / `hc_ffn_scale` / `hc_ffn_base` | F16/F32 | riduzione HC pre-FFN |
| Norm ffn | `ffn_norm` | F32 | RMSNorm pre-FFN |
| Shared FFN | `ffn_gate_shexp` / `ffn_up_shexp` / `ffn_down_shexp` | Q8 | expert condiviso (sempre attivo) |
| Router | `ffn_gate_inp` | Q8 o F16 | logit di selezione dei 256 expert |
| Routed experts | `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` | Q4_K o IQ2/Q2_K | i 256 expert (6 usati/token) |
| Compressore NSA | `attn_compressor_kv` / `_gate` / `_ape` / `_norm` | F16/F32 | solo layer compressi (ratio≠0) |

La testa di output (`OutputHeadWeights`): `token_embd` (embedding, F16),
`output_hc_fn`/`output_hc_scale`/`output_hc_base` (collasso HC),
`output_norm` (F32), `output` (proiezione vocab, Q8).

---

## 14. Tool calling (function calling)

**File:** `Sources/DS4Core/Inference/ChatTools.swift` (rendering/parse, puro) ·
`Sources/DS4Engine/Tools.swift` (registry + tool demo) ·
`Sources/DS4Engine/InferenceService.swift` (stato multi-turno + loop).

### Riconoscimento dei token speciali

Per tokenizzare correttamente il markup dei tool, in `tokenizeRenderedChat` il
`Tokenizer` riconosce **tutti i token di tipo CONTROL**
(`tokenizer.ggml.token_type == 3`) come atomici, in match *longest-first*, e vi
**unisce sempre i 7 special nominati** (BOS/EOS, ruoli, think, `｜DSML｜` —
dedup per id). L'unione non è un fallback: alcuni GGUF taggano `｜DSML｜` come
USER_DEFINED (tipo 4) invece di CONTROL, e senza unione l'esempio
`<｜DSML｜tool_calls>` nel prompt verrebbe **spezzato dal BPE in `DS`+`ML`** —
il modello, vedendo l'esempio rotto, impara a emettere quei frammenti di testo
invece del token di controllo (bug osservato e corretto). Il pannello
Diagnostica mostra per ogni special il tipo e se tokenizza atomico.
`tokenizer.tokenId(_:)` risolve l'id di un token arbitrario per stringa.

### Formato DSML (allineato al `tokenizer.chat_template` reale)

`ToolSpec`, `ToolCall`, `ChatTurn` sono i tipi base. Il `ChatRenderer` **rispecchia
il template Jinja del modello** (verificato sul GGUF). Punti salienti:

- **Dichiarazione** dei tool in un blocco system `## Tools` (testo verbatim del
  template) + schemi delle funzioni in JSON (chiavi ordinate, ≈ `tojson`);
- **Chiamata** in XML sul token `｜DSML｜`:
  ```
  <｜DSML｜tool_calls>
  <｜DSML｜invoke name="get_weather">
  <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
  <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
  </｜DSML｜invoke>
  </｜DSML｜tool_calls>
  ```
  stringhe → `string="true"` valore grezzo; altri tipi → `string="false"` valore JSON;
- **Risultato** tool dentro un turno utente: `<｜User｜><tool_result>…</tool_result>`
  (risultati consecutivi non ripetono `<｜User｜>`);
- ogni turno assistant apre `<｜Assistant｜>` poi `</think>` (o `<think>…` per un
  turno con ragionamento) e chiude con `<｜end▁of▁sentence｜>`;
- niente newline spuria dopo il `BOS` (system attaccato al BOS).

> Allineato 1:1 al template del modello (no Jinja a runtime). Se il modello cambia
> template, `ChatRenderer` è l'unico punto da adeguare.

### Rendering e parsing

- **`ChatRenderer.render(turns:tools:think:markup:)`** produce la stringa di chat:
  `BOS` + system (+ blocco dichiarazione "## Tools") + turni (user/assistant con
  le tool-call in DSML, e i risultati tool) + turno assistant aperto
  (`<think>`/`</think>`). I parametri si convertono da `argumentsJSON` ai tag
  DSML (e viceversa nel parse).
- **`ToolCallParser.parse(_:markup:)`** estrae le `ToolCall` dal blocco DSML
  dell'output (gestisce `invoke`/`parameter`, ricostruisce `argumentsJSON`),
  restituendo anche il testo visibile ripulito.

### Loop nell'`InferenceService`

Il servizio mantiene `turns` (conversazione) e `tools`. Ogni generazione:

1. **render** dell'intera conversazione + tool → tokenizza → **prefill da pos 0**
   (nessun riuso KV tra turni);
2. **decode** con rilevamento del blocco tool: quando il token campionato è
   **`｜DSML｜`** (`tok.dsmlId`, token singolo), smette di emettere `.text` e
   bufferizza i byte del blocco; a fine generazione ricompone testo+blocco, fa il
   parse ed emette **`GenEvent.toolCall([ToolCall])`**;
3. registra il turno assistant (testo visibile + tool-call) nello storico.

`provideToolResults(_:)` accoda i `toolResult` e rilancia la generazione: il
modello vede gli output e produce la risposta finale o altre chiamate.

### Registry e tool

`ToolRegistry` (DS4Engine, `Tools/`) raccoglie i built-in auto-eseguibili —
**uno per file** in `Tools/Builtins/` (ognuno è una `extension ToolRegistry {
static let X }`); il core `ToolRegistry.swift` tiene la superficie del registry
(`builtins`, `projectScoped`, `subAgentGrantable`, `execute`, `specs`) e gli
helper condivisi (parsing argomenti, `binaryTool`, valutatore aritmetico).
`execute(_:)` restituisce un `ToolOutput` oppure `nil` per i tool non integrati
(risultato manuale dalla UI). Gruppi: aritmetica (`now`, `calculator`,
`add/subtract/multiply`), progetto (`project_list/read/search/write/edit`), file
grezzi (`file_read/lines/write/add/modify`, anche per intervallo di righe), `git`,
e sub-agenti (`agents_list`, `subagent_search`, `subagent_run`). `projectScoped`
marca i tool che richiedono un progetto; `subAgentGrantable` quelli concedibili a
un sub-agent (esclusi i `subagent_*` e `agents_list` → niente sub-agent annidati).

### Sub-agenti — context-switch sul decoder

`subagent_run` è **gestito dall'engine**: `ChatStore` lo intercetta e chiama
`InferenceService.runSubAgent(target:question:agent:tools:)` invece di
`ToolRegistry.execute`. Il runner:

1. fa lo **snapshot** del KV main (`decoder.exportKV` — finestra raw + stato del
   compressore, economico) e salva `committedIds`;
2. costruisce/ripristina la **KV cache content-key** del target (file o `"project"`)
   da un `DiskKVStore` dedicato (`subagent-kv/`): prima volta `prefill` + `store`,
   poi `snapshot(forTokens:)`;
3. esegue il loop tool del sub-agent (`decodeSubTurn`, eseguendo i suoi tool via
   `ToolRegistry.execute`), bounded dal contesto;
4. **ripristina** il main (`decoder.importKV(snapshot)` + `committedIds`).

Al main torna solo la risposta (come tool-result), quindi committa `[chiamata
subagent_run] + [risposta]`, non l'elaborazione interna. Il context-switch è
corretto perché `importKV` ristabilisce uno stato **risumibile completo**
(finestra SWA + compressore ricorrente), esattamente come per il disk-KV.

---

## 15. Inferenza distribuita

**File:** `Sources/DS4Engine/Distributed/` (`DistProtocol`, `DistTransport`,
`DistEngine`, `DistWorker`, `DistCoordinator`) ·
`Sources/DS4Metal/Decode/StreamingDecoder.swift` (API slice).
Modellata su `ds4_distributed.c` (pipeline parallelism), con protocollo
Swift-nativo (tutti i nodi eseguono DwarfStar: nessuna byte-compat col C).

### API slice del decoder

Tra un layer e il successivo scorre lo **stato HC** (`nHC×nEmbd` float): è
l'unico dato che deve attraversare la rete. Lo `StreamingDecoder` espone tre
operazioni additive che riusano `embedToken`/`runLayer`/`outputHead` — uno
slice `[a,b]` è numericamente identico agli stessi layer dentro `forward()`:

```swift
embed(token:pos:) -> [Float]                       // coordinatore: inizio pipeline
forwardSlice(hc:pos:nKeys:start:end:) -> [Float]   // worker: i suoi layer
forwardSliceBatch(hcs:posBase:start:end:)          // prefill a chunk (token-outer)
head(hc:) -> [Float]                               // nodo terminale: logits
```

Con `kvLayers:` il decoder alloca KV-cache e stato compressore **solo per il
range** (dummy altrove): il worker non paga la cache dei layer altrui, il
coordinatore puro (`0..<0`) non alloca nulla. I pesi sono mmap **lazy**: un
worker che esegue solo il suo slice non fa mai page-fault sugli altri layer —
da qui il risparmio di RAM per nodo (e meno thrashing della page cache: ogni
nodo stream-a ~1/N degli esperti).

### Protocollo e topologia

Frame TCP `magic DS4D + type + length` (`DistConnection`, Network.framework,
wrapper async). Messaggi:

- **HELLO** (worker → coordinatore alla connessione): nome modello, slice
  `[start,end]`, `hasOutput`, nLayers, contesto;
- **WORK**: `nTokens` stati HC concatenati + posizione base + range layer +
  flag (`resetSession` a pos 0, `outputLogits` per l'hop terminale) + bit di
  trasporto + (per l'inoltro) route completa e indirizzo di ritorno;
- **RESULT**: stati HC (relay), logits (terminale) o ack (flow control inoltro).

Le attivazioni viaggiano a **32/16/8 bit** (`ActivationCodec`: float16, o int8
con scala absmax per-vettore).

Il **coordinatore** valida che la route copra tutti i layer in modo contiguo,
fa il **prefill a chunk** (default 32 token/frame: il round-trip di rete si
ammortizza sul chunk) e il decode token-per-token. Due trasporti:

- **relay** (default): coordinatore → worker → coordinatore per ogni hop;
  nessuna connettività in ingresso richiesta sul coordinatore;
- **inoltro worker→worker**: il WORK trasporta la route; ogni worker inoltra lo
  stato HC al prossimo hop e il terminale risponde al `DistReturnListener` del
  coordinatore (serve il suo IP LAN) — metà degli hop per token.

---

## Riferimenti incrociati C → Swift

Ogni file Swift cita la funzione C di origine. I principali punti d'aggancio:

| Swift | C (`ds4.c` / `ds4_metal.m`) |
|---|---|
| `GGUFModel` | `model_open`, `parse_metadata`, `parse_tensors` |
| `Tokenizer` | `bpe_tokenize_text`, `encode_chat_prompt`, `ds4_token_text` |
| `ModelConfig` | `config_validate_model`, `ds4_shape` |
| `DSV4Decoder.forward` | `metal_graph_eval_token` + `metal_graph_encode_output_head` |
| `decodeLayer` | `metal_graph_encode_decode_layer` |
| `runCompressor` | `compressor_decode_one`, `ds4_gpu_compressor_update_tensor` |
| `StreamingDecoder` | il modello `--ssd-streaming` (split command buffer) |
| `Sampler.sample` | `ds4_sample_logits` → `sample_top_p_min_p` → `sample_full_vocab` |
| `MetalRuntime` | `ds4_gpu_full_source` (compilazione kernel a runtime) |
| `DistWorker` / `DistCoordinator` | `ds4_dist_run`, `ds4_dist_session_eval` (`ds4_distributed.c`) |
| `LocalServer` (DwarfStar) | `ds4_server.c` (route + wire format OpenAI/Anthropic/Responses) |

La validazione numerica end-to-end richiede il modello reale (≥64 GB); i singoli
kernel e sotto-operazioni sono validati contro CPU/`./ds4` nella suite
`Tests/DS4CoreTests/`.
