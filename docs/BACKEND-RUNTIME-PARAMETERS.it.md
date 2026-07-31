# Parametri runtime: DeepSeek V4, GLM 5.2 e Laguna S 2.1

Questo documento inventaria le variabili d'ambiente che influenzano i tre
percorsi di inferenza. Sono escluse le variabili relative a download, server,
tool, distribuzione e UI che non modificano il runtime del modello.

## Regola di risoluzione

I nomi comuni `DS4_*` sono l'interfaccia pubblica e backend-agnostica. I nomi
storici specifici restano supportati solo come fallback deprecato:

1. nome comune, per esempio `DS4_PREFILL_BATCH`;
2. alias storico, per esempio `DS4_GLM_PREFILL_BATCH`;
3. default del backend.

GUI, demo, diagnostica e auto-tune scrivono esclusivamente i nomi comuni.
I default possono restare diversi quando sono il risultato di A/B differenti:
un nome comune uniforma il controllo, non forza una politica identica su
architetture diverse.

## Schema condiviso

La fonte di verità compilabile è `DS4RuntimeKnob`: ogni controllo condiviso
dichiara nome canonico, categoria, tipo del valore e soli alias legacy
applicabili a ciascun backend. `DS4RuntimeEnvironment` risolve questo schema
con precedenza canonica e mantiene gli alias isolati, quindi per esempio un
vecchio `DS4_GLM_MTLIO` non può configurare Laguna.

Regola per i backend futuri:

- riusare un caso `DS4RuntimeKnob` quando la semantica operativa coincide;
- aggiungere un nuovo `DS4_*` allo schema quando il concetto è riutilizzabile;
- usare `DS4_<BACKEND>_*` soltanto per gate, formati o algoritmi esclusivi;
- non aggiungere un alias backend-specifico per un nuovo backend.

## Parametri comuni della demo

Questi controlli sono condivisi dal comando `DS4Demo`; tokenizer e template
restano naturalmente specifici del modello.

| Parametro | Default | Funzione |
|---|---:|---|
| `DS4_PROMPT` | prompt incorporato | Testo utente. |
| `DS4_PROMPT_FILE` | non impostato | Prompt UTF-8 da file. |
| `DS4_PROMPT_MAX_CHARS` | `12000` | Troncamento del prompt da file. |
| `DS4_SYSTEM` | template del backend | System prompt. |
| `DS4_RAW_PROMPT` | off | Evita il template chat dove supportato. |
| `DS4_MAX_NEW` | `128` | Token massimi generati. |
| `DS4_NOTHINK` | off | `=1` disabilita il reasoning esplicito. |
| `DS4_DEMO_CONTEXT` | `4096` | Capacità KV configurata. |
| `DS4_DEMO_LIVE_CONTEXT` | non impostato | Contesto sintetico di N token per benchmark. |
| `DS4_DEMO_TEMPERATURE` | backend/dimostrazione | Temperatura. |
| `DS4_DEMO_TOP_K` | backend/dimostrazione | Top-K del sampler. |
| `DS4_DEMO_TOP_P` | backend/dimostrazione | Top-P del sampler. |
| `DS4_DEMO_MIN_P` | backend/dimostrazione | Min-P del sampler. |
| `DS4_DEMO_REPEAT_PENALTY` | `1` | Penalità di ripetizione. |
| `DS4_DEMO_REPEAT_LAST_N` | `64` | Finestra della penalità. |
| `DS4_DEMO_SEED` | backend | Seed del sampler. |
| `DS4_WARMUP` | `0` | Token iniziali esclusi dal profilo. |
| `DS4_DIAG` | off | Diagnostica I/O/cache/routing. |
| `DS4_TYPES_ONLY` | off | Audit GGUF e tokenizer, poi uscita. |
| `DS4_DUMP_TOKENS` | off | Stampa la tokenizzazione. |
| `DS4_AB_TRACE` | non impostato | Traccia logit per A/B. |
| `DS4_AB_TRACE_FRAMES` | `9` | Frame massimi della traccia. |

## Confronto semantico

`—` significa che il backend non espone quel concetto o lo determina
automaticamente.

| Concetto comune | Nome comune | DeepSeek V4 | GLM 5.2 | Laguna S 2.1 |
|---|---|---|---|---|
| Esperti attivi | `DS4_ACTIVE_EXPERTS` | `1...6`, default `6` | `1...8`, default `8` | `1...10`, default `10` |
| Cache esperti in slot/layer | `DS4_EXPERT_CACHE_SLOTS` | default `0`; pool per layer | default `16` | pool globale, altrimenti derivato dai MiB |
| Cache esperti in MiB | `DS4_EXPERT_CACHE_MB` | — | — | default `2048` |
| Layer residenti | `DS4_RESIDENT_LAYERS` | dense residency separata | adattivo | prefisso di layer routed residenti |
| Slot dello streamer | `DS4_STREAM_SLOTS` | ring denso controllato da `DS4_DENSE_AHEAD` | default `3` | staging/cache derivati dal budget |
| Streaming SSD | `DS4_SSD_STREAM` | implicito nel gather | implicito per i layer non residenti | on nel preset GUI |
| Prefill batch principale | `DS4_PREFILL_BATCH` | on | off | on |
| Proiezioni dense multi-token | `DS4_PREFILL_DENSE_MM` | on | — | on |
| MoE multi-token nel prefill | `DS4_PREFILL_MOE_BATCH` | on | on | off sperimentale |
| Route batch nel prefill | `DS4_PREFILL_ROUTE_BATCH` | default `32` | default `16` | incorporato nel chunk layer-major |
| Chunk prefill | `DS4_PREFILL_CHUNK` | default `512` | intero prompt layer-major | default/max richiesto `256`, ridotto automaticamente dalla cache |
| Wiring RAM | `DS4_MLOCK` | off | on | off |
| MetalIO | `DS4_MTLIO` | off | on | off, fallback automatico a pread |
| Split delle `pread` | `DS4_PREAD_SPLIT` | default `1` | default `4` | default `1`, `1...8` |
| Directory bundle | `DS4_BUNDLE_DIR` | `.expbundle` | `.glm-experts` | — |
| Storico routing | `DS4_USAGE_FILE` | `.usage.json` | `.glm-usage.json` | — |
| Decode speculativo | `DS4_SPEC_K` | self-speculative | prompt lookup | — |
| SIMD group generico | `DS4_NSG` | usa controlli distinti Q8/MoE | default `4` | default `4`, `1...8` |
| Attention lunga indicizzata | `DS4_INDEXED_ATTN` | on, indexer appreso DSA | — | on oltre 4096 token, indice a centroidi |
| Overlap shared expert/I/O | `DS4_SHARED_EXPERT_OVERLAP` | incorporato in `DS4_ASYNC_ROUTE` | — | off nel preset M1 Pro |

Lo stesso nome non promette lo stesso default o la stessa implementazione
interna. Promette la stessa intenzione operativa: per esempio
`DS4_PREFILL_BATCH=0` seleziona il riferimento non batchato di ogni backend.

## Inventario DeepSeek V4

### Prefill

| Parametro | Default | Funzione |
|---|---:|---|
| `DS4_PREFILL_CHUNK` | `512` | Token per chunk. |
| `DS4_PREFILL_UNION` | `192` | Esperti massimi per unione I/O. |
| `DS4_PREFILL_ROUTE_BATCH` | `32` | Token per commit della fase route. |
| `DS4_PREFILL_BATCH` | on | Attention multi-query nel route batch. |
| `DS4_PREFILL_MOE_BATCH` | on | Un command buffer FFN per gruppo. |
| `DS4_PREFILL_MM` | off | FFN matmul multi-token, non bit-identica. |
| `DS4_PREFILL_DENSE_MM` | on | Proiezioni dense matmul nel batch. |
| `DS4_PREFILL_FULL_LAYER` | on | Stream dell'intero layer routed sui prompt lunghi. |
| `DS4_PREFILL_FULL_LAYER_MIN` | `512` | Soglia del full-layer. |
| `DS4_PREFILL_RESIDENT_IDS` | on | Routing consumato dai buffer GPU. |
| `DS4_PREFILL_MICRO_BATCH` | on | Micro-batch interno. |
| `DS4_PROFILE_PREFILL` | off | Profilo sincrono delle sottofasi. |

### Cache, esperti e I/O

| Parametro | Default | Funzione |
|---|---:|---|
| `DS4_ACTIVE_EXPERTS` | `6` | Esperti routed realmente eseguiti. |
| `DS4_EXPERT_CACHE_SLOTS` | `0` | Slot LRU per layer. |
| `DS4_EXPERT_CACHE_UNIFORM` | off | Distribuzione uniforme invece che usage-aware. |
| `DS4_EXPERT_CACHE_NO_CLAMP` | off | Disabilita il clamp di sicurezza del pool. |
| `DS4_MULTI_QUANT_CACHE` | off | Pool corretti per tipi quant misti. |
| `DS4_EXPERT_BUNDLE` | off | Sidecar esperti contigui. |
| `DS4_BUNDLE_DIR` | sibling GGUF | Directory del bundle. |
| `DS4_POOL_INTERLEAVE` | on | Record gate/up/down contiguo nello slot. |
| `DS4_EXPERT_PREAD` | off | `pread/F_NOCACHE` per gli esperti. |
| `DS4_PREAD_SPLIT` | `1` | Sottoletture concorrenti per slab. |
| `DS4_WILLNEED_EXPERTS` | on | `madvise(WILLNEED)` dopo il routing. |
| `DS4_PREFETCH` | off | Read-ahead del layer successivo. |
| `DS4_PREFETCH_EXPERTS` | `0` | Esperti speculativamente prefetched. |
| `DS4_EXPERT_LOOKAHEAD` | `0` | Riempimento anticipato della cache. |
| `DS4_USAGE_FILE` | sibling `.usage.json` | Storico delle selezioni. |
| `DS4_MTLIO` | off | Metal fast resource loading. |
| `DS4_MTLIO_MIN_GBS` | `1.5` | Circuit breaker MetalIO. |
| `DS4_MLOCK` | off | Wiring best-effort dei buffer caldi. |

### Pesi densi, KV e attention

| Parametro | Default | Funzione |
|---|---:|---|
| `DS4_DENSE_STREAM` | off | Stream denso a slot. |
| `DS4_DENSE_AHEAD` | `1` | Profondità di read-ahead denso. |
| `DS4_RESIDENT_DENSE` | off | Copia wired dei pesi non-routed. |
| `DS4_RESIDENT_COMP` | on con dense stream | Compressori residenti. |
| `DS4_LAZY_IDX` | on con dense stream | Caricamento tardivo dello scorer. |
| `DS4_DENSE_Q4` | off, lossy | Requant Q4 delle grandi proiezioni. |
| `DS4_SHARED_Q4` | off, lossy | Requant Q4 dello shared expert. |
| `DS4_QKV_Q4` | off, lossy | Requant Q4 di Q/KV. |
| `DS4_COMP_Q8` | off, lossy | Compressori Q8. |
| `DS4_Q4_CACHE_DIR` | sibling GGUF | Directory cache requant. |
| `DS4_RAW_RING` | off | Ring KV raw sliding-window. |
| `DS4_GPU_INDEXER_TOPK` | on | Top-K indexer su GPU. |
| `DS4_INDEXED_ATTN` | off | Percorso attention indexed sperimentale. |
| `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` | `1024` | Soglia dense/sparse. |
| `DS4_ADAPTIVE_SPLITK` | on | Split-K adattivo. |
| `DS4_FLASH_KV_STAGE` | off | Staging KV fuso. |
| `DS4_ROPE_PAIR` | off | RoPE specializzata a coppie. |
| `DS4_ROPE_AFFINE` | on con pair | Posizioni affini ricostruite su GPU. |

### Kernel, scheduling e diagnostica

| Parametro | Default | Funzione |
|---|---:|---|
| `DS4_FUSED_MOE` | on | Kernel MoE fusi. |
| `DS4_FUSED_HC` | on | Coda HC fusa. |
| `DS4_FUSED_ROUTER_PROBS` | on | Probabilità router fuse. |
| `DS4_FUSED_ROUTER_FINALIZE` | on | Top-K+normalizzazione fusi. |
| `DS4_FUSED_COMP_PROJ` | on | Proiezioni compressore accoppiate. |
| `DS4_ASYNC_FFN` | on | Commit FFN senza attesa immediata. |
| `DS4_ASYNC_ROUTE` | on | Scheduling asincrono route. |
| `DS4_DENSE_Q4_KERNEL` | on | Matvec Q4 denso dedicato. |
| `DS4_Q8_NSG` | `4` | Occupancy matvec Q8. |
| `DS4_MOE_NSG` | `4` | Occupancy MoE. |
| `DS4_DENSE_Q4_NSG` | MoE NSG | Occupancy Q4 denso. |
| `DS4_VECTOR_COPY` | off | Copie vectorized sperimentali. |
| `DS4_PROFILE_ROUTE` | off | Profilo route sincrono. |
| `DS4_SPEC_K` | off | Finestra self-speculative. |
| `DS4_SPEC_DRAFT_EXPERTS` | `2` | Esperti del draft. |
| `DS4_SPEC_VERIFY_BATCH` | on | Verifica speculativa batchata. |
| `DS4_MTP_GGUF` | non impostato | Modello MTP separato. |
| `DS4_N_INDEXER_TOP_K` | modello | Override geometria indexer. |
| `DS4_N_SWA` | modello | Override sliding window. |
| `DS4_SHAPE_FLASH` | auto | Override shape diagnostico. |
| `DS4_SWIGLU_CLAMP_EXP` | modello | Clamp SwiGLU. |

## Inventario GLM 5.2

| Parametro | Default | Funzione |
|---|---:|---|
| `DS4_RESIDENT_LAYERS` | adattivo | Prefisso di layer residenti. |
| `DS4_ACTIVE_EXPERTS` | `8` | Cap esperti routed. |
| `DS4_EXPERT_CACHE_SLOTS` | `16` | Slot cache per layer sparse. |
| `DS4_GLM_EXPERT_ARENA` | `24` | Record dell'arena expert-major. |
| `DS4_STREAM_SLOTS` | `3` | Slot dello streamer di layer. |
| `DS4_GLM_SG` | on | Matvec cooperativi simdgroup. |
| `DS4_NSG` | `4` | Righe/simdgroup per threadgroup. |
| `DS4_GLM_MOE_BATCH` | on | MoE batchato nel decode. |
| `DS4_GLM_GPU_ROUTER` | on | Router fuso su GPU. |
| `DS4_GLM_FUSE` | on | Fusione commit FFN/trunk. |
| `DS4_MLOCK` | on | Wiring dei pesi residenti. |
| `DS4_MTLIO` | on | MetalIO per layer streaming. |
| `DS4_GLM_NOCACHE` | off | `F_NOCACHE` sul reader. |
| `DS4_PREAD_SPLIT` | `4` | Split concorrenti nel prefill. |
| `DS4_PREFILL_BATCH` | off | Route batch multi-token. |
| `DS4_PREFILL_ROUTE_BATCH` | `16` | Token per route batch. |
| `DS4_PREFILL_MOE_BATCH` | on | Fase MoE expert-major multi-token. |
| `DS4_GLM_SPEC_EXPERTS` | backend | Esperti del percorso speculativo. |
| `DS4_SPEC_K` | off | Prompt-lookup speculative decode. |
| `DS4_USAGE_FILE` | sibling `.glm-usage.json` | Storico routing. |
| `DS4_BUNDLE_DIR` | sibling `.glm-experts` | Bundle esperti legacy. |
| `DS4_GLM_LAYERQ4` | on | Usa tensori Q4 del sidecar unificato. |
| `DS4_GLM_LAYERQ4_DIR` | sibling `.glm-layers-q4` | Directory sidecar. |
| `DS4_GLM_BUILD_BUNDLES` | off | Prepass builder bundle legacy. |
| `DS4_GLM_BUNDLE_LAYERS` | tutti | Limite builder bundle. |
| `DS4_GLM_BUILD_LAYERQ4` | off | Prepass builder sidecar Q4. |
| `DS4_GLM_LAYERQ4_LAYERS` | tutti | Limite builder Q4. |
| `DS4_GLM_AUTOTUNE` | off | Sweep in-process dei knob lossless. |
| `DS4_GLM52_SPARSE_GGUF` | non impostato | Fixture d'integrazione, non runtime normale. |

## Inventario Laguna S 2.1

| Parametro | Default | Funzione |
|---|---:|---|
| `DS4_LAGUNA_RUNTIME` | off | Gate esplicito del backend. |
| `DS4_SSD_STREAM` | on nel preset GUI, off nella demo | Streaming degli esperti routed; i chiamanti senza preset usano la selezione automatica file/RAM. |
| `DS4_EXPERT_CACHE_MB` | `3072` nel preset GUI, `2048` nel motore | Budget globale cache esperti. Sul target M1 Pro con circa 10 GiB liberi, l’A/B top-10 favorisce 3 GiB; 4 GiB aumenta gli hit ma introduce pressione memoria. |
| `DS4_EXPERT_CACHE_SLOTS` | derivato dai MiB | Numero esplicito di slot globali; ha precedenza sul budget. |
| `DS4_MULTI_QUANT_CACHE` | on | Dimensiona gli slot Q2_K/Q3_K sui byte reali mantenendo lo stesso budget; `0` ripristina gli slot legacy tutti grandi come Q3_K. |
| `DS4_ACTIVE_EXPERTS` | `10` | Top-N routed realmente eseguito (`1...10`), con rinormalizzazione dei pesi. |
| `DS4_RESIDENT_LAYERS` | `0` con streaming | Prefisso di layer routed i cui esperti restano residenti. |
| `DS4_LAGUNA_LAYERS` | tutti | Tronca lo stack per bring-up. |
| `DS4_KV_INITIAL` | `512` | Righe iniziali dei 12 layer full-attention; crescita geometrica fino al contesto configurato. Evita di riservare subito la KV massima. |
| `DS4_PREFILL_BATCH` | on | QK/RoPE e attention causale batchati. |
| `DS4_PREFILL_DENSE_MM` | on | RMSNorm, Q/K/V/gate, output attention, router e FFN denso/condiviso multi-token; `0` ripristina il percorso matvec. |
| `DS4_PREFILL_MOE_BATCH` | off | MoE expert-major sperimentale; A/B regressivo. |
| `DS4_PREFILL_CHUNK` | `256` | Limite del chunk layer-major (`1...1024`); la cache può ridurlo. |
| `DS4_DECODE_CHAINED` | off | Variante A/B: accoda tail esperti N + trunk N+1; regressiva sul M1 Pro misurato. |
| `DS4_DECODE_SPLIT_K` | off | Variante A/B GQA split-K sui 12 layer globali; regressiva sul M1 Pro misurato. |
| `DS4_DECODE_SPLIT_K_MIN` | `384` | Soglia di token visibili per lo split-K globale. |
| `DS4_SHARED_EXPERT_OVERLAP` | off | Calcola lo shared expert residente mentre avanzano le letture SSD routed; l'A/B stabilizzato sul M1 Pro favorisce off. |
| `DS4_INDEXED_ATTN` | on | Indice compresso a blocchi e attention sparsa GPU sui layer globali oltre soglia. |
| `DS4_LONG_ATTN_BLOCK` | `16` | Token per centroide dell'indice. |
| `DS4_LONG_ATTN_TOP_BLOCKS` | `32` | Blocchi storici selezionati per testa. |
| `DS4_LONG_ATTN_RECENT` | `512` | Coda recente sempre densa. |
| `DS4_LONG_ATTN_THRESHOLD` | `4096` | Token visibili prima di attivare la selezione sparsa. |
| `DS4_DISCARD_UPLOAD_PAGES` | off | Variante A/B: scarta le pagine mmap già copiate; leggermente regressiva sul target misurato. |
| `DS4_RESIDENT_PRIVATE` | off | Variante A/B: copia i pesi residenti in buffer Metal private via staging riusabile. |
| `DS4_EXPERT_CACHE_PARTITIONED` | off | Variante A/B: partiziona gli slot per layer quando il budget copre tutti i top-N; regressiva sul target misurato. |
| `DS4_EXPERT_PREAD` | on | Miss esperti via `pread/F_NOCACHE`; off usa mmap+memcpy. |
| `DS4_PREAD_SPLIT` | `1` | Range concorrenti per slab (`1...8`). |
| `DS4_WILLNEED_EXPERTS` | on | `WILLNEED` sui soli slab selezionati nel percorso mmap. |
| `DS4_MTLIO` | off | Caricamento diretto SSD → `MTLBuffer`, con fallback permanente. |
| `DS4_MLOCK` | off | Pin best-effort di output head e pool/staging esperti. |
| `DS4_NSG` | `4` | SIMD group per threadgroup (`1...8`). |

Il preset GUI per M1 Pro con circa 10 GiB disponibili usa streaming SSD,
cache esperti 3.072 MiB multi-quant, top-10, nessun layer routed residente, chunk 256,
attention e proiezioni dense batchate, MoE expert-major disattivato,
KV full-attention lazy da 512 righe, decode chained/split-K e rilascio
pagine upload disattivati, `pread×1`,
MetalIO/mlock disattivati e NSG 4. La GUI applica il preset canonico soltanto
dopo aver identificato Laguna, così i valori comuni rimasti dal backend
caricato in precedenza non contaminano quello selezionato.

La GUI espone “KV full-attention iniziale” con 256/512/1024/2048/4096
righe e persiste la scelta in `DS4LagunaKVInitial`; 512 è il default
consigliato. La scelta viene passata direttamente al servizio al prossimo
caricamento e ha precedenza sul preset d'ambiente della GUI.

## Alias storici deprecati

| Nome comune | Alias storici |
|---|---|
| `DS4_ACTIVE_EXPERTS` | `DS4_GLM_ACTIVE_EXPERTS`, `DS4_LAGUNA_ACTIVE_EXPERTS` |
| `DS4_EXPERT_CACHE_SLOTS` | `DS4_GLM_EXPERT_SLOTS`, `DS4_LAGUNA_EXPERT_CACHE_SLOTS` |
| `DS4_EXPERT_CACHE_MB` | `DS4_LAGUNA_EXPERT_CACHE_MB` |
| `DS4_SSD_STREAM` | `DS4_LAGUNA_SSD_STREAM` |
| `DS4_RESIDENT_LAYERS` | `DS4_GLM_RESIDENT_LAYERS`, `DS4_LAGUNA_RESIDENT_LAYERS` |
| `DS4_STREAM_SLOTS` | `DS4_GLM_STREAM_SLOTS` |
| `DS4_PREFILL_BATCH` | `DS4_PREFILL_BATCH_ATTN`, `DS4_GLM_PREFILL_BATCH`, `DS4_LAGUNA_PREFILL_BATCH` |
| `DS4_PREFILL_DENSE_MM` | `DS4_LAGUNA_PREFILL_DENSE_MM` |
| `DS4_PREFILL_MOE_BATCH` | `DS4_PREFILL_FFN_BATCH`, `DS4_GLM_PREFILL_MOE`, `DS4_LAGUNA_PREFILL_MOE_BATCH` |
| `DS4_PREFILL_ROUTE_BATCH` | `DS4_GLM_PREFILL_ROUTE_BATCH` |
| `DS4_PREFILL_CHUNK` | `DS4_LAGUNA_PREFILL_CHUNK` |
| `DS4_KV_INITIAL` | `DS4_LAGUNA_KV_INITIAL` |
| `DS4_SHARED_EXPERT_OVERLAP` | `DS4_LAGUNA_SHARED_EXPERT_OVERLAP` |
| `DS4_INDEXED_ATTN` | `DS4_LAGUNA_INDEXED_ATTN` |
| `DS4_LONG_ATTN_BLOCK` | `DS4_LAGUNA_INDEXED_ATTN_BLOCK` |
| `DS4_LONG_ATTN_TOP_BLOCKS` | `DS4_LAGUNA_INDEXED_ATTN_TOP_BLOCKS` |
| `DS4_LONG_ATTN_RECENT` | `DS4_LAGUNA_INDEXED_ATTN_RECENT` |
| `DS4_LONG_ATTN_THRESHOLD` | `DS4_LAGUNA_INDEXED_ATTN_THRESHOLD` |
| `DS4_MLOCK` | `DS4_GLM_MLOCK`, `DS4_LAGUNA_MLOCK` |
| `DS4_MTLIO` | `DS4_GLM_MTLIO`, `DS4_LAGUNA_MTLIO` |
| `DS4_PREAD_SPLIT` | `DS4_GLM_READ_SPLIT`, `DS4_LAGUNA_PREAD_SPLIT` |
| `DS4_EXPERT_PREAD` | `DS4_LAGUNA_EXPERT_PREAD` |
| `DS4_WILLNEED_EXPERTS` | `DS4_LAGUNA_WILLNEED_EXPERTS` |
| `DS4_BUNDLE_DIR` | `DS4_GLM_BUNDLE_DIR` |
| `DS4_USAGE_FILE` | `DS4_GLM_USAGE_FILE` |
| `DS4_SPEC_K` | `DS4_GLM_SPEC_K` |
| `DS4_NSG` | `DS4_GLM_NSG`, `DS4_LAGUNA_NSG` |
| `DS4_DECODE_CHAINED` | `DS4_LAGUNA_DECODE_CHAINED` |
| `DS4_DECODE_SPLIT_K` | `DS4_LAGUNA_DECODE_SPLIT_K` |
| `DS4_DECODE_SPLIT_K_MIN` | `DS4_LAGUNA_DECODE_SPLIT_K_MIN` |
| `DS4_DISCARD_UPLOAD_PAGES` | `DS4_LAGUNA_DISCARD_UPLOAD_PAGES` |
| `DS4_RESIDENT_PRIVATE` | `DS4_LAGUNA_RESIDENT_PRIVATE` |
| `DS4_EXPERT_CACHE_PARTITIONED` | `DS4_LAGUNA_EXPERT_CACHE_PARTITIONED` |

## Cosa non va unificato alla cieca

- `DS4_LAGUNA_LAYERS` tronca il modello; `DS4_RESIDENT_LAYERS` conserva il
  modello completo e rende residenti i primi N layer routed. Non sono alias.
- Gli slot cache DeepSeek/GLM sono per layer; il budget Laguna è globale e
  dipende dalla dimensione Q2/Q3/Q4 dello slab. Anche
  `DS4_EXPERT_CACHE_SLOTS` è globale su Laguna: uniforma il conteggio degli
  slot, non la loro distribuzione.
- `DS4_DENSE_AHEAD` e `DS4_STREAM_SLOTS` regolano pipeline diverse.
- `DS4_GLM_NOCACHE` coinvolge il reader del layer; `DS4_EXPERT_PREAD`
  DeepSeek riguarda il gather degli esperti. Non sono equivalenti.
- `DS4_SPEC_K` condivide l'intenzione ma non l'algoritmo: self-speculative
  su DeepSeek, prompt lookup su GLM.
