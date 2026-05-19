# Caricamento del modello

Come un checkpoint da 142 GB diventa un `Transformer` usabile su un
Mac da 16 GB, cosa fa `LoadPlan` prima di committare, in cosa
differiscono le tre strategie del loader, cosa ti compra lo streaming
pool, e come è cablata la persistenza KV cache cross-restart.

Documenti complementari:

- [`MODEL.md`](MODEL.md) — cosa il loader consegna.
- [`MEMORY.md`](MEMORY.md) — walkthrough mmap, stime working-set,
  comportamento page-cache. Un po' di contenuto si sovrappone;
  questo doc si focalizza sul *control flow* (decidere, aprire,
  streamare).
- [`MODULES.md`](MODULES.md) — indice per-file.
- [`USAGE.md`](USAGE.md) — riferimento operativo (flag, model
  picker).

> 🇬🇧 La versione inglese è [`LOADING.md`](LOADING.md).

---

## 1. La pipeline

```
~/Downloads/V4-Flash-HF/                       (o directory post-converter)
├── config.json
├── generation_config.json
├── tokenizer.json + tokenizer_config.json
├── model.safetensors.index.json
├── model-00001-of-00046.safetensors
├── …
└── model-00046-of-00046.safetensors

           Transformer.load(config:from:strategyOverride:forceLoad:warmupOnLoad:kvCacheFile:)
                                  │
                                  ▼
                  ┌─────────────────────────────────────┐
                  │ LoadPlan.decide(modelDir:override:)   │
                  │                                       │
                  │  • WeightLoader.discoverShards(...)   │
                  │  • SystemProbe.effectiveProcessBudget │
                  │  • pickStrategy(...) → preload/mmap/  │
                  │    streaming                          │
                  │  • refusal shardTooLarge /            │
                  │    kvCacheTooLarge (force-load bypassa)│
                  └──────────────┬────────────────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────────────┐
                  │ WeightLoader(plan: ...)               │
                  │                                       │
                  │  • mmap di ogni shard → MTLBuffer     │
                  │  • parse header → name → shard idx    │
                  │  • build shardLayers ownership table  │
                  │  • streaming? build StreamingPool     │
                  └──────────────┬────────────────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────────────┐
                  │ ModelConfig.inferred(from: loader)    │
                  │ (patch campi config missing/stale     │
                  │  dalle shape reali dei tensor)        │
                  └──────────────┬────────────────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────────────┐
                  │ KV cache: proiezione byte, rifiuto se │
                  │ projectedKVCacheBytes > budget.       │
                  │ Opzionale: apri KVCacheFile per       │
                  │ persistenza cross-restart.            │
                  └──────────────┬────────────────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────────────┐
                  │ Percorri l'albero canonico V4:        │
                  │ embed, norm, head, hc_head_*,         │
                  │ per layer: attn, ffn, hc_attn_*,      │
                  │ hc_ffn_*, compressor, indexer, …      │
                  │ (Sources/DeepSeekKit/Assembly.swift)  │
                  │ Nomi mancanti → init random + warn.   │
                  └──────────────┬────────────────────────┘
                                 │
                                 ▼
                       Transformer (pronto per forward)
```

Tre fasi che vale la pena capire separatamente: **planning** (decidere
senza committare), **opening** (mmap + index), **assembly** (leggere
l'albero canonico dei nomi, costruire gli oggetti Swift). Lo streaming
sovrappone una quarta fase che gira *durante* l'inference.

---

## 2. Planning: `LoadPlan.decide`

`Sources/DeepSeekKit/LoadStrategy.swift:193`.

Input: directory del modello + override di strategia opzionale
(`"auto"|"preload"|"mmap"|"streaming"` da `--load-strategy`) +
`forceLoad: Bool` (il flag CLI `--force-load`).

Output: un `LoadPlan` con la strategia scelta + ogni numero della
sonda che vale la pena loggare (byte totali, byte max shard, RAM
disponibile, RAM fisica, MTL recommended working set, cpu cores) +
una ragione one-line "perché questa strategia".

### Le tre strategie

| Strategia | Quando | Comportamento |
|---|---|---|
| `.preload` | totale ≤ 80% del budget effettivo | `pread(2)` di ogni shard in un MTLBuffer appena allocato. Steady-state più veloce, cold-start più costoso. Concorrente (capped a 4 stream per evitare contention SSD). |
| `.mmap` | totale > 80% del budget ma rapporto < 10× | `mmap(PROT_READ, MAP_PRIVATE)` per shard, wrap come `MTLBuffer(bytesNoCopy:)`. L'OS pagina on-demand. Default per "il checkpoint entra in RAM ma a fatica". |
| `.streaming` | rapporto > 10× il budget, O forzato | Stesso mmap di sopra, più uno **StreamingPool** che ruota gli shard per-layer in un pool a dimensione fissa di slot pinnati. Working set bounded, più lento per-token, l'unico modo per far girare V4-Flash su un Mac da 16 GB. |

### La matrice di decisione

`pickStrategy(total: available: totalOversubMultiplier: override:)`
a `LoadStrategy.swift:252`:

```swift
override == nil/"auto":
    if available == 0:                     → .mmap     (probe fallita, fallback safe)
    if total > available * 10:             → .streaming (oversub selvaggia)
    if total ≤ 0.80 × available:           → .preload   (entra comodamente)
    altrimenti:                            → .mmap     (entra ma stretto)
override == "preload"/"mmap"/"streaming":  → quello, con l'override come ragione
```

La coppia numeratore/denominatore (4/5 = 0,80) e il moltiplicatore
(10) sono stati lockati dopo testing su host da 16 GB, 64 GB, 128 GB,
e 192 GB. Vedi i commenti inline in `LoadStrategy.swift:140-169` per
il razionale.

### I guard di rifiuto

Due errori possono essere lanciati *prima* che la strategia sia
anche solo scelta (a meno che `--force-load` non sia attivo):

- **`shardTooLarge`** — il più grande shard singolo eccede il 50% del
  budget effettivo. Il kernel non può mmappare uno shard
  contiguamente se compete per la memoria unified con il compositor
  della GUI e il working-set di command-buffer della GPU a ogni nuovo
  hot-tensor fault. Rifiutare qui lascia headroom. Risoluzioni nel
  messaggio d'errore: liberare memoria, ri-shardare con
  `--shard-size-gb` minore, o `--force-load`.
- **`kvCacheTooLarge`** — guard separato, scatta *dopo* che il
  loader è girato ma prima dell'assembly dei block.
  `ModelConfig.projectedKVCacheBytes` eccede il budget. Lo streaming
  non aiuta qui: le KV cache sono `MTLBuffer` `storageModeShared`
  dense, non pagine di file mmappate, e la GPU ci scrive durante il
  forward. `--force-load` non aiuta neanche. L'errore dice all'utente
  di abbassare `max_position_embeddings` / `max_batch_size` in
  `config.json` (la trappola comune: HF spedisce
  `max_position_embeddings = 1 M` che su un Mac da 16 GB vuole ~50 GB
  di KV state).

### Il budget unified memory

`SystemProbe.effectiveProcessBudget()` ritorna
`min(processAvailableRAM, physicalRAM × 0.60)`. Su Apple Silicon CPU
e GPU condividono le stesse pagine fisiche, quindi la cifra
"available" (free + inactive + speculative pages da
`host_statistics64`) è ottimista — quelle pagine inactive sono file
cache e working set di altre app che il kernel evicta solo sotto
pressione reale. Una volta che iniziamo a mmappare un checkpoint da
qualche centinaio di GB il kernel è forzato a evictare, il working
set della GPU compete per le stesse pagine, e l'intero sistema
thrasha.

Trattare il 60% del fisico come tetto tiene ~40% per l'OS, la GUI,
la command queue GPU, e qualunque altra app aperta. Tunabile via
l'argomento `physicalCap:`; il default 0,60 è quello che i guard di
rifiuto usano.

---

## 3. Opening: `WeightLoader.init`

`Sources/DeepSeekKit/WeightLoader.swift:53`. Dopo che il plan è
deciso, il loader:

1. **Scopre gli shard**: `discoverShards(in:)` enumera i file
   `*.safetensors` nella directory, ordina per filename, droppa
   tutto < 1 KiB (stub LFS pointer). Throwa se non resta niente.
2. **Apre ogni shard**: `SafeTensorsFile(url:)` fa `open` → `fstat`
   → arrotonda al page size → `mmap(MAP_PRIVATE)` →
   `device.makeBuffer(bytesNoCopy: ..., deallocator: { munmap(...) })`.
   Il deallocator gira quando l'MTLBuffer è rilasciato (ARC), così
   il `munmap` è automatico all'exit del processo.
3. **Parsa gli header**: ogni shard inizia con una lunghezza
   little-endian da 8 byte + un header JSON che elenca il nome,
   dtype, shape, e `[data_offset_start, data_offset_end]` di ogni
   tensor. Il loader costruisce una mappa flat
   `[String: shardIndex]` per i lookup `load(_:)`.
4. **Classifica gli shard** (ownership per-layer): `buildShardLayers`
   percorre ogni nome. Uno shard "possiede" il layer K iff ogni
   tensor in esso ha un prefisso `layers.K.…`; altrimenti è uno
   shard "shared" (owner = -1), che tiene tensor top-level (embed,
   head, norm, `hc_head_*`) che vengono toccati a ogni forward.
5. **(solo streaming) Costruisce lo StreamingPool**:
   `StreamingPool(shards:shardLayers:...)`. Alloca un MTLBuffer per
   gli shard shared (mlock'd) e un altro `slotCount × slotSize`
   MTLBuffer per gli slot rotating per-layer. Risolve la
   `TensorLocation` di ogni tensor (slot + offset + shape + dtype)
   in fase di costruzione così che le successive call `load(name)`
   non abbiano neanche bisogno di un dictionary lookup oltre alla
   mappa di location risolta.

### Perché esiste lo streaming pool

Quando wrappi una regione mmap'd come MTLBuffer via `bytesNoCopy:`,
il driver Apple *pinna* quelle pagine ogni volta che il buffer è
referenziato. `madvise(MADV_DONTNEED)` ritorna successo ma `mincore()`
mostra zero pagine droppate — il kernel non evicta pagine che il
driver reclama. Con 147 GB di V4-Flash mmappati attraverso 45 buffer
del genere, il sistema gira al 100% memory pressure
permanentemente. O crasha o rende l'inference inusabilmente lenta.

Lo streaming pool by-passa questo allocando i propri MTLBuffer
`storageModeShared` (da cui il kernel *può* evictare, perché ne
gestiamo i contenuti esplicitamente) e usando `pread(2)` per copiare
dati shard dentro / fuori al volo. Lo mmap resta per il parse degli
header; i byte effettivi dei pesi fluiscono attraverso i buffer del
pool.

### Lookup dei tensor

`WeightLoader.load(_:)`:

- **modalità mmap / preload**: lookup `index[name] → shardIdx`,
  delegato a `shards[shardIdx].load(name)` che ritorna un Tensor che
  condivide l'MTLBuffer dello shard con l'offset giusto.
- **modalità streaming**: lookup `pool.tensorLocation[name]`, ritorna
  un Tensor che punta in `pool.sharedSlot` o `pool.rotatingSlot` a
  `loc.offsetInSlot`. I dati di backing del pool non sono validi
  finché `ensureLayer(K)` non è stato chiamato per il layer che lo
  possiede — il forward pass (`Transformer.forward`) lo assicura
  prima di referenziare il tensor.

`tryLoad(_:[fallbackNames])` esiste per coppie alias HF / convertite:
`"head.weight"` vs `"lm_head.weight"`, `"<base>.scale"` vs
`"<base>.weight_scale_inv"`, etc.

`shape(of name:)` / `shape(ofAny:)` interrogano la shape on-disk
senza caricare byte — usate da `ModelConfig.inferred(from:)` per
patchare un config stale.

### Nomi mancanti

`WeightLoader.missing: Set<String>` raccoglie ogni nome che è
tornato nil. Assembly.swift riempie i tensor mancanti con init random
(via `MiniRNG`) e stampa un summary su stderr alla fine. È
intenzionale: lascia che un checkpoint parzialmente prunato produca
comunque un forward (utile in fase di porting, debug, o iterazione
sulle shape dei layer prima che il release sia finale).

---

## 4. Lo StreamingPool

`Sources/DeepSeekKit/StreamingPool.swift:58`. Attivato solo in
strategia `.streaming`. Due regioni MTLBuffer, entrambe
`.storageModeShared`, allocate una volta al load time:

### Slot shared

Dati concatenati di ogni shard "shared" (tensor top-level).
Dimensionato esattamente per stare nella somma dei loro byte count.
`mlock`'d alla costruzione così il kernel non può evictare — questi
tensor vengono toccati a ogni layer.

### Slot rotating

Un MTLBuffer di dimensione `slotCount × slotSize` byte, ritagliato in
`slotCount` sub-slot. Ogni shard per-layer K è **permanentemente
assegnato** al sub-slot `K mod slotCount` — l'indice di slot è una
proprietà di K, non dell'ordine di accesso.

```
rotatingSlot (un MTLBuffer):
[ sub-slot 0 ][ sub-slot 1 ][ sub-slot 2 ] ... [ sub-slot N-1 ]
  slotSize     slotSize       slotSize           slotSize

layer K=0  → sub-slot 0
layer K=1  → sub-slot 1
…
layer K=N  → sub-slot 0      (sovrascrive layer 0 in quello slot)
layer K=N+1 → sub-slot 1
…
```

### Perché assegnazione modulare

`Tensor` cattura `MTLBuffer + offset` in fase di costruzione
(Assembly.swift costruisce i pesi di ogni block upfront, prima del
primo forward). Per evitare un layer di indirezione su ogni accesso
tensor, i byte del layer K devono sempre vivere allo stesso indirizzo.
Con sub-slot = `K mod N`, l'indirizzo è
`rotatingSlot.contents() + (K mod N) * slotSize + inShardOffset` —
stabile per la lifetime del pool. La rotazione successiva
sovrascrive lo stesso spazio di indirizzi.

### Sliding window

Con N sub-slot e forward strettamente sequenziale (layer 0, 1, ...,
L-1), il working set per layer è 1 e la window prefetched è N-1.
Dopo aver computato il layer K e fatto `releaseLayer(K)`, il pool
schedula un `pread` in background del layer K+N nel sub-slot
`(K+N) mod N = K mod N` — cioè lo slot che tiene K, che non è più
necessario perché la GPU ha finito con esso prima che `releaseLayer`
girasse (`cmdL.waitUntilCompleted` prima della call).

Quando `ensureLayer(K+N)` è chiamato, il prefetch è già completo e
il fast path ritorna senza I/O.

### Sizing del pool

`WeightLoader.computeStreamingSlotCount(...)` decide N all'init:

```
sharedBytes = somma dei size degli shard shared
maxLayerBytes = max(per-layer shard size, arrotondato al page)
slotSize = aligned(maxLayerBytes)
reserveBytes = 4 GiB (env DEEPSEEK_STREAMING_RESERVE_GB)
budgetCap = SystemProbe.effectiveProcessBudget()

rotatingBudget = budgetCap - sharedBytes - reserveBytes
N = min(layerShardCount, rotatingBudget / slotSize)
```

Cappato al numero totale di shard per-layer (= "ogni shard
preloadato"). Lower bound 1 (ogni transizione layer paga un pread
bloccante). Il riserva 4 GiB copre la KV cache, buffer di
attivazione, la Metal command queue, e headroom per l'OS / GUI.

Override via env var `DEEPSEEK_STREAMING_SLOTS` o
`DEEPSEEK_STREAMING_RESERVE_GB`.

---

## 5. Rifiuto da proiezione KV cache

Dopo che il loader è costruito ma prima dell'assembly dei block,
`Assembly.swift` chiama `config.projectedKVCacheBytes` e rifiuta se
eccede il budget:

```swift
let kvProjected = config.projectedKVCacheBytes
let kvBudget = SystemProbe.effectiveProcessBudget()
if kvBudget > 0, kvProjected > kvBudget {
    throw LoadStrategyError.kvCacheTooLarge(
        projected: kvProjected, available: kvBudget,
        maxSeqLen: config.maxSeqLen,
        maxBatchSize: config.maxBatchSize)
}
```

`projectedKVCacheBytes` percorre il `compress_ratio` di ogni layer e
somma:

- KV cache attention: `maxBatchSize · (windowSize + maxSeqLen/ratio) · headDim · 4 byte` (ratio > 0) o `maxBatchSize · windowSize · headDim · 4` (ratio == 0).
- KV cache indexer: stessa formula, solo sui layer `ratio == 4`.
- State del compressor: `hcMult · ratio · maxBatchSize · headDim · 4 byte` (ratio > 0).
- KV cache attention di ogni layer MTP (~uno batch).

Per il config V4-Flash di default (`max_seq_len = 4096`, `max_batch = 4`)
sta a < 100 MB totali. Per `max_seq_len = 1 M` scala a decine di GB —
che è quello che il rifiuto protegge contro.

Il testo di risoluzione dell'errore raccomanda `jq` su `config.json`
per abbassare `max_position_embeddings` e `max_batch_size`.

---

## 6. Persistenza KV cache cross-restart

Opzionale. Quando `Transformer.load(..., kvCacheFile:)` riceve un
`KVCacheFile`, ogni KV cache per-layer + ogni tensor di state
Compressor / Indexer diventa una slice zero-copy di un file di
backing mmappato invece di un `MTLBuffer` fresco in-memory. Chiudere
il modello e riaprirlo più tardi ri-mmappa lo stesso file; i tensor
puntano agli stessi offset; lo state KV è "automaticamente"
preservato.

### `KVCacheFile`

`Sources/DeepSeekKit/KVCacheFile.swift:28`. Formato on-disk:

```
[ header 4096-byte ][ payload (page-aligned) ]
```

Campi dell'header (`KVCacheFile.Header`):
- `magic` = `'KVC1'` (0x4B564331)
- `version`
- `payloadBytes`
- `prefilledTokens` (ultimo checkpoint)
- `historyHashLow` / `historyHashHigh` (hash a 64-bit della history
  prompt all'ultimo checkpoint, per rilevare resumption mismatched)
- `modelPathHash` (quale modello ha prodotto questo state, per
  rifiutare resumption contro un modello diverso)

API:
- `init(url:payloadBytes:modelPathHash:)` — crea il file se mancante,
  dimensiona il payload, lo mmappa.
- `readHeader()` / `resetHeader(...)` / `updateCheckpoint(...)`.
- `region(offset:length:)` — ottieni una coppia `MTLBuffer + offset`
  per una slice contigua (usato da `Assembly` per fare da backing ai
  singoli KV tensor).
- `tensor(at: KVCacheLayout.Region, shape:, dtype:)` — wrapper di
  convenienza.
- `attemptResume(newTokens:)` — confronta il `historyHash` salvato
  con l'hash del nuovo prompt, trova il prefisso comune, ritorna
  `.fullMatch / .partialMatch(P) / .mismatch`.

### `KVCacheLayout`

`Sources/DeepSeekKit/KVCacheLayout.swift:25`. Calcola gli offset byte
per-layer in anticipo da un `ModelConfig`. Un `LayerOffsets` per main
layer + uno per MTP layer; ognuno contiene `Region` opzionali (offset
+ byte) per `attnKVCache`, `compressorKVState`,
`compressorScoreState`, `indexerKVCache`, `indexerCompressorKVState`,
`indexerCompressorScoreState`.

Allineato alle pagine da 16 KiB (page size Apple Silicon); costa
fino a 16 KiB di slack per layer.

### Manifest

Il KVCacheFile salva anche un manifest (`<file>.manifest`) che porta
la *sequenza di token* contro cui la cache è stata costruita.
`readManifest()` / `writeManifest(_:)` / `readManifestFull()` (che
aggiunge `lastLogits` + `chunkAlignment` opzionali).

Il manifest è quello che rende il resume safe: alla riapertura,
l'host re-tokenizza la history della conversazione, confronta col
manifest salvato (`commonPrefixLength`), e o:
- **Full match**: la cache è esattamente quella che serve, niente
  prefill.
- **Partial match (lunghezza P)**: rewind della KV cache a position
  P (`Transformer.rewindKVTo(pos: P)`), poi prefill solo del delta.
- **Mismatch**: scarta la cache, cold prefill da zero.

`KVCacheFile.attemptResume(newTokens:)` ritorna l'esito matching.

### Dove è collegato

L'app desktop usa un `KVCacheFile` per conversazione, salvato sotto
`Application Support/.../conversations/<id>.kvcache`. La CLI non lo
usa (cold-prefill sempre). Vedi `PersistencePaths.swift` per il
layout dei file.

---

## 7. Warm-up opzionale

`Transformer.load(..., warmupOnLoad: true)` chiama
`loader.warmupAllShards()` tra l'index + l'assembly dei block. Questo
pre-faulta tutte le pagine di ogni shard in RAM così il primo
forward non paga la latenza di page-fault per-tensor.

Auto-skippato quando `dimensione modello > RAM fisica × 1.5` (il
warm-up di un file più grande della RAM è inutile — l'OS evicterà).

La CLI lo espone via `--warmup-on-load`; il tab Loading Settings
della GUI ha un toggle.

---

## 8. Hook streaming-mode nel forward pass

`Transformer.forward(...)` (`Sources/DeepSeekKit/Model.swift:214`)
controlla `weightLoader?.streamingEnabled` e chiama:

```swift
for (k, layer) in layers.enumerated() {
    loader?.ensureLayer(k)                   // (1) page-in (o pool-load) shard K
    let cmdL = Device.shared.queue.makeCommandBuffer()!
    x = layer(x, startPos: ..., inputIds: ..., in: &cmdL)
    cmdL.commit(); cmdL.waitUntilCompleted()
    loader?.releaseLayer(k)                  // (2) MADV_DONTNEED o schedula prefetch
}
```

(1) `ensureLayer(K)` in modalità pool kicka il `pread` rotating-slot
se lo slot per `K mod N` non tiene attualmente il layer K; in
modalità mmap è un hint `MADV_WILLNEED`.

(2) `releaseLayer(K)` in modalità pool schedula un prefetch
background del layer K+N; in modalità mmap è `MADV_DONTNEED` sullo
shard di K-1 (un layer indietro così le letture correnti della GPU
non sono disturbate).

Una revisione precedente chiamava `MADV_WILLNEED` su K+1 *prima* di
computare K. Andò male: il kernel iniziava a tirare le pagine di K+1
mentre il `MADV_DONTNEED` di K-1 non era ancora stato onorato → ~3
layer simultaneamente residenti → OOM su Mac da 16 GB. Lasciare che
il path naturale di page-fault gestisca il layer successivo tiene la
residency strettamente a "il layer che la GPU sta leggendo
attualmente".

---

## 9. Mapping dei flag CLI e controlli GUI

```
swift run deepseek <model-dir> "prompt" \
    [--load-strategy auto|preload|mmap|streaming] \
    [--force-load] \
    [--warmup-on-load] \
    [--max-seq-len N] \
    [--max-batch-size N]
```

| Flag | Effetto |
|---|---|
| `--load-strategy` | Override della scelta automatica di `LoadPlan.decide`. |
| `--force-load` | Bypass del rifiuto `shardTooLarge` (NON bypassa `kvCacheTooLarge` — quello è hard). |
| `--warmup-on-load` | Pre-faulta le pagine di ogni shard. |
| `--max-seq-len` | Abbassa le righe della KV cache per layer per stare in RAM. |
| `--max-batch-size` | Stessa cosa, dimensione batch. |

Controparti GUI (Settings → tab Loading):
- Picker strategia (`auto` / `preload` / `mmap` / `streaming`).
- Toggle force-load.
- Toggle warm-up.
- (Tab Model Config) override field scrivono su
  `Application Support/.../config-overrides.json`.

---

## 10. Diagnostica

Il loader stampa un summary multi-line su stderr di default:

```
system: 16.00 GB unified (CPU + GPU share this pool)
        8.54 GB effective budget for this process
        8 cores · GPU rec. working-set 10.50 GB (same pool)
checkpoint: 46 shards, 142.30 GB total, largest 3.42 GB
oversubscription: 16.7× of effective budget
strategy: streaming (auto: total is 16.7× effective budget (cap 10×) — streaming with per-layer madvise hints)

Indexed 1843 tensors across 46 shard(s).
Projected KV cache: 0.04 GB at max_seq_len=4096, max_batch_size=4.
…
load:start → load:after-mmap → load:embed+head-built → load:layers-built → load:complete
```

Ogni transizione è una call `MemoryLogger.snapshot(...)` gated da
`DEEPSEEK_MEM_LOG=1`. Utile quando si investiga "dove è spike-ato il
working set?" — ogni transizione ha la VM stat matching dumpata.

---

## 11. Source map

| Topic | File |
|---|---|
| Decisione di strategia + guard di rifiuto | `Sources/DeepSeekKit/LoadStrategy.swift` |
| Sonde di sistema (RAM, GPU, core) | `Sources/DeepSeekKit/SystemProbe.swift` |
| Loader (modalità mmap / preload) | `Sources/DeepSeekKit/WeightLoader.swift` |
| Reader mmap SafeTensors | `Sources/DeepSeekKit/SafeTensors.swift` |
| Streaming pool | `Sources/DeepSeekKit/StreamingPool.swift` |
| Assembly dell'albero pesi | `Sources/DeepSeekKit/Assembly.swift` |
| Inferenza config da shape dei tensor | `Sources/DeepSeekKit/Config.swift` (`inferred(from:)`) |
| Formato file KV cache | `Sources/DeepSeekKit/KVCacheFile.swift` |
| Layout KV cache (offset byte) | `Sources/DeepSeekKit/KVCacheLayout.swift` |
| Layout dei path di persistenza | `Sources/DeepSeekUI/Utility/PersistencePaths.swift` |
| MemoryLogger | `Sources/DeepSeekKit/MemoryLogger.swift` |
| Plumbing dei flag CLI | `Sources/deepseek/main.swift` |
| Settings GUI loading-related | `Sources/DeepSeekUI/Views/Settings/LoadingSettingsTab.swift` |

---

## 12. Modi di fallimento e recovery

Gli errori ricorrenti visibili all'utente:

| Sintomo | Causa | Risoluzione |
|---|---|---|
| `largest shard is X GB which exceeds the conservative cap of Y GB` | Uno shard eccede il 50% del budget | Liberare memoria, ri-shardare il checkpoint più piccolo (`--shard-size-gb 2`), o `--force-load`. |
| `projected KV cache is X GB but only Y GB available` | `max_position_embeddings` × `max_batch_size` esplode il budget | Edita `config.json`: abbassa uno o entrambi. Niente `--force-load` per questo. |
| `no .safetensors files in <dir>` | Directory sbagliata, o download fallito | Verifica il path; rerun `huggingface-cli download`. |
| `all safetensors files were LFS pointers` | git LFS non pullato | `git lfs pull` o re-download via huggingface-cli. |
| Loader logga "N tensor name(s) were not found … filled with random init" | Checkpoint parziale / prunato | Verifica che il download sia completato; controlla shard `*-of-NNNNN` mancanti. Il modello gira comunque ma produce spazzatura ai layer affetti. |
| Il primo token impiega minuti | Strategia `.streaming` su Mac 16 GB, pagine cold | Atteso al primo turn; i token successivi sono molto più veloci (lo slot rotating resta caldo). Opzionalmente `--warmup-on-load`. |
| "no Metal device" / Mac Intel | Hardware non supportato | Apple Silicon richiesto per l'inference locale. Il remoto (OpenRouter) funziona comunque. |
| 100% memory pressure, sistema che freeza | Buffer mmappati multipli in competizione con la GUI | Forza `.streaming`, abbassa il working set via env var, o riavvia con meno app aperte. |

Il flag `force-load` bypassa **solo** il guard `shardTooLarge`. Il
guard `kvCacheTooLarge` è genuinamente non sconfiggibile (il runtime
alloca `MTLBuffer` `storageModeShared` upfront; non c'è un path
streaming che possa aiutare). Abbassa seq-len / batch-size o
ri-quantizza.
