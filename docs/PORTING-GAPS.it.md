[English](PORTING-GAPS.md) | **Italiano**

# Gap di porting vs. `antirez/ds4` (runtime / Metal)

Questo documento è il piano turnkey per i gap runtime tra il port Swift e il
riferimento C upstream che **non sono validabili senza hardware Apple** (toccano
kernel Metal e il percorso decode/prefill, quindi la parità dei logit va
verificata su Mac). I gap di tooling offline (GGUF writer, requantizzatore
offline) sono già implementati in puro Swift — vedi
`Sources/DS4Core/Formats/GGUF/GGUFWriter.swift` e
`Sources/DS4Core/Formats/Quantization/GGUFRequantizer.swift`.

Fuori scope per scelta (vedi [`UPSTREAM-SYNC.it.md`](UPSTREAM-SYNC.it.md)):
backend CUDA/ROCm, tensor-parallel + RDMA (`ds4_tp.c`), il path speculativo MTP
upstream, `ds4-eval` e l'agente da terminale.

Per ogni gap: riferimento upstream esatto, punti d'innesto Swift, piano a passi
e gate di validazione. I percorsi upstream sono relativi a un clone di
`https://github.com/antirez/ds4.git`.

---

## Gap 1 — Formati quant routed Q8_K e Q8_0

### Stato attuale
`MoEQuant` (`Sources/DS4Metal/Model/Quantization/MoEQuant.swift`) accetta per i
routed experts solo `q4_K (12)`, `q2_K (10)`, `iq2_xxs (16)`, e il loader
**rifiuta con errore** qualunque altro tipo
(`Sources/DS4Metal/Backends/DeepSeekV4/Model/GGUFWeights.swift:63`
`validateRuntimeLayout`). Un GGUF con routed experts Q8_K o Q8_0 non carica.
`get_rows` esiste solo in `f32/f16/i32` (`metal/deepseek/get_rows.metal`).

> Nota priorità: nessun modello in `ModelCatalog` usa oggi routed Q8_K/Q8_0,
> quindi il payoff pratico è basso ora. Farlo se/quando un GGUF simile diventa
> target, o per eliminare la requant on-device `Q8_0→Q4_K` dei pesi densi.

### Riferimento upstream
- `ds4.c` `layer_routed_moe_batch` (~riga 10967): rami `routed_q8_0` e
  `routed_q8_k` — quantizzazione attivazione per-formato + matvec per-esperto.
- `metal/moe.metal`: famiglia `kernel_mul_mv_id_*`; `metal/get_rows.metal`:
  `kernel_get_rows_q8_0_f32`, `kernel_get_rows_q4_K_f32`.
- Layout blocco Q8_K: 256 elem / 292 byte (già in `GGUF.typeTable[15]`); Q8_0:
  32 elem / 34 byte (`typeTable[8]`).

### Punti d'innesto Swift
1. `MoEQuant.swift`: aggiungi `case q8_K` (ed eventualmente `q8_0`) con
   `kernel`, `blockBytes` (292 / 34), `nr0`, `threadgroupBytes` e
   `from(ggufType:)` (15 / 8).
2. `metal/deepseek/moe.metal`: porta `kernel_mul_mv_id_q8_K_f32` (e `_q8_0` se
   serve) dall'upstream. Rigenera i kernel embedded: `make embed-kernels`
   (scrive `Sources/DS4Metal/Runtime/Generated/KernelSources.swift`).
3. `GGUFWeights.swift:validateRuntimeLayout`: allarga l'insieme accettato per
   combaciare con `MoEQuant.from`.
4. Verifica che `GGUFWeights.gatherExperts` dimensioni già dai `blockBytes`
   (lo fa) così non resta nulla hardcodato per tipo.

### Passi
1. Genera un piccolo GGUF fixture con routed Q8_K dal requantizzatore offline:
   `DS4Demo requantize in.gguf q8k.gguf q4_k>q8_k@ffn` (estendi `QuantEncode` se
   un tipo sorgente necessario non è ancora dequantizzabile).
2. Aggiungi il kernel; cabla `MoEQuant`; rilassa il validatore.
3. Aggiungi un test fixture byte-exact del matvec (sulla falsariga di
   `QuantEncodeTests` / `scripts/quant-fixtures/`).

### Gate di validazione
Parità logit del fixture Q8_K vs. il riferimento C entro la tolleranza
esistente; nessuna regressione su decode + prefill batchato `q4_K/q2_K/iq2_xxs`.

---

## Gap 2 — DeepSeek V4 Pro Q4 split-load (pacchetto a due shard)

### Stato attuale
Il Pro Q2 single-file gira; il pacchetto Pro Q4 a due shard è **download-only**
e non selezionabile come modello locale (`docs/ARCHITETTURE-SUPPORTATE.md`; il
percorso GUI/`Browse` rifiuta uno shard isolato). È lavoro di
loader/model-management, in gran parte NON nuovi shader Metal.

> **Progresso:**
> - `GGUFShardSet` (puro Swift, testato): unione per nome + routing per-layer
>   allo shard proprietario (`shard(forLayer:)`, `shard(owning:)`, `primary`).
> - `StreamingDecoder.fromGGUFShards` (DS4Metal, additiva): costruisce il decoder
>   instradando ogni layer al suo shard, così **`GGUFWeights` è riusato
>   invariato** (lo split è per-layer, quindi un layer intero risolve su un solo
>   shard). È il percorso resident semplice (analogo di `fromGGUF`).
>
> - CLI `DS4Demo` (additiva): una lista di path separati da virgola
>   (`DS4Demo shardA.gguf,shardB.gguf …`) apre lo split e lo esegue via
>   `fromGGUFShards` — così l'intera catena è **eseguibile end-to-end** per la
>   validazione on-device senza toccare la GUI.
>
> Resta, da validare on-device: (a) le varianti a streaming esperti per shard (i
> pool esperti mapped/cached attraversano gli shard); (b) la cablatura della GUI
> `InferenceService` (il suo init è costruito su un solo `GGUFModel`; aggiungere
> un percorso shard che lasci invariato il single-file); (c) catalogo/GUI che
> riconoscano i due file come un unico modello selezionabile.
>
> Nota di design: un "protocollo di lettura" ingenuo su `GGUFWeights` è stato
> scartato — il caricamento pesi usa accesso raw per-tensore (`mapBase`,
> `uncachedFD`, `path`) che è per-shard, quindi il routing per-layer (sopra) è il
> seam corretto e a rischio minore; `GGUFWeights` non ha dovuto cambiare.

### Riferimento upstream
- L'upstream assembla GGUF multi-file; vedi la gestione shard/loader nel
  percorso di apertura modello in `ds4.c` e `download_model.sh` (target shard
  Pro Q4).
- `docs/ARCHITETTURE-SUPPORTATE.md` documenta già il confine previsto.

### Punti d'innesto Swift
- `Sources/DS4Core/Formats/GGUF/GGUFShardSet.swift` — **FATTO**: directory
  unificata dei tensori su N mmap (`find`/`tensorData` instradano sullo shard
  proprietario; metadata first-shard-wins).
- `Sources/DS4Metal/Backends/DeepSeekV4/Model/GGUFWeights.swift`: l'accesso ai
  byte deve risolvere sullo shard proprietario — accetta un `GGUFShardSet` (o un
  piccolo protocollo soddisfatto sia da `GGUFModel` che da `GGUFShardSet`)
  invece di un solo `GGUFModel`.
- `Sources/DS4Engine/ModelManagement/Catalog/ModelCatalog.swift`: marca il
  pacchetto Pro Q4 `runnable` una volta pronto l'assemblaggio; lo scan/`Browse`
  della GUI deve riconoscere l'insieme di shard come un unico modello.

### Passi
1. ~~Manifest/naming degli shard~~ — il pacchetto è due GGUF per range di layer
   (`…Layers00-30.gguf`, `…Layers-31-output.gguf`), nomi tensori disgiunti.
2. ~~Apertura multi-shard con indice unificato~~ — fatto in `GGUFShardSet`.
3. Instrada l'accesso ai pesi (`GGUFWeights`, gather esperti) attraverso lo
   shard set — più semplice via un protocollo di lettura implementato sia da
   `GGUFModel` che da `GGUFShardSet` (`findTensor`, `tensorData`, accessori
   metadata).
4. Attiva `runnable` nel catalogo; insegna alla GUI a raggruppare i due file
   Pro Q4 come un unico modello selezionabile.

### Gate di validazione
Il Pro Q4 carica e produce logit uguali al riferimento C; il percorso Pro
distribuito (già testato a livello di protocollo) ottiene la sua validazione
numerica pendente (vedi voce aperta `f2d701a` in `UPSTREAM-SYNC.it.md`).

---

## Gap 3 — Server: batch misto prefill+decode (continuous batching)

### Stato attuale
Il server **serializza** le richieste dietro `RequestGate`
(`Sources/DwarfStar/Features/Server/Concurrency/RequestGate.swift`) su un unico
motore mutabile — una generazione alla volta. L'upstream serve più sessioni
fondendo un batch di prefill con i decode in volo in un unico encode di FFN.

> Gap di valore più alto ma più invasivo: cambia la geometria di batch del
> motore e lo scheduler del server. Farlo per ultimo, dietro un flag env, con il
> percorso serializzato come fallback.

### Riferimento upstream
- `ds4.c` `metal_graph_encode_layer_ffn_batch` (~riga 28412): prende `n_tokens`
  (prefill) **e** `decode_items`/`decode_count`; "solo il dispatch degli esperti
  routed è condiviso tra i due percorsi aritmetici".
- `ds4_server.c`: `server_slot`, `batched_mode`, `active_generations`,
  `server_prefill_quantum`, `server_session_sync` — lo scheduler a slot.
- Knob env: `DS4_CUDA_MIXED_PREFILL_DECODE`, `DS4_CUDA_PREFILL_PIPELINE_*`
  (specifici CUDA; l'analogo Metal è l'encode FFN condiviso sopra).

### Punti d'innesto Swift
- `Sources/DS4Metal/Backends/DeepSeekV4/Decode/Prefill/StreamingDecoder+Prefill.swift`
  (`batchedExpertLayer`, `PrefillStage`): estendi l'FFN batchata per accettare
  una coda di decode nelle righe `[n, n+decodeCount)` condividendo il gather
  degli esperti routed — l'analogo diretto dell'encode C.
- `Sources/DS4Engine/Inference/Service/*`: uno scheduler a slot che ammette
  nuovi prefill su un batch di decode in corso (prefill quantum, KV per sessione).
- `Sources/DwarfStar/Features/Server/*`: sostituisci il `RequestGate`
  single-flight con lo scheduler; mantieni la modalità serializzata come
  default/fallback.

### Passi
1. Prima la primitiva di motore: un layer batchato che esegue `n` righe di
   prefill + `k` righe di decode condividendo un solo gather-union degli
   esperti routed. Unit-test contro la somma dei percorsi separati
   (numericamente identico).
2. Aggiungi KV per-sessione + scheduler a slot nel service del motore.
3. Cabla il server allo scheduler dietro `DS4_SERVER_BATCH` (default off).

### Gate di validazione
Il throughput aggregato cresce con sessioni concorrenti; l'output a sessione
singola è byte-identico al percorso serializzato; isolamento KV tra sessioni
verificato.

---

## Ordine consigliato

1. **Primitiva di motore del Gap 3** (FFN prefill+decode condivisa) — valore più
   alto; parti dal core solo-motore, unit-testabile.
2. **Gap 2** (Pro Q4 split-load) — sblocca un artefatto già distribuito; quasi
   tutto Swift.
3. **Gap 1** (routed Q8_K/Q8_0) — quando un GGUF target lo richiede.

Ogni passo qui richiede un Mac (Metal) per il suo gate di validazione; la CI
Linux di questo repo può costruire/testare solo la superficie puro-Swift
`DS4Core`.

## Dal drift upstream (vedi UPSTREAM-SYNC.it.md, rivisto `80ebbc3..0a7ad77`)

Due voci in-scope aggiuntive emerse rieseguendo il confronto upstream, entrambe
da leggere nel codice su Mac:

- **`519c4d8` — correttezza cache esperti / sampling.** L'upstream rende i logit
  calcolati indipendenti dal budget della cache esperti in streaming SSD
  (+ warning cache troppo piccola e un fix di sampling). Verificare che il
  percorso Swift `ExpertSlotCache`/streaming mantenga lo stesso invariante
  "logit ⟂ budget".
- **`427e281` — resync kernel Metal.** Grande pass di ottimizzazione kernel
  upstream; i kernel del port (derivati da `metal/*.metal`) sono ora indietro.
  Valutare il port delle modifiche e rigenerare i kernel embedded.

Inoltre: `36cd0ca` (session batching Metal nativo) è ora il riferimento Metal per
il Gap 3, e `005afed` (GLM 5.2 + fixture di qualità) supporta la certificazione
di parità GLM.
