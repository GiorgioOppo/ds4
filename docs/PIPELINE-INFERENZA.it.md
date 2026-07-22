[English](PIPELINE-INFERENZA.md) | **Italiano**

# Pipeline di inferenza

Questo documento segue una richiesta dalla GUI o dall'API fino al token
mostrato all'utente. Per i dettagli matematici dei singoli layer vedere
[ARCHITETTURA-MOTORE.md](ARCHITETTURA-MOTORE.it.md); per la collocazione dei file
vedere [STRUTTURA-PROGETTO.md](STRUTTURA-PROGETTO.it.md).

## Vista d'insieme

```text
Chat / HTTP / benchmark
        |
        v
InferenceService (stato, prefill/decode ed emissione eventi)
        |
        +--> ChatRenderer + Tokenizer
        |
        +--> riuso o ripristino del prefisso KV
        |
        +--> prefill del solo suffisso nuovo
        |
        +--> ciclo decode: forward -> logits -> sampler -> token
        |
        +--> eventi testo / reasoning / tool call / metriche
                         |
                         v
        ChatStore+ToolLoop / DistributedController
        esecuzione tool -> risultati -> ripresa inferenza
```

La GUI, il server e il benchmark locale condividono una sola istanza di
`InferenceService`. Questo evita di caricare due copie dei buffer residenti,
della cache Q4 e dello scratch GPU, requisito essenziale sui Mac con 16 GB.

## 1. Caricamento del modello

1. `GGUFModel` apre il GGUF e valida header, metadata e descrittori tensoriali.
2. `ModelConfig` e `DSV4Shape` riconoscono il profilo supportato e rifiutano
   forme incompatibili prima di avviare il decode.
3. `Tokenizer` legge vocabolario, merge e token di controllo dal GGUF.
4. `GGUFWeights` prepara le viste mappate dei pesi e i provider dei layer.
5. `StreamingDecoder` sceglie il percorso di memoria richiesto: pesi densi
   residenti o in streaming, cache esperti, bundle, `pread` o MetalIO.
6. `InferenceService` pubblica avanzamento e log e rende il motore disponibile
   a chat, server e benchmark.

I toggle che cambiano layout o residenza dei pesi vengono letti al caricamento:
dopo una modifica occorre ricaricare il modello. Le eccezioni aggiornabili tra
due prefill sono documentate nella
[Configuration Reference](../README.it.md#riferimento-di-configurazione).

## 2. Preparazione della conversazione

`ChatRenderer` trasforma sistema, turni, specifiche dei tool e risultati in
DSML tramite un'implementazione Swift costruita sul template DeepSeek-V4 di
riferimento. Il runtime non interpreta il Jinja incorporato nel GGUF: quel
metadato è disponibile per diagnostica, mentre la modalità tool compatta
predefinita abbrevia intenzionalmente la dichiarazione per ridurre il prefill.
`Tokenizer` converte il testo renderizzato in token id. I tipi condivisi
(`ChatTurn`, `ToolSpec` e `ToolCall`) vivono in `DS4Core`, quindi il servizio non
introduce un secondo formato di conversazione.

Prima del prefill il servizio confronta gli id renderizzati con quelli già
committati:

- se il nuovo prompt estende esattamente il prefisso corrente, elabora solo il
  suffisso;
- se esiste un checkpoint compatibile, `DiskKVStore` lo ripristina e prosegue
  dal prefisso salvato;
- dopo uno stop o una divergenza ricostruisce lo stato dal prefisso sicuro.

La ricorrenza del compressore NSA non può essere semplicemente riavvolta. Per
questo lo stato visibile della conversazione e lo stato KV committato vengono
tenuti distinti fino alla conclusione pulita del turno.

## 3. Prefill

Il prefill elabora molti token già noti. Il percorso è organizzato per layer e
per chunk:

1. lo stage di route/attention prepara un gruppo di token;
2. i pesi densi del layer sono caricati una volta per il chunk;
3. gli esperti richiesti dai token vengono uniti in gruppi limitati da
   `DS4_PREFILL_UNION`;
4. la FFN applica gli esperti ai token interessati;
5. KV raw, cache compressa e stato ricorrente avanzano in ordine.

`DS4_PREFILL_CHUNK`, `DS4_PREFILL_UNION` e `DS4_PREFILL_ROUTE_BATCH`
controllano rispettivamente ammortamento dei pesi densi, memoria transitoria e
numero di sincronizzazioni. Il prefill non è un decode ripetuto in un ciclo:
usa strutture dedicate sotto
`DS4Metal/Backends/DeepSeekV4/Decode/Prefill`.

Con `DS4_PREFILL_BATCH_ATTN` (default attivo) la fase route/attention esegue
UNA FlashAttention multi-query per run di route-batch — il kernel di prefill
del motore C (8 righe di query per threadgroup, MMA simdgroup) sull'unione
dello span KV, con una maschera per-query che riproduce esattamente la
visibilità per-token (causale + finestra SWA + righe compresse emesse fino a
ciascun token) — invece di un dispatch vec a query singola per token. Stessa
matematica sulle stesse chiavi visibili; cambia solo l'ordine di accumulo,
quindi gli output sono vicini ma non bit-identici al percorso per-token (`=0`
lo ripristina per parità A/B). I run che sforerebbero la cache raw
`DS4_RAW_RING` ricadono automaticamente sul percorso per-token.

In aggiunta, `DS4_PREFILL_DENSE_MM` (default attivo) trasforma OGNI proiezione
densa del run batchato — q_a, q_b, kv, l'output low-rank a gruppi, output_b,
il router — più le due riduzioni HyperConnection in kernel matrice-matrice
sull'intero run: i pesi densi si leggono una volta per route-batch invece che
una volta per token, cioè la forma layer-major del prefill del motore C. Il
lavoro per-token si riduce all'aggiornamento di stato del compressore NSA,
allo store fp8 della riga KV e alla finalize top-6 del router (anche le
proiezioni kv/score del compressore sono batchate). Ogni proiezione dispatcha
sul quant RESIDENTE — Q8_0/F16 di default, Q4_K (`kernel_mul_mm_q4_K_f32`)
quando `DS4_DENSE_Q4`/`DS4_QKV_Q4` l'ha riquantizzata — quindi il prefill
batchato resta attivo anche nel profilo residente-Q4 orientato al decode.
`=0` ripristina il denso per-token per A/B. I contatori di profilo
`prefillFlashRuns`/`prefillDenseRuns` riportano quanti run hanno preso ciascun
percorso.

Con `DS4_PREFILL_FULL_LAYER` (default attivo) e un chunk di almeno
`DS4_PREFILL_FULL_LAYER_MIN` token (512), la fase B streama il layer routed
INTERO una sola volta — tutti gli esperti, id GLOBALI, niente unioni né
remap — e mette in double-buffer lo slab del layer successivo durante le FFN
di quello corrente (la forma di prefill del motore C; ~2×1.7 GiB transienti,
rilasciati a fine prefill). I byte/token diventano byteLayer/tokenChunk:
misurati 26.8 MB/token con chunk 4096 contro ~155 del percorso a unioni, con
il gather esposto che scende a ~0. Anche la coda routed dei percorsi MM è
batchata (collasso a pesi unitari + add a righe + un HC expand per run,
bit-identica alla coda per-token). Miglior env di prefill misurato sul Flash
2-bit: `DS4_PREFILL_ROUTE_BATCH=128 DS4_PREFILL_CHUNK=4096 DS4_PREFILL_MM=1
DS4_EXPERT_PREAD=1 DS4_PREAD_SPLIT=4` — 26 tok/s su un prompt reale da 2.7k
token (M1 Pro), dai 3.6 di partenza.

## 4. Decode di un token

Per ogni token il decoder esegue tutti i layer in ordine:

1. embedding e stato HC iniziale;
2. normalizzazione e proiezioni Q/KV;
3. compressore e attenzione NSA;
4. router MoE o tabella hash dei primi layer;
5. gather dei sei esperti routed selezionati;
6. FFN routed e shared, riduzione HC e passaggio al layer successivo;
7. norm finale e output head;
8. sampling e detokenizzazione.

`StreamingDecoder+LayerExecution.swift` orchestra il layer, mentre `Graph/*`
compone operazioni GPU più piccole. I wrapper sotto `Kernels/*` devono restare
privi di politica applicativa.

## 5. Streaming degli esperti

Il modello MoE completo non deve risiedere in RAM. Per ogni layer vengono letti
solo gli esperti scelti. Le strategie possono essere combinate:

- slot-cache LRU pre-riscaldata dalla usage imatrix;
- letture dirette `pread` con `F_NOCACHE`;
- bundle sidecar con gate/up/down contigui per esperto;
- MetalIO con circuit breaker e fallback a `pread`;
- prefetch esatto o guidato dallo storico;
- pesi densi in streaming con ring di staging indipendente.

La cache accelera il gather ma non cambia router, id o pesi degli esperti. I
percorsi lossy sono dichiarati esplicitamente nella configurazione (`DENSE_Q4`,
`QKV_Q4`, `SHARED_Q4`, `COMP_Q8`, riduzione degli esperti attivi).

## 6. Sampling, reasoning e tool

`Sampler` applica temperature, top-k, top-p, min-p e repetition penalty. Il
token ottenuto viene convertito in byte senza assumere che ogni token sia una
stringa UTF-8 completa.

Gli eventi del servizio distinguono:

- testo visibile;
- contenuto di reasoning;
- stream e completamento di una chiamata tool;
- avanzamento e metriche;
- completamento, stop ed errore.

Quando il parser DSML riconosce una chiamata, `InferenceService` la emette come
evento `.toolCall`: non esegue il tool. Nel percorso locale il loop è
orchestrato da `ChatStore+ToolLoop.swift`; nel percorso distribuito da
`DistributedController`. Questi consumer invocano `ToolRegistry.executeAuto`,
raccolgono i risultati e avviano la continuazione. Il percorso locale li passa
a `InferenceService.provideToolResults`; quello distribuito aggiunge i turni
`toolResult` e richiama la generazione sul coordinator.

## 7. Proprietà dello stato

| Stato | Proprietario |
|---|---|
| Conversazioni, selezione agente e impostazioni UI | `ChatStore` |
| Loop tool locale/distribuito | `ChatStore+ToolLoop` / `DistributedController` |
| Decoder condiviso, token committati e generazione attiva | `InferenceService` |
| KV, scratch, cache esperti e profilazione GPU | `StreamingDecoder` |
| Checkpoint persistenti | `DiskKVStore` |
| Rendering DSML e tipi di conversazione | `DS4Core/Conversation` |
| Sampling | `DS4Core/Generation` |

Regola pratica: la GUI non deve mutare direttamente buffer o KV; il backend
Metal non deve conoscere sessioni, viste o protocollo HTTP.

## 8. Mappa del codice

- `Sources/DS4Core/Conversation` — modelli, rendering e parsing DSML.
- `Sources/DS4Core/Tokenization` — BPE e detokenizzazione byte-level.
- `Sources/DS4Core/Generation` — sampler.
- `Sources/DS4Metal/Backends/DeepSeekV4/Decode` — prefill, decode, KV e cache
  del backend attualmente operativo.
- `Sources/DS4Metal/Backends/DeepSeekV4` — forma Metal, pesi, esperti e
  streaming DeepSeek; runtime e kernel condivisi restano fuori dal backend.
- `Sources/DS4Engine/Inference` — API ed actor applicativo.
- `Sources/DS4Engine/Persistence/KV` — checkpoint su disco.
- `Sources/DwarfStar/Features/Chat` — stato e presentazione della chat.

## 9. Verifica delle modifiche

Una modifica alla pipeline richiede, in proporzione al livello toccato:

1. test puri per tokenizer, renderer, sampler o protocollo;
2. test di parità per wrapper e grafo Metal;
3. build della demo Release;
4. confronto di prompt, seed e configurazione identici;
5. misura separata di prefill e decode, dopo il warm-up.

La procedura completa è in [TESTING-E-VALIDAZIONE.md](TESTING-E-VALIDAZIONE.it.md).
