# Backend DeepSeek V4 — architettura del motore

Questo documento descrive il **backend DeepSeek V4** operativo alla base di
DwarfStar: come viene aperto il suo GGUF, come il testo diventa token DeepSeek,
come i token attraversano il suo grafo di decode Metal, come gli esperti routed
vengono letti in streaming dall'SSD e come il livello di servizio ritrasforma i
logits in uno stream di chat. Il rilevamento dell'architettura e le regole
condivise con i backend futuri sono documentati separatamente in
[`ARCHITETTURE-SUPPORTATE.md`](ARCHITETTURE-SUPPORTATE.md).

L'obiettivo del port è la fedeltà comportamentale agli upstream `ds4.c` /
`ds4_metal.m`, integrandosi al tempo stesso in modo pulito con Swift, SwiftUI,
gli actor, `Network.framework` e la sandbox di macOS.

Collegamenti incrociati:

- [`DOCUMENTAZIONE.md`](DOCUMENTAZIONE.md) — uso dell'app, pannelli, flussi di
  lavoro.
- [`ARCHITETTURE-SUPPORTATE.md`](ARCHITETTURE-SUPPORTATE.md) — rilevamento
  della famiglia di modelli, confini dei backend e matrice di supporto corrente.
- [`PIPELINE-INFERENZA.md`](PIPELINE-INFERENZA.md) — ciclo di vita della
  richiesta, proprietà dello stato, prefill e decode.
- [`BACKEND-METAL.md`](BACKEND-METAL.md) — runtime, wrapper, kernel generati e
  regole di validazione numerica.
- [`INFERENZA-DISTRIBUITA.md`](INFERENZA-DISTRIBUITA.md) — topologie
  orizzontali e verticali e protocollo v11.
- [`../README.md`](../README.md) — panoramica del repository e comandi di build.
- [`../README.md#configuration-reference`](../README.md#configuration-reference)
  — l'elenco autorevole di ogni knob `DS4_*` citato più avanti, con default e
  indicazioni di tuning.
- [`../Sources/DS4Demo/README.md`](../Sources/DS4Demo/README.md) — demo CLI e
  knob di runtime.

## Indice

1. [Pipeline end-to-end](#1-pipeline-end-to-end)
2. [Caricamento del modello: GGUF](#2-caricamento-del-modello-gguf)
3. [Tokenizer](#3-tokenizer)
4. [Forma del modello e validazione](#4-forma-del-modello-e-validazione)
5. [Substrato di esecuzione Metal](#5-substrato-di-esecuzione-metal)
6. [Decoder: forward pass completo](#6-decoder-forward-pass-completo)
7. [Dettagli del layer di decode](#7-dettagli-del-layer-di-decode)
8. [Compressore NSA](#8-compressore-nsa)
9. [MoE: router ed esperti](#9-moe-router-ed-esperti)
10. [StreamingDecoder e strategie di streaming](#10-streamingdecoder-e-strategie-di-streaming)
11. [Campionamento: dai logits al token](#11-campionamento-dai-logits-al-token)
12. [Quantizzazione](#12-quantizzazione)
13. [Riepilogo dei tensori per layer](#13-riepilogo-dei-tensori-per-layer)
14. [Tool calling](#14-tool-calling)
15. [Inferenza distribuita](#15-inferenza-distribuita)
16. [Riferimento incrociato da C a Swift](#16-riferimento-incrociato-da-c-a-swift)

## 1. Pipeline end-to-end

```text
Testo utente
  -> ChatRenderer / template chat del tokenizer
  -> token id
  -> embedding
  -> StreamingDecoder
       layer 0
       layer 1
       ...
       layer N-1
  -> norm di output + output head
  -> logits
  -> Sampler
  -> token id
  -> byte detokenizzati
  -> testo visibile / reasoning / tool call DSML
```

A livello applicativo, `InferenceService` possiede il decoder ed espone stream
asincroni. Il servizio tiene traccia degli esatti token id committati nella
conversazione. Quando l'utente invia un nuovo turno, viene passato al prefill
solo il suffisso non già presente nella KV. Quando vengono chiamati i tool, i
loro risultati sono accodati come turni tool-result e il decode continua dallo
stesso stato append-only.

La demo CLI usa lo stesso percorso di basso livello senza stato chat: apre un
GGUF, costruisce uno `StreamingDecoder`, esegue opzionalmente il prefill di un
prompt e poi decodifica i token in modalità greedy.

## 2. Caricamento del modello: GGUF

`Sources/DS4Core/Formats/GGUF/` contiene i tipi GGUF, il cursore binario e il
modello mappato usati per interpretare metadata e descrittori tensoriali. Il
file del modello viene aperto con mmap, così i dati dei tensori possono essere
referenziati senza copie.

Responsabilità principali:

- interpretare header e versione GGUF;
- interpretare i metadata chiave/valore;
- interpretare i metadata dei tensori: nome, forma, tipo, offset, dimensione in
  byte;
- esporre lookup tipizzati per nome del tensore;
- supportare il mapping Metal senza copie e il gather degli slab degli esperti;
- fornire le tabelle del tokenizer dai metadata.

Il loader è volutamente conservativo. Se il file non contiene i metadata o i
nomi di tensore attesi, il fallimento deve avvenire prima del costoso lavoro di
decode. È per questo che la modalità di audit della CLI (`DS4_TYPES_ONLY=1`)
stampa dtype tensoriali rappresentativi, gli id dei token speciali e gli id del
prompt.

### Caricamento dei pesi in `GPUTensor`

Esistono due grandi categorie di pesi:

| Categoria | Trattamento a runtime |
|---|---|
| Pesi densi non routed | Viste mmap senza copie per default, opzionalmente in streaming attraverso un ring di staging `pread + F_NOCACHE` con `DS4_DENSE_STREAM=1`, oppure copiati in buffer GPU residenti con `DS4_RESIDENT_DENSE=1` sui sistemi ricchi di RAM. |
| Pesi degli esperti routed | Raccolti via gather per token/layer, opzionalmente messi in cache negli slot esperti, opzionalmente letti via `pread + F_NOCACHE`, opzionalmente serviti da un sidecar expert-bundle contiguo. |

`GGUFWeights` costruisce valori `LayerWeights` che referenziano i tensori
richiesti da ciascun layer. I tensori degli esperti possono restare non
caricati come tensori completi pur rimanendo disponibili per il gather, perché
il percorso di gather legge gli slab degli esperti selezionati dal tensore GGUF
mappato.

I layer di esperti routed a precisione mista sono supportati memorizzando la
quantizzazione su `LayerWeights` (`gateQuant`, `upQuant`, `downQuant`) invece
di assumere un'unica quantizzazione globale degli esperti per ogni layer.

## 3. Tokenizer

`Sources/DS4Core/Tokenization/Backends/DeepSeekV4/DeepSeekV4Tokenizer.swift` è
il tokenizer in puro Swift usato dall'attuale app DeepSeek, dalla demo, dal
server, dalla diagnostica e dai test. Il nome pubblico storico `Tokenizer`
resta un alias per compatibilità dei sorgenti; non è l'API di tokenizer
neutrale rispetto all'architettura.

### Componenti

- vocabolario e byte dei token caricati dai metadata GGUF;
- dati di merge/rank per il comportamento BPE;
- id dei token speciali come BOS, EOS, user, assistant, marcatori di thinking e
  DSML;
- gestione dei byte di fallback;
- helper di rendering del prompt chat allineati manualmente al template di
  riferimento; il testo Jinja del GGUF è esposto per diagnostica ma non viene
  interpretato a runtime.

### API pubblica

Chiamate tipiche:

- `Tokenizer(model:)` costruisce a partire da un `GGUFModel`;
- `encodeChatPrompt(system:prompt:think:)` renderizza un prompt chat semplice;
- `tokenizeRenderedChat(_:)` tokenizza testo di template già renderizzato;
- `tokenText(_:)` restituisce i byte del token per la detokenizzazione.

Il pannello di diagnostica apre il GGUF solo per i metadata del tokenizer e
stampa lo stesso percorso di tokenizzazione nativo usato dall'inferenza. Non
rimane alcun tokenizer in sottoprocesso.

## 4. Forma del modello e validazione

`DS4Core/Model/Backends/DeepSeekV4/DeepSeekV4Configuration.swift` descrive e
valida entrambi i profili noti. Dopo la validazione,
`DS4Metal/Backends/DeepSeekV4/Architecture/DSV4RuntimeGeometry.swift` trasforma
il profilo selezionato e i suoi metadata per layer in dimensioni immutabili
possedute dall'istanza. `DSV4Shape.swift` resta solo come facciata Flash
compatibile a livello di sorgenti per i chiamanti legacy; la costruzione del
decoder consapevole del modello non lo usa per sovrascrivere un file Pro.

`ModelConfig(model:)` valida i metadata come il loader C
(`config_validate_model`): ogni campo che definisce la forma (numero di layer,
dimensioni delle teste, rank LoRA, numero di esperti, numero di layer hash,
geometria dell'indexer, conteggi HC) deve corrispondere esattamente a un
profilo noto. I metadata a livello di motore vengono verificati in modo
incrociato con il profilo selezionato — i rapporti di compressione per layer
contro la formula attesa, i clamp SwiGLU per layer, i parametri di scaling
RoPE, `expert_weights_scale`, `expert_weights_norm` e gli epsilon RMS/HC
(entrambi `1.0e-6`, come nel riferimento C). Una discrepanza rifiuta il file al
caricamento invece di decodificare silenziosamente con una matematica diversa.

La selezione della forma distingue i profili Flash e Pro per corrispondenza
esatta. Flash usa 43 layer, larghezza 4096, 64 teste e 256 esperti; Pro usa 61
layer, larghezza 7168, 128 teste e 384 esperti. Il servizio locale e la CLI
associano una `DSV4RuntimeGeometry` prima di costruire pesi, scratch, KV, cache
e decoder. Il router accetta 256 esperti oppure un dispatch bitonico a 512 lane
con padding oltre i 384 esperti reali di Pro, e applica la scala del profilo
(1.5 o 2.5). Questo è il percorso locale supportato per il GGUF Pro Q2 a file
singolo. Il pacchetto Pro Q4 resta solo scaricabile perché è composto da due
shard, mentre l'esecuzione distribuita di Pro resta in fase di verifica.

Anche la validazione a livello di tensore avviene al caricamento: la tabella di
hash-routing (`ffn_gate_tid2eid.weight`) è obbligatoria sui layer hash-routed
con verifica del suo layout, il layout del bias del router viene validato e
`GGUFWeights.validateRoutedExperts` verifica i tensori degli esperti routed
contro le classi di quantizzazione rilevate (i layer a precisione mista vengono
contati e decodificati per layer, bypassando la cache esperti a classe singola).

Il livello di forma esiste per tenere le costanti centralizzate senza
trasformarle in stato globale di runtime. Il codice del decoder legge le
dimensioni dalla propria geometria o da `DSV4Dims`; aggiungere un fallback
statico Flash in un percorso consapevole del modello è una regressione.

## 5. Substrato di esecuzione Metal

### `MetalRuntime`

`DS4Metal/Runtime/Core/MetalRuntime.swift` possiede:

- il `MTLDevice` selezionato;
- la creazione della command queue;
- la compilazione dei sorgenti kernel incorporati;
- il lookup delle funzioni per nome del kernel;
- piccoli self-test di runtime.

I kernel sono incorporati in `KernelSources.swift`, generato da `metal/*.metal`.
Questo rende l'app pacchettizzata indipendente da una directory di sorgenti
kernel a runtime.

### `GPUTensor`

`GPUTensor` è l'oggetto dati di base del motore:

- metadata di forma e numero di elementi;
- proprietà del buffer Metal o vista senza copia;
- dimensione in byte e interpretazione del dtype;
- costruzione helper per range GGUF mappati e scratch allocato.

Swift ARC possiede il ciclo di vita dell'oggetto. Questo elimina dal port le
classi di bug legate alla liberazione degli handle bridged C/ObjC.

### `GraphContext`

`GraphContext` incapsula una sequenza di command buffer e lo stato transiente
usato dai dispatch dei kernel. Nel percorso di streaming, i commit di
route/attention attendono il completamento — la CPU deve rileggere gli id degli
esperti selezionati — il che evita anche race in cui uno slot esperto viene
sfrattato mentre un command buffer precedente lo sta ancora usando.

Il command buffer della FFN routed è l'eccezione: con `DS4_ASYNC_FFN` (default
attivo) viene committato *senza* attesa CPU (`commitAsync`). Il commit+wait del
route del layer successivo arriva sulla stessa coda in-order, quindi la
correttezza è garantita dall'ordine di coda mentre l'encode CPU del layer i+1
si sovrappone all'esecuzione GPU della FFN del layer i — la bolla di encode per
layer (43 layer Flash o 61 layer Pro) scompare. Il buffer in volo viene
esplicitamente drenato a fine token (prima della readback dell'output head /
`readHC` / export KV) e su ogni percorso d'errore, incluso quello del prefill.
`DS4_ASYNC_FFN=0` ripristina il commit sincrono.

`DS4_PROFILE_ROUTE=1` aggiunge deliberatamente confini di fase extra per
separare i tempi di `route/attn`, e mantiene l'attesa sincrona della FFN così
che l'attribuzione per fase resti accurata. Quei tempi sono diagnostici: i
rapporti sono utili, il throughput assoluto non è rappresentativo perché viene
introdotta sincronizzazione extra.

## 6. Decoder: forward pass completo

Ad alto livello, il forward pass di un token esegue:

1. embedding del token;
2. inizializzazione o aggiornamento dello stato Hyper-Connection;
3. per ciascun layer:
   - pre-lavoro di route / attention / compressore;
   - gather degli esperti selezionati o hit nella cache esperti;
   - esecuzione della FFN condivisa e del MoE routed;
   - applicazione del residuo e aggiornamento HC;
4. normalizzazione di output;
5. proiezione dell'output head;
6. restituzione dei logits.

`StreamingDecoder.forward(token:pos:nKeys:)` guida tutto questo per il decode.
`StreamingDecoder.prefill(tokens:)` usa un'implementazione layer-major per i
token del prompt, così da ammortizzare l'I/O dei pesi.

### Hyper-Connection (HC)

DeepSeek-V4 usa lo stato Hyper-Connection in aggiunta al normale flusso
residuale. DwarfStar trasporta i tensori HC attraverso il grafo e attraverso le
fette dei worker distribuiti. In modalità distribuita, il coordinatore esegue
l'embedding dei token, poi i worker fanno passare lo stato HC attraverso i
propri intervalli di layer e restituiscono stato finale/logits secondo
necessità.

## 7. Dettagli del layer di decode

Ogni layer è rappresentato da `DecodeLayer` e dagli helper di grafo associati.

### Fase 1 — `decodeRoute`

Questa fase copre il lavoro che porta al routing:

- normalizzazione dell'attention;
- proiezioni Q/KV;
- rotary position embedding;
- aggiornamento del compressore NSA;
- selezione indexer / sparse attention;
- attention MLA;
- normalizzazione pre-FFN;
- logits del router e id dei top-k esperti.

### Attention MLA

Il percorso di attention è la variante multi-head latent attention di DeepSeek.
Il port Swift separa proiezioni dense, normalizzazione, RoPE, KV compressa,
selezione sparsa e output finale dell'attention. I kernel Metal sono wrapper
sottili attorno alle stesse operazioni attese dal grafo upstream.

### Fase 2 — `decodeExperts`

Questa fase esegue:

- ramo FFN condiviso;
- proiezioni gate/up/down del MoE routed;
- pair-SwiGLU fusa dove disponibile;
- down-sum sugli esperti attivi;
- aggiornamento del residuo.

### HC-Reduce fusa

La coda di riduzione Hyper-Connection (split + collapse + RMSNorm) viene
eseguita come UN solo dispatch fuso invece di tre (`DS4_FUSED_HC`, default
attivo). Viene eseguita due volte per layer, quindi la fusione rimuove circa
170 dispatch per token. La matematica è invariata; cambia solo l'ordine di
riduzione della RMSNorm (classe ±1 ulp). `DS4_FUSED_HC=0` ripristina il
percorso non fuso a tre dispatch per i confronti A/B.

Gli esperti selezionati sono la parte costosa dello streaming da SSD. Possono
provenire da:

- gather diretto via mmap;
- `pread + F_NOCACHE`;
- slab contigui dell'expert-bundle;
- hit nello slot-cache degli esperti;
- riempimento dello slot-cache su miss.

## 8. Compressore NSA

Il compressore NSA è ricorrente. Questo conta per la gestione dello stato: una
sequenza parzialmente generata non può essere riavvolta arbitrariamente, a meno
di ripristinare l'esatto stato KV/compressore o di ricostruirlo dagli id
committati.

Il percorso del compressore mantiene le righe compresse e supporta la selezione
di sparse attention usata dal decode. La porzione raw a finestra scorrevole può
opzionalmente essere memorizzata come ring via `DS4_RAW_RING=1`, perché
l'attention legge solo le ultime `nSWA` righe raw. Questo ring resta un
`MTLBuffer` nella memoria condivisa di Apple Silicon; non fa streaming della KV
dall'SSD e non va confuso con lo store dei checkpoint Disk-KV. Quando la
finestra cronologica va a capo, un kernel Metal 2D dedicato riordina le sue
righe e converte F32→F16 in un solo dispatch invece di dividere la finestra in
due dispatch di copia.

La FlashAttention del decode dimensiona lo split-K adattivo a partire da ogni
riga visibile (raw più compresse):

```text
nwg = min(32, max(1, ceil((nRaw + nCompressed) / 32)))
```

Tutte le profondità di workgroup da 1 a 32 sono valide. Non c'è arrotondamento
a potenze di due, quindi la prima riga compressa dopo una finestra raw piena di
128 righe seleziona 5 workgroup per 129 righe totali invece di saltare da 4
a 8. Impostare `DS4_ADAPTIVE_SPLITK=0` ripristina la profondità fissa storica
di 32 per i test A/B.

Con `DS4_DENSE_STREAM=1` le quattro proiezioni del compressore NSA vengono
deviate fuori dal ring di staging e mantenute RESIDENTI (`DS4_RESIDENT_COMP`,
default attivo; la cifra di ~0.6 GB vale per Flash): vengono lette a ogni token
su 41 dei 43 layer Flash e su tutti i 61 layer Pro, la singola rilettura
ripetuta più densa dello stream denso. Stessi byte, numerica identica;
`DS4_RESIDENT_COMP=0` ripristina lo streaming completo come ripiego per RAM
limitata.

### Indexer NSA

L'indexer seleziona le posizioni compresse rilevanti usate dall'attention.
Flash usa top-512 e Pro top-1024, entrambi forniti dalla geometria di runtime.
Indexer e compressore sono parte del motivo per cui lo stato KV non è un
semplice array append-only troncabile senza cautele.

In decode, l'attention resta DENSA su tutte le righe compresse finché il loro
numero non supera una soglia sparse, rispecchiando il motore C
(`metal_graph_decode_indexer_sparse_threshold`): attorno alla frontiera dei ~2K
il setup score/top-k del percorso sparse domina la scansione di attention più
piccola. Il default è 1024, sovrascrivibile con
`DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` (valori ammessi 64…4096, come nel
C). L'attivazione viene verificata in modo prospettico per layer: il conteggio
prospettico delle righe compresse deve superare sia la soglia sparse sia il
top-K dell'indexer.

La selezione top-K in sé (`IndexerSelect.swift`) è basata su heap — un min-heap
binario dei k migliori indici, O(n log k) invece di un ordinamento completo —
perché, una volta attiva, viene eseguita per ogni layer ratio-4 per token.

Due conseguenze della regola di attivazione sulla gestione delle risorse:

- **Staging pigro dello scoring dell'indexer** (`DS4_LAZY_IDX`, default attivo,
  richiede `DS4_DENSE_STREAM=1`) segue il contesto **usato**, non il `maxKeys`
  configurato. Finché il conteggio vivo delle righe compresse ratio-4 non può
  superare sia la soglia sparse sia il top-K, le proiezioni di *scoring*
  dell'indexer (`indexer.attn_q_b` + `indexer.proj`) sono assenti dal piano di
  staging per token. Con i default (soglia 1024, top-K 512), le chiavi vive
  4096 e 4099 restano dense; la chiave 4100 è il primo confine che può
  richiedere lo scoring. A quel confine le proiezioni dello scorer vengono
  caricate una volta in buffer Metal residenti e riusate per il resto della
  sessione, invece di aggiungere circa 360 MB di letture SSD a ogni token
  precedente. La coppia del *compressore* dell'indexer continua a girare dal
  primo token perché il suo stato ricorrente deve restare coerente. È lossless:
  cambiano solo il momento del caricamento e la posizione di memorizzazione.
  `=0` ripristina il comportamento always-stage per l'A/B.
- **Scratch di attention high-water**: parte dalle righe necessarie alla
  sequenza viva e cresce geometricamente solo quando viene raggiunto un nuovo
  massimo storico. Il numero di righe richiesto è la finestra scorrevole raw
  corrente (`min(liveKeys, nSWA)`) più le righe compresse ratio-4 prospettiche
  e un piccolo margine di sicurezza; la crescita è limitata dal massimo
  configurato. Alzare il limite di contesto quindi non impegna più lo scratch
  massimo di attention/indexer alla prima domanda. I buffer esistenti vengono
  mantenuti tra i token, così il decode a regime non alloca a ogni passo.
- **Controllo dei limiti al ripristino dei checkpoint**: il ripristino di un
  checkpoint KV/compressore valida ogni lunghezza proveniente dal file contro
  le capacità dei tensori vivi *e* contro il programma di emissione
  (`count <= maxKeys / ratio`, non l'allocazione, che porta con sé righe di
  slack) prima di qualsiasi memcpy — un checkpoint corrotto non può mai
  ripristinare un numero di righe che nessuna esecuzione legittima può
  raggiungere.

## 9. MoE: router ed esperti

Il router sceglie gli esperti attivi per token e layer. Flash e Pro usano
entrambi 6 esperti attivi. I tensori degli esperti routed dominano la
dimensione del modello e l'I/O su SSD.

### Hash routing e bias di selezione

Due dettagli di routing seguono esattamente il riferimento C:

- I primi `nHashLayer` (3) layer non fanno affatto routing dalle probabilità:
  fanno routing per TOKEN ID attraverso la tabella `ffn_gate_tid2eid.weight`
  (I32, `[6 x nVocab]`; il kernel limita il token a `rows - 1`). La tabella è
  OBBLIGATORIA al caricamento su quei layer, con layout validato. Poiché la
  selezione dipende solo dal token id, l'I/O degli esperti di questi layer può
  sempre essere risolto in anticipo (vedi il look-ahead dello slot-cache più
  sotto).
- `exp_probs_b.bias` (F32 `[nExperts]`, opzionale per layer) sposta le
  probabilità del router SOLO per la SELEZIONE — i pesi di routing applicati
  agli output degli esperti sono calcolati dalle probabilità non distorte, come
  in `layer_topk_selected_experts_from_probs`.

Il percorso di finalizzazione del router riceve `nExperts` ed
`expertWeightScale` dalla geometria attiva. Flash lancia un ordinamento
bitonico a 256 thread con scala 1.5; Pro lancia 512 thread, riempie le lane
384...511 con infinito negativo e usa scala 2.5. Entrambi producono lo stesso
contratto top-6 senza leggere oltre la riga delle probabilità.

### Kernel MoE fusi

I kernel fusi riducono l'overhead di dispatch e la memoria intermedia:

- fusione pair-SwiGLU;
- proiezione down più somma sugli esperti attivi;
- formati esperti quantizzati q4_K, q2_K, iq2_xxs.

`DS4_FUSED_MOE=0` disabilita questo percorso per debugging e confronti A/B. Non
è una normale impostazione di prestazioni e può cambiare gli arrotondamenti.

### Esperti attivi configurabili

`DS4_ACTIVE_EXPERTS=1...6` cambia `DSV4Dims.activeExperts`. Questo cambia
intenzionalmente calcolo e output. È utile per esperimenti a bassa RAM o per
profilare il costo dell'I/O degli esperti, non come ottimizzazione che preserva
la qualità.

## 10. StreamingDecoder e strategie di streaming

`StreamingDecoder` è il motore di decode concreto usato sia dalla CLI sia
dall'app. Possiede i pesi dei layer, lo stato KV, le statistiche di utilizzo, i
contatori di profilo e la cache esperti opzionale.

### Prefill layer-major

`prefill(tokens:chunk:)` elabora i token del prompt in chunk di
`DS4_PREFILL_CHUNK` token (default 512). Per ogni chunk, carica ciascun layer
una volta e lo applica a tutti i token del chunk. È l'opposto del prefill
token-major ingenuo, in cui ogni token del prompt ricaricherebbe gli stessi
pesi del layer; il prefill layer-major è essenziale per i prompt lunghi. Un
chunk più grande ammortizza la ricarica densa per chunk su più token, al costo
di memoria transiente per le attivazioni.

All'interno di un layer il lavoro è diviso in due fasi (`batchedExpertLayer`):

- **Fase A — route.** L'attention è causale all'interno del layer (il token j
  fa attention sulla KV scritta dai token 0..j), quindi le route restano
  sequenziali per token — ma non richiedono più un round-trip CPU ciascuna:
  sequenze fino a `DS4_PREFILL_ROUTE_BATCH` token (default 32) vengono
  codificate in UN solo command buffer, gli input FFN e la selezione del router
  di ciascun token vengono copiati via blit lato GPU in un'area di staging, e
  la CPU legge tutte le selezioni dopo una singola attesa. I token con indexer
  attivo (che richiedono un top-k CPU a metà route) e `DS4_PROFILE_ROUTE`
  ripiegano sul percorso per token.
- **Fase B — FFN degli esperti.** I token vengono raggruppati in modo che
  l'UNIONE degli esperti selezionati di ciascun gruppo resti sotto
  `DS4_PREFILL_UNION` (default 192, mai sotto il numero di esperti attivi);
  l'unione viene raccolta UNA volta sola e la FFN di ogni token viene eseguita
  su di essa con id rimappati. Con `DS4_PREFILL_FFN_BATCH` (default attivo)
  tutte le FFN dei token di un gruppo vengono codificate in UN solo command
  buffer — un commit+wait per gruppo invece di uno per token, il che rimuove
  decine di migliaia di sync GPU per chunk. La codifica seriale mantiene
  identici l'ordine di dispatch e la numerica; `=0` ripristina il percorso per
  token per i controlli di parità A/B.

`DS4_PREFILL_MM=1` (opt-in) esegue in aggiunta la FFN routed di ogni gruppo
attraverso kernel matrice-matrice `mul_mm_id` in batch invece dei matvec per
token, così i pesi degli esperti vengono letti una volta per gruppo anziché una
volta per token. L'ordine di accumulazione matrice-matrice differisce dal
percorso matvec, motivo per cui resta opt-in finché non è validato A/B.

Numericamente la pipeline in batch è identica al percorso per token (la FFN di
un token non alimenta altri token all'interno del layer); viene deduplicato
solo l'I/O degli esperti — al massimo `min(6·tokens, 256)` letture di esperti
per layer per gruppo invece di `6·tokens`.

### Slot-cache degli esperti

`ExpertSlotCache` è una cache LRU per layer per gli slab degli esperti.
Supporta:

- budget di slot fissi;
- budget minimo effettivo di 8 slot quando abilitata;
- allocazione tra i layer guidata dall'utilizzo (modalità uniforme per l'A/B
  via `DS4_EXPERT_CACHE_UNIFORM=1`);
- pre-warming dalla usage imatrix persistita.

La cache è un compromesso di memoria wired. Aiuta solo quando il routing è
abbastanza concentrato da far ripetere gli esperti caldi. La scheda Tuning
espone hit-rate e concentrazione, così questo può essere misurato invece che
indovinato.

**Layout del pool interfogliato** (`DS4_POOL_INTERLEAVE`, default attivo): ogni
slot contiene gli slab gate|up|down del suo esperto CONTIGUI in un solo buffer
— lo stesso layout di record del sidecar expert-bundle — così un miss servito
dal bundle diventa UNA pread di ~7 MB direttamente nello slot (1 syscall invece
di 3, I/O più grandi a parità di profondità di coda). I kernel non cambiano:
gate/up/down sono tre viste dello stesso buffer e lo stride tra gli esperti è
la dimensione del record. `=0` ripristina il layout storico a tre buffer
stretti.

**Percorso di riempimento**: su un miss senza bundle, i tre slab di un esperto
vengono letti IN CONCORRENZA (tre job paralleli per miss), e
`DS4_PREAD_SPLIT=N` (default 1, max 8) divide ulteriormente ciascuno slab in N
range disgiunti letti con pread in parallelo sul percorso `F_NOCACHE` — i miss
in decode sono pochi per layer, e alzare la profondità di coda NVMe è ciò che
mantiene il disco al suo tetto di parallelismo.

**Concorrenza**: le operazioni sono serializzate per layer (un lock per layer),
così l'`acquire(layer: i)` del thread di decode può girare mentre un prefill in
background riempie il layer i+1. Il percorso a domanda ha priorità: un prefill
speculativo in corso riempie a chunk e cede il passo tra un chunk e l'altro
quando un acquire a domanda è in attesa, quindi la speculazione può ritardare
il percorso critico al massimo di circa un chunk di riempimento. Un riempimento
speculativo non sfratta mai gli slot dell'ultimo acquire a domanda (potrebbero
essere ancora letti da un command buffer in volo).

**Look-ahead speculativo** (`kickLookahead`): all'inizio del layer i (e di ogni
token), il decoder risolve sul thread di decode i probabili id degli esperti
del layer i+1 e avvia il loro prefill nel pool su una coda in background, così
l'I/O di riempimento gira sotto il calcolo del layer i — il trucco
`begin_selected_load` del motore C. Per i layer hash-routed (0–2) gli id sono
ESATTI (una lettura mmap `tid2eid` dal token id), quindi il loro I/O degli
esperti è sempre nascosto; per gli altri layer l'ipotesi è il top-N del prior
di utilizzo, opt-in con `DS4_EXPERT_LOOKAHEAD=N` (default 0; un'ipotesi
sbagliata spreca solo banda della finestra di inattività). Gli slab prefillati
non contano come miss nelle statistiche — il loro I/O è girato fuori dal
percorso critico. I layer a precisione mista (fuori dalla classe di dimensione
della cache) sono esclusi.

### Sidecar expert-bundle

`DS4_EXPERT_BUNDLE=1` aggiunge un'ottimizzazione lato disco per i miss di
cache. Il motore cerca o costruisce un sidecar (`<gguf>.expbundle`, record
allineati a 4 KB ordinati per layer e poi per id esperto) in cui gli slab gate,
up e down di ogni esperto sono memorizzati contigui. Un miss può allora essere
soddisfatto da una sola lettura sequenziale di ~7 MB invece di tre letture
sparse di ~2 MB dal layout tensoriale GGUF originale — e con il pool di slot
interfogliato il layout del record coincide con il layout dello slot, così la
copia è una singola pread nello slot.

Il sidecar non è una nuova quantizzazione e non cambia la matematica. Duplica
la regione di byte degli esperti su disco, viene validato contro il modello
sorgente (dimensione/geometria più fingerprint di contenuto per layer) e viene
saltato quando lo spazio scrivibile è insufficiente. Nelle build sandboxed
dell'app, `DS4_BUNDLE_DIR` indirizza la creazione verso Application Support; un
sidecar leggibile accanto al GGUF può comunque essere riusato. Ogni caricamento
logga lo stato del bundle e l'uso è dimostrato a runtime da un heartbeat
logaritmico (primo esperto servito, poi 5k, 10k, 20k, …).

### Streaming denso e pesi densi residenti

I pesi densi di attention/condivisi servono sempre, quindi l'eviction della
page cache può dominare il decode a bassa RAM anche quando l'I/O degli esperti
è ottimizzato. Il motore supporta tre strategie:

| Strategia | Knob | Note |
|---|---|---|
| mmap/page cache | default | Memoria wired minima; ideale quando la RAM può tenere residenti le pagine dense calde. |
| streaming denso | `DS4_DENSE_STREAM=1` | Legge i tensori densi di ciascun layer in un piccolo ring di staging (`pread + F_NOCACHE`) un layer in anticipo rispetto al calcolo, così la lettura SSD del layer i+1 si sovrappone al calcolo GPU del layer i. ~300 MB di staging invece di ~6 GB residenti. Ha precedenza sul denso residente. |
| denso residente | `DS4_RESIDENT_DENSE=1` | Copia i pesi densi in buffer GPU residenti. Utile sui sistemi ricchi di RAM, rischioso su 16 GB. |

`DS4_DENSE_AHEAD` controlla la profondità di read-ahead dello staging (default
1, il classico ring a 2 slot; limitata a un massimo di 3). Profondità maggiori
possono migliorare la sovrapposizione ma competono anche con le letture degli
esperti sullo stesso SSD.

Lo stream denso ospita diversi scorpori, pesi che escono dal ring perché farne
streaming a ogni token è il compromesso sbagliato:

- l'**output head** (`output.weight`, ~560 MB Q8, letto per intero a ogni
  token) viene copiato RESIDENTE ogni volta che `DS4_DENSE_STREAM=1` —
  mappato, veniva riletto a ogni token attraverso una page cache fredda. La
  tabella di embedding resta mappata: il decode mette in staging una riga di
  ~8 KB per token in un piccolo buffer di staging invece di fissare in memoria
  l'intera tabella;
- le **proiezioni del compressore NSA** (`DS4_RESIDENT_COMP`, sezione 8);
- le **proiezioni di scoring dell'indexer**, rinviate da `DS4_LAZY_IDX` fino al
  confine sparse vivo e poi mantenute residenti (sezione 8);
- le **proiezioni riquantizzate Q4** di `DS4_DENSE_Q4` / `DS4_SHARED_Q4`
  (sezione 12).

### Pinning dei buffer caldi

`DS4_MLOCK=1` richiede un `mlock()` best-effort sui buffer Metal condivisi
caldi come i pool della cache esperti, lo staging dello stream denso, l'output
head residente, i buffer densi Q4 e il ring raw-KV se abilitato. È una
protezione di prestazioni contro il churn del compressore di memoria di macOS,
non un requisito di correttezza. Il mancato pinning viene loggato o ignorato a
seconda del punto di chiamata; il decode continua.

### Pattern del command buffer diviso

Il percorso di streaming è intenzionalmente diviso attorno alle decisioni di
routing visibili alla CPU:

1. eseguire il lavoro di route/attention;
2. rileggere gli id degli esperti selezionati;
3. fare gather/cache di quegli esperti;
4. eseguire il lavoro FFN degli esperti.

Questo rende l'I/O SSD degli esperti esplicito e misurabile. Significa anche
che l'overhead di round-trip dei command buffer può contare; `DS4_DIAG=1`
misura il costo di un command buffer vuoto nella demo CLI. Il passo 4 è la metà
asincrona della pipeline: il buffer della FFN routed viene committato senza
attesa CPU (`DS4_ASYNC_FFN`, sezione 5) e l'attesa del route del layer
successivo fa anche da punto di join, così solo il round-trip del route resta
sincrono.

Tutti i knob `DS4_*` di questa sezione sono documentati con i default nella
[Configuration Reference](../README.md#configuration-reference) alla radice.

### Costruttori

Le famiglie di costruzione importanti sono:

| Costruttore | Strategia |
|---|---|
| `fromGGUFExpertCachedMapped` | Percorso veloce di streaming da SSD: pesi non routed mappati, esperti selezionati raccolti via gather, slot-cache opzionale. |
| varianti expert-cache | Tengono residenti gli esperti caldi e riempiono su miss. |
| varianti quant per layer | Decodificano correttamente la quantizzazione mista degli esperti leggendo i tipi tensoriali per layer. |

## 11. Campionamento: dai logits al token

`DS4Core/Generation/Sampler.swift` implementa:

- modalità greedy (`temperature = 0`);
- campionamento con temperatura;
- top-k;
- top-p;
- min-p;
- penalità di ripetizione.

La demo usa il campionamento greedy. L'app espone temperatura e penalità di
ripetizione. Il percorso servizio/server usa `SamplingParams`, così UI, HTTP e
coordinatore distribuito possono condividere il comportamento.

La penalità di ripetizione è importante per i modelli pesantemente quantizzati,
perché dopo molti token generati possono verificarsi loop di collasso.

## 12. Quantizzazione

Il motore gestisce i formati quantizzati richiesti dai GGUF di destinazione:

- matvec densi Q8_0;
- esperti routed Q4_K;
- esperti routed Q2_K;
- esperti routed IQ2_XXS;
- tensori scalari o di normalizzazione F16/F32 dove richiesto.

`DS4_Q8_NSG` regola lo scheduling dei matvec densi Q8_0. Partiziona il lavoro e
le somme parziali tra i simdgroup: l'operazione matematica è invariata, ma
l'ordine di riduzione in virgola mobile può alterare gli ultimi bit.

La quantizzazione mista degli esperti routed è supportata per layer. I modelli
uniformi mantengono lo stesso percorso di selezione a classe di quantizzazione
singola; i layer misti fuori classe bypassano quella cache e selezionano il
formato di gather/decodifica dichiarato per il singolo layer. Questa
affermazione riguarda la selezione di formato/percorso, non logits o token
generati identici byte per byte.

### Riquantizzazione densa Q4

`DS4_DENSE_Q4=1` (richiede `DS4_DENSE_STREAM=1`) è un percorso di velocità
deliberatamente lossy. Il motore riquantizza in Q4_K le proiezioni di attention
Q8 più grandi (`q_b`, `output_a`, `output_b`) e mantiene residenti i buffer
ridotti. Rimuove diversi GB di traffico denso per token dal percorso SSD.
`DS4_SHARED_Q4=1` estende la stessa idea alle proiezioni FFN degli esperti
condivisi, e `DS4_QKV_Q4=1` alle restanti proiezioni di attention di media
dimensione (`q_a`, `kv` — gli ultimi slab di attention Q8 ancora in streaming,
~0.7 GB/token per ~0.35 GB residenti); entrambi vanno trattati come esperimenti
A/B separati. Tutti e tre i knob condividono la stessa cache `.q4dense`: i
record sono abbinati per chiave (layer, tensore), quindi abilitare un nuovo
knob riusa la cache esistente e riquantizza solo i tensori che quel knob
aggiunge.

La conversione è persistita in una **cache di riquantizzazione Q4**
(`<gguf>.q4dense`, circa 1.4 GB per il trio denso-Q4 di base e più grande
quando i record QKV/shared sono abilitati): il primo caricamento paga la
riquantizzazione una volta, mentre i caricamenti successivi leggono con pread
la cache validata invece di convertire di nuovo gli stessi tensori; il tempo di
caricamento dipende da SSD, pressione di memoria e hardware. La
riquantizzazione **crea subito il file di cache vuoto** (header valido, zero
record — un preflight di scrittura: problemi di permessi, percorso o spazio su
disco emergono nel log *prima* di qualsiasi lavoro di conversione, e la
presenza del file dimostra che i checkpoint hanno un posto dove atterrare), poi
procede a batch e **scrive su disco checkpoint della cache parziale tra un
batch e l'altro** (stesso formato, meno record): un primo caricamento
interrotto a metà riquantizzazione (uscita forzata, crash, riavvio)
riprende dai tensori completati invece di ripartire da zero, e la barra di
caricamento avanza per **MB di sorgente convertiti** anziché per tensore, così
una conversione lunga appare visibilmente in corso invece di sembrare bloccata.
La validazione è contro i *byte* del modello (fingerprint di contenuto di ogni
tensore sorgente, verificati per record), non solo contro la dimensione del
file. Un preflight fallito o una scrittura di checkpoint fallita viene loggata
come `DS4 q4cache:`; le scritture usano un file temporaneo adiacente più
rename, così un file troncato non sostituisce mai una cache valida. Il
caricamento corrente continua in sicurezza con i tensori convertiti in memoria,
mentre un caricamento successivo riquantizza i record non persistiti. La cache
vive accanto al modello per default (demo/CLI); l'app sandboxed non può
scrivere accanto a un file scelto dal picker, quindi `DS4_Q4_CACHE_DIR` la
indirizza verso Application Support. Le letture provano entrambe le posizioni,
e una cache completa trovata nella posizione secondaria (per esempio prodotta
dalla demo accanto al GGUF) viene PROMOSSA in quella primaria, così demo e app
condividono una sola conversione. Le fette di layer distribuite leggono dalla
cache completa ma scrivono un proprio nome di cache per intervallo, così la
riquantizzazione di una fetta non può mai sovrascrivere la cache completa.

Poiché cambia i pesi, non è una modalità di parità. Va usata per il throughput,
non quando si confrontano logits esatti con un percorso denso full-Q8.

## 13. Riepilogo dei tensori per layer

Ogni layer di decode usa gruppi di tensori che corrispondono all'incirca a:

| Gruppo | Esempi |
|---|---|
| Norm/proiezioni di attention | `attn_norm`, `attn_q_a`, `attn_q_b`, `attn_kv`, `attn_output_a`, `attn_output_b` |
| Compressore/indexer | pesi del compressore e della selezione sparsa |
| Router | `ffn_gate_inp` o tensore di selezione routed equivalente |
| FFN condivisa | tensori gate/up/down condivisi |
| Esperti routed | `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` |
| HC | tensori Hyper-Connection di attention/FFN/output |
| Output | norm finale e output head |

`DS4_TYPES_ONLY=1` stampa dtype tensoriali rappresentativi per un GGUF, così
una discrepanza può essere diagnosticata prima del costoso decode.

## 14. Tool calling

### Token speciali

Il tokenizer riconosce i token del protocollo DS4:

- delimitatori dei turni user / assistant;
- inizio/fine think;
- token di tool call DSML;
- marcatori di fine frase.

### Formato DSML

DSML è il formato di tool call nativo del modello. Il `ChatRenderer` compilato
di DwarfStar implementa il formato rispetto a un template DeepSeek-V4 di
riferimento; non esegue il testo Jinja del GGUF. La modalità GUI compatta è
abilitata per default e invia una dichiarazione più corta per un costo di
prefill inferiore, deviando deliberatamente dal testo completo orientato al
training; la modalità completa resta più vicina a quello schema di riferimento.

### Rendering e parsing

`ChatRenderer` renderizza:

- prompt di sistema;
- turni user/assistant;
- dichiarazioni dei tool;
- risultati dei tool.

I payload dei risultati dei tool sono avvolti in `<tool_result>…</tool_result>`.
Il renderer deliberatamente NON fa escape HTML del contenuto (output di shell e
frammenti di file devono restare intatti) ma fa escape dell'esatta sentinella
di chiusura (`escapeToolResult`, rispecchiando la
`bpe_tokenize_tool_result_text` del C): un `</tool_result>` malevolo o
accidentale dentro un payload non può quindi mai terminare il wrapper in
anticipo e iniettare token di controllo nel prompt.

`ToolCallParser` rimuove il markup trapelato e interpreta i blocchi DSML
completati in valori `ToolCall`.

### Orchestrazione del ciclo dei tool

`InferenceService` trasmette in streaming il testo dell'assistente, interpreta
un blocco DSML completo ed emette `.toolCall`; non esegue il tool richiesto.
L'applicazione possiede il ciclo:

1. `ChatStore+ToolLoop` gestisce la chat locale, mentre `DistributedController`
   gestisce la chat distribuita;
2. l'orchestratore smista built-in/MCP attraverso `ToolRegistry.executeAuto` e
   gestisce i flussi di sub-agent o di risultato manuale dove supportati;
3. registra gli output e richiede una continuazione;
4. la chat locale chiama `InferenceService.provideToolResults`, mentre la chat
   distribuita accoda turni `toolResult` prima di richiamare il coordinatore.

### Registry e tool

`ToolRegistry` possiede i built-in e gli schemi JSON. I tool sono divisi uno
per file per facilitarne la revisione. I tool di progetto operano su
`ProjectCache`, che resta separata dalla memoria della chat finché i risultati
dei tool non vengono inseriti nella conversazione.

### Client MCP

`DS4Engine/Tools/MCP` aggiunge un client Model Context Protocol, così i server
MCP esterni appaiono al modello accanto ai built-in:

- `MCPConfig` parla l'interscambio JSON `{"mcpServers": …}` usato da Claude
  Desktop / Cursor / VS Code, così le configurazioni si importano ed esportano
  alla lettera;
- `MCPTransport` implementa i due trasporti della specifica: stdio (server
  avviato come processo figlio, JSON-RPC delimitato da newline su
  stdin/stdout) e Streamable HTTP (frame inviati via POST, risposte JSON o
  SSE, `Mcp-Session-Id` rimandato in eco);
- `MCPClient` possiede una connessione: handshake di initialize, timeout per
  richiesta, ping del server, segnalazione delle disconnessioni e
  cancellazione dei task (Stop chiude immediatamente una chiamata in volo);
- `MCPManager` è il registry a livello di processo: possiede i client e serve
  snapshot in cache (stati, `ToolSpec` con namespace) a chat, agenti e
  modalità distribuita. Il tool `read_file` sul server `fs` diventa
  `mcp_fs_read_file`; la mappatura inversa è un indice esplicito, mai parsing
  dei nomi.

I cicli dei tool non chiamano mai direttamente questo livello:
`ToolRegistry.executeAuto(_:)` smista prima i built-in, poi MCP.

### Sub-agent

I sub-agent sono implementati come un cambio di contesto lato motore:

- snapshot della KV principale;
- costruzione o ripristino del prefisso di contenuto del sub-agent;
- esecuzione del ciclo dei tool in un contesto isolato;
- restituzione della risposta;
- ripristino della KV principale.

Questo consente alla conversazione principale di delegare lavoro senza ingerire
ogni lettura interna di file o passaggio intermedio di ragionamento.

## 15. Inferenza distribuita

L'inferenza distribuita divide intervalli contigui di layer tra i worker.

### API delle fette del decoder

I worker possiedono istanze di `StreamingDecoder` per una fetta di layer.
Allocano solo i pesi e la KV di quella fetta. Il coordinatore possiede
embedding, campionamento, rendering del prompt e orchestrazione finale.

### Protocollo e topologia

Il protocollo di rete (`Sources/DS4Engine/Distributed/Protocol/`) è un framing
nativo DwarfStar, attualmente alla versione 11. Il coordinatore valida la
versione PER PRIMA COSA, così un cluster misto fallisce con un errore chiaro
invece che con frame corrotti. La cronologia delle versioni fa anche da elenco
delle funzionalità:

- **v2** — robustezza: HELLO trasporta la versione; WORK/RESULT trasportano un
  id di `session` per turno rimandato in eco dai worker, così un risultato
  lasciato in un buffer TCP da un turno annullato non può mai essere scambiato
  per la risposta del turno successivo.
- **v3** — il COORDINATORE definisce il compito di ogni worker. I worker
  partono inattivi (in ascolto, nessun modello caricato); il coordinatore invia
  ASSIGN (gguf, contesto, fetta di layer, slot di cache) e il worker risponde
  READY una volta caricato il suo motore.
- **v4** — continuità KV distribuita: ASSIGN trasporta la usage imatrix
  (pre-warm della slot cache) e un budget di token per il disk-KV; i frame di
  controllo KV (`kvQuery` / `kvLengths` / `kvRestore` / `kvSave` / `kvAck`)
  permettono al coordinatore di fare checkpoint/ripristino dello shard di ogni
  worker; WORK ha guadagnato `turnStart`, così un turno può iniziare a metà
  contesto.
- **v5** — il coordinatore DISTRIBUISCE i file. Dopo HELLO invia una FILE
  OFFER (nome, dimensione, SHA-256 per gguf + sidecar); il worker risponde con
  ciò che gli manca (verificato via hash contro il suo store) e solo quello
  viene trasmesso — l'enorme setup gira una volta, le connessioni successive
  verificano gli hash dai manifest in cache in millisecondi.
- **v6** — viaggiano anche le cache derivate: l'offerta può includere la cache
  di riquantizzazione densa Q4 (`<gguf>.q4dense`), e ASSIGN trasporta la
  decisione Q4 on/off.
- **v7** — WORK trasporta i token id del chunk: i primi 3 layer fanno routing
  degli esperti per TOKEN ID (`ffn_gate_tid2eid`), quindi uno shard che li
  copre non può fare routing dal solo stato HC.
- **v8** — trasferimenti RIPRENDIBILI: l'offerta trasporta per ogni file una
  lista di checkpoint a hash concatenato (uno SHA-256 ogni 256 MB, ciascuno
  ripiegato sul precedente, così `chain[k]` attesta l'intero prefisso); il
  worker conserva il suo `.part` tra disconnessioni e sessioni, lo valida
  blocco per blocco, lo tronca all'ultimo checkpoint valido e risponde FILE
  NEED con un offset di ripresa per file. Il coordinatore ritenta il setup di
  un peer interrotto fino a 3 volte.
- **v9** — ASSIGN trasporta i KNOB DI PRESTAZIONI del coordinatore: un insieme
  in whitelist di variabili d'ambiente `DS4_*` (`Dist.perfKnobKeys`). La
  whitelist impedisce che la rete imposti un ambiente arbitrario; non è un
  contratto di parità numerica. I percorsi ammessi di fusione, batching e
  prefill-MM possono cambiare l'ordine di riduzione/accumulazione in virgola
  mobile. `DS4_DENSE_Q4`, deliberatamente lossy, viaggia come campo tipizzato
  separato con la sua cache. Il worker applica la configurazione del compito
  del coordinatore prima del caricamento, senza promettere risultati identici
  al bit tra hardware o percorsi di esecuzione diversi.
- **v10** — parallelismo degli esperti: `expertAssign`, `expertWork` ed
  `expertSum` permettono ai worker di possedere maschere sui 256 esperti
  routed attraverso tutti i layer. Il coordinatore esegue la dorsale densa e
  combina le somme FFN parziali remote. Chat verticale e benchmark sono
  cablati nell'app; la topologia richiede un collegamento cablato a bassa
  latenza perché esegue un round-trip di rete per ogni layer routed.
- **v11** — geometria distribuita posseduta dal modello: le fette orizzontali
  validano 43 layer Flash o 61 Pro, mentre `expertAssign` trasporta una
  maschera con prefisso di lunghezza per 256 esperti Flash o 384 Pro. `READY`
  riporta la geometria effettivamente caricata. Il GGUF Pro Q2 completo è
  accettato; il Pro Q4 diviso richiede ancora un loader multi-shard.

Il setup dei peer gira IN PARALLELO: il trasferimento dei file e il caricamento
del motore di ogni worker procedono insieme in un task group, così
l'attivazione della route costa `max(worker setup)` invece della somma.
L'ordine della route resta l'ordine della lista dei peer, e la copertura
contigua viene comunque validata dopo l'assemblaggio.

Ogni token o chunk di prefill invia lo stato HC lungo la route. In modalità
relay, il coordinatore fa un round-trip attraverso ogni worker. In modalità
forwarding, i worker passano lo stato al worker successivo e il worker
terminale ritorna al listener del coordinatore.

Il benchmark distribuito riusa il coordinatore già connesso. Non deve girare
mentre la chat distribuita sta generando, perché entrambi condividono la stessa
route e azzerano la KV dei worker.

Il flusso operativo completo, il comportamento di ripresa dei file e il
confronto tra le topologie sono in
[`INFERENZA-DISTRIBUITA.md`](INFERENZA-DISTRIBUITA.md).

## 16. Riferimento incrociato da C a Swift

| Area C upstream | Port Swift |
|---|---|
| forma del modello, uso del tokenizer, flusso del decoder di `ds4.c` | `DS4Core/Model/Backends/DeepSeekV4`, `DS4Core/Tokenization/Backends/DeepSeekV4`, `DS4Metal/Backends/DeepSeekV4/Decode`, `DS4Engine/Inference/Service` |
| runtime e kernel di `ds4_metal.m` | `DS4Metal/Runtime`, `DS4Metal/Kernels`, `metal/*.metal` |
| parsing GGUF | `Sources/DS4Core/Formats/GGUF/` |
| modello a esperti in streaming da SSD | `Sources/DS4Metal/Backends/DeepSeekV4/Weights/GGUFWeights.swift`, `StreamingDecoder` |
| streaming denso / cache di riquantizzazione Q4 | `Sources/DS4Metal/Backends/DeepSeekV4/Streaming/DenseStreamer.swift` |
| cache esperti | `Sources/DS4Metal/Backends/DeepSeekV4/Decode/Cache/`, usage imatrix in `StreamingDecoder` |
| sidecar expert-bundle | `Sources/DS4Metal/Backends/DeepSeekV4/Experts/ExpertBundle.swift` |
| store KV | `Sources/DS4Core/Formats/KVCheckpoint/KVCFile.swift`, `Sources/DS4Engine/Persistence/KV/DiskKVStore.swift` |
| server | `Sources/DwarfStar/Features/Server`, `LocalServer` |
| runtime distribuito | `DS4Engine/Distributed` |
| client MCP (nessun equivalente C) | `DS4Engine/Tools/MCP` |
| bring-up CLI | `Sources/DS4Demo/Command/main.swift` |

Il port Swift sostituisce deliberatamente la gestione della memoria specifica
del C con ARC, il parsing Foundation dove appropriato, gli actor Swift per
l'isolamento dei servizi e i modelli di stato SwiftUI per l'app. La matematica
dei kernel e il comportamento visibile al modello restano le parti che devono
seguire l'upstream più da vicino.
