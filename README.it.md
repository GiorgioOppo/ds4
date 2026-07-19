[English](README.md) | **Italiano**

# DwarfStar — inferenza LLM nativa su macOS

DwarfStar è un'applicazione nativa Swift / SwiftUI per eseguire modelli
linguistici locali su Apple Silicon con un **motore di inferenza Metal in puro
Swift**. **DeepSeek V4** è oggi il backend operativo: Flash e il profilo Pro Q2
a file singolo girano localmente attraverso lo stesso decoder Metal guidato
dalla geometria. Il pacchetto Pro Q4 a due shard resta solo scaricabile. Il Pro
Q2 distribuito è collegato attraverso la pipeline guidata dalla geometria e i
percorsi a shard di esperti; la validazione numerica multi-Mac con modello
reale è ancora in sospeso. L'albero dei sorgenti e il loader dei modelli sono
in preparazione per un backend Qwen indipendente. I file GGUF Qwen vengono
riconosciuti ma rifiutati intenzionalmente finché tokenizer, mappatura dei
tensori e decoder non saranno implementati. GLM 5.2 è in una fase di porting
nativo a tappe: le tre varianti GGUF monolitiche di `antirez/glm-5.2-gguf`
sono catalogate con dimensioni fissate e digest SHA-256; questa build
riconosce `glm-dsa`, ne valida forma/tensori e ne fornisce il frontend di
tokenizer/chat, ma rifiuta ancora l'inferenza finché il decoder Metal completo
non supera i test end-to-end sui logits. Il motore DeepSeek è un port fedele
degli upstream `ds4.c` / `ds4_metal.m`: nessun motore runtime in C, nessuna
libreria statica precompilata né processo esterno per l'inferenza normale. Il
GGUF Flash a 2 bit gira su un MacBook da 16 GB facendo streaming dei pesi
degli esperti instradati da SSD; quell'affermazione sulla memoria non descrive
l'artefatto Pro, molto più grande.

> La documentazione parte da [`docs/README.md`](docs/README.it.md). Il
> [Riferimento di configurazione](#riferimento-di-configurazione) qui sotto è
> la tabella autorevole dei parametri; le guide dedicate coprono le
> [architetture supportate](docs/ARCHITETTURE-SUPPORTATE.it.md),
> la [pipeline di inferenza](docs/PIPELINE-INFERENZA.it.md),
> il [backend Metal](docs/BACKEND-METAL.it.md),
> l'[inferenza distribuita](docs/INFERENZA-DISTRIBUITA.it.md),
> [GUI/server](docs/GUI-SERVER-E-API.it.md) e
> il [testing](docs/TESTING-E-VALIDAZIONE.it.md).

## Architettura

```text
DwarfStar (SwiftUI)            <- chat · agenti · progetti · tuning · server ·
        |                         worker distribuito · benchmark · diagnostica
   DS4Engine (Swift)           <- ispettore dei modelli + selettore di backend,
        |                         InferenceService, strumenti, agenti, ProjectCache,
        |                         server HTTP, coordinatore/worker distribuito
   DS4Core + DS4Metal (Swift)  <- substrato condiviso GGUF/campionamento/Metal +
        |                         backend concreto (DeepSeek V4 Flash/Pro)
   metal/*.metal               <- sorgente di verità dei kernel, incorporata alla build
```

Il GGUF viene ispezionato prima di costruire un tokenizer o un decoder. La
selezione avviene una sola volta al caricamento del modello e il loop dei
token mantiene tipi di backend concreti, quindi il confine multi-modello non
aggiunge dispatch dinamico all'esecuzione per layer.

La correttezza è la regola primaria del progetto. Il motore Swift è validato
dai test in `Tests/DS4CoreTests/`, con molti kernel e stadi del grafo
verificati rispetto al comportamento upstream.

## Fatti essenziali del backend DeepSeek

- **Lo streaming da SSD è l'unico percorso di caricamento del modello.** I
  pesi non instradati sono viste `mmap` senza copia sostenute dalla page cache
  del sistema operativo. A ogni token vengono raccolti solo gli esperti
  instradati selezionati per il layer corrente. Il modello completo non deve
  mai stare in RAM.
- **I kernel Metal sono incorporati nel binario.** `make embed-kernels`
  rigenera `Sources/DS4Metal/Runtime/Generated/KernelSources.swift` da
  `metal/*.metal`, così un'app pacchettizzata non ha bisogno di una cartella
  `metal/` su disco a runtime.
- **I kernel MoE fusi coprono i formati di esperti in uso.** I percorsi
  Pair-SwiGLU e down-sum6 coprono esperti q4_K, q2_K e iq2_xxs.
- **La slot-cache degli esperti è opzionale e guidata dall'uso.** Una cache
  LRU per layer può mantenere gli esperti caldi residenti sulla GPU. Le
  statistiche di frequenza di instradamento vengono persistite per modello e
  per agente, poi usate per pre-riscaldare e ridistribuire gli slot della
  cache.
- **Il riuso del KV multi-turno è solo in append.** `InferenceService` traccia
  gli id esatti dei token committati. Ogni nuovo turno esegue il prefill solo
  del suffisso non già presente nel KV. Se la generazione viene interrotta, il
  turno successivo ricostruisce dagli id committati perché il compressore NSA
  è ricorrente e non può tornare indietro.
- **La chiamata di strumenti usa DSML nativo.** Il token di controllo
  `｜DSML｜` del modello apre un formato di chiamata in stile XML prodotto dal
  `ChatRenderer` compilato, implementato sul template di riferimento
  DeepSeek-V4. Il runtime non interpreta il template Jinja del GGUF; la
  modalità compatta accorcia deliberatamente le dichiarazioni degli strumenti
  per ridurre il prefill. I risultati degli strumenti tornano dentro un turno
  utente come `<tool_result>...</tool_result>`.
- **Il prefill layer-major ammortizza l'I/O.** Il prefill gira a chunk: i pesi
  di ogni layer vengono caricati una volta per chunk e applicati a tutti i
  token del prompt. La fase FFN instradata raccoglie l'unione degli esperti di
  quel chunk invece di 6 esperti per token.

## L'app

| Scheda | Scopo |
|---|---|
| **Chat** | Chat markdown in streaming, ragionamento richiudibile, chiamate di strumenti in tempo reale, riuso del KV multi-turno, allegati di file di testo, menu del progetto attivo, avvisi di contesto quasi pieno, modalità Local o Distributed. |
| **Settings** | Percorso condiviso del modello, dimensione del contesto, modalità di esecuzione consapevole della RAM, regolazioni di memoria/I/O, caricamento del modello locale e rotta verso il coordinatore distribuito. |
| **Agents** | Editor dei ruoli: prompt di sistema, icona, strumenti, import/export JSON e profilo di uso degli esperti per agente. |
| **MCP** | Server MCP esterni (processo figlio stdio o Streamable HTTP): stato della connessione, strumenti esposti, import/export del JSON `mcpServers`. |
| **Project** | Libreria delle cartelle importate (bookmark sandbox) e dei cloni GitHub fatti in chat con `github_clone` (elencati automaticamente). Gli strumenti di progetto esplorano il progetto attivo senza consumare contesto di chat finché uno strumento non ne legge il contenuto. |
| **Tuning** | Slot della cache degli esperti, hit-rate, concentrazione di instradamento per layer e la imatrix d'uso. |
| **Server** | Server HTTP nativo in-process compatibile OpenAI/Anthropic. |
| **Worker** | Esegue questo Mac come worker distribuito: parte inattivo e il coordinatore gli assegna il lavoro — GGUF, impostazioni e fetta di layer. |
| **Benchmark** | Benchmark nativo di throughput per prefill e generazione su dimensioni di contesto crescenti, in locale o distribuito. |
| **Diagnostics** | Dump nativo del tokenizer e ispezione del template di chat e del formato degli strumenti. |

## Strumenti e agenti integrati

Gli strumenti DSML integrati vivono uno per file sotto
`Sources/DS4Engine/Tools/Builtins/`.

- **Strumenti dell'indice di progetto:** `project_list`, `project_read`,
  `project_search`, `project_write`, `project_edit`, `project_reload`
  (re-indicizza dopo modifiche fuori banda; `git stash` re-indicizza
  automaticamente).
- **Import da GitHub:** `github_clone` scarica un repository pubblico (tarball
  HTTPS, senza binario git né credenziali) in Application Support e lo rende
  il progetto attivo, restituendo un riassunto compatto di orientamento
  (albero + file di documentazione). Gli agenti Coding e Code lo usano per
  analizzare un repo con gli strumenti di progetto — struttura via
  `project_tree`, ricerca mirata nel codice via
  `project_find`/`project_search` — invece di leggere ogni file nel contesto
  della chat. I cloni compaiono anche automaticamente nella scheda Project (e
  nel menu Project della chat) come normali voci di libreria, quindi possono
  essere riattivati dalla GUI in qualsiasi momento.
- **Strumenti file grezzi sulla radice del progetto:** `file_read`,
  `file_lines`, `file_write`, `file_add`, `file_modify`.
- **Utilità:** `git` (whitelist locale, senza rete), `calculator`,
  `add`, `subtract`, `multiply`, `now`.
- **Sub-agenti:** `agents_list`, `subagent_search`,
  `subagent_run(target, question, agent?, tools?)`. I sub-agenti girano in un
  contesto isolato con la propria cache di prefissi KV a chiave di contenuto.
  La chat principale riceve solo la domanda delegata e la risposta restituita,
  non il lavoro interno del sub-agente.

Gli agenti predefiniti sono **General**, **Coding**, **Code**,
**Orchestrator**, **Math**, **Writing**, **LaTeX** e **Documentation**. Ogni
agente è un prompt di sistema più una allow-list di strumenti più un profilo
dedicato di uso degli esperti.

Oltre agli integrati, la scheda **MCP** collega l'app a server
[Model Context Protocol](https://modelcontextprotocol.io) esterni (stdio o
Streamable HTTP). I loro strumenti compaiono accanto agli integrati — nel
foglio Tool della chat e nelle liste di strumenti degli agenti — come
`mcp_<server>_<tool>`, e le chiamate vengono inoltrate al server via
`tools/call` (vedi `Sources/DS4Engine/Tools/MCP/`).

## Avvio rapido

```sh
make                  # swift build
make xcodeproj        # rigenera DwarfStar.xcodeproj dopo aver aggiunto file
swift run DwarfStar   # avvia l'app
make test             # test unitari
```

In **Settings**, usare **Scarica…** per acquisire un modello del catalogo
direttamente dalla GUI. Le tre varianti DeepSeek V4 Flash e il modello
DeepSeek V4 Pro Q2 a file singolo possono essere scaricati, selezionati ed
eseguiti localmente. Il pacchetto Pro Q4 a due file resta solo scaricabile
perché il loader locale non assembla shard GGUF divisi. Anche le tre varianti
GLM 5.2 di `antirez/glm-5.2-gguf` sono scaricabili, riprendibili e con
verifica di integrità. I loro metadati e frontend possono essere ispezionati,
ma restano non selezionabili finché il decoder GLM nativo non sarà completo. I
download vivono sotto `~/Library/Application Support/DwarfStar/models/`,
riprendono dai file `.part` e riusano un file regolare già presente invece di
scaricarlo di nuovo; le voci con un conteggio di byte fissato, incluse quelle
GLM 5.2, richiedono una corrispondenza esatta della dimensione prima del
riuso.

**Browse** resta disponibile per un GGUF avanzato/manuale. Il selettore
ispeziona il file e lo accetta solo quando il runtime corrente può eseguirne
architettura e profilo. Con App Sandbox abilitata crea anche un bookmark
security-scoped persistente. Premere **Load Model**, poi aprire **Chat**.
L'interruttore **Thinking** abilita la gestione dei token di ragionamento;
**Stop** annulla la generazione e il turno successivo ricostruisce il KV se
necessario.

Il preset GUI corrente privilegia il percorso veloce misurato per 16 GB: 22
slot di cache esperti, pool mixed-quant consapevoli del layer attivi, ring
raw-KV attivo, `pread` diretto degli esperti, streaming dei pesi densi,
`mlock` best-effort, il set Q4 completo (`DENSE_Q4`, `QKV_Q4`, `SHARED_Q4`),
KV su disco, expert bundle e MetalIO. Il prefill usa union/chunk/route-batch
256/512/32; il read-ahead denso è 2, la FFN asincrona è attiva e gli NSG
Q8/MoE/denso-Q4 sono tutti 4. La maggior parte degli interruttori di memoria
si applica al successivo caricamento del modello.

## Server HTTP

La scheda Server avvia un server compatibile OpenAI/Anthropic su
`Network.framework`. È nativo e in-process ed espone **il singolo motore
condiviso** caricato in Settings — nessun sottoprocesso e nessuna seconda
copia del modello (un secondo motore raddoppierebbe i buffer Q4 residenti +
mlockati e andrebbe in OOM su 16 GB). L'actor `InferenceService` serializza i
chiamanti, quindi una richiesta al server attende un turno di chat in corso e
viceversa. Il Benchmark riusa lo stesso motore condiviso.

| Metodo | Percorso | API |
|---|---|---|
| GET | `/v1/models`, `/v1/models/{id}` | OpenAI |
| POST | `/v1/chat/completions` | OpenAI Chat Completions, stream e non-stream |
| POST | `/v1/responses` | OpenAI Responses, stream e non-stream |
| POST | `/v1/completions` | Completions legacy OpenAI |
| POST | `/v1/messages` | Anthropic Messages |

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","stream":true,
       "messages":[{"role":"user","content":"Hello"}]}'
```

Host, porta, chiave API, CORS e i parametri di campionamento accettati per
richiesta sono documentati nel
[Riferimento di configurazione](#server-http-scheda-server).

## Inferenza distribuita

La modalità distribuita implementa il parallelismo di pipeline per intervalli
contigui di layer, modellato su `ds4_distributed.c`.

- Ogni **worker** possiede una fetta di layer, i suoi pesi e lo shard KV solo
  di quella fetta.
- Il **coordinatore** possiede embedding, campionamento, rendering del prompt
  ed esecuzione degli strumenti.
- Lo stato nascosto HC (`nHC x nEmbd` float, trasportato a 32/16/8 bit)
  attraversa i worker per ogni token.
- I worker partono PER PRIMI ma inattivi (nessun modello caricato): al momento
  della connessione il coordinatore partiziona i layer sulla lista dei peer,
  assegna a ogni worker il suo lavoro (GGUF, dimensione del contesto, budget
  di cache, fetta di layer — l'ultima fetta esegue anche la testa di output) e
  attende che ogni worker carichi e risponda pronto. Flash ha 43 layer e 256
  esperti; Pro ha 61 layer e 384 esperti. Il GGUF Pro Q2 completo usa gli
  stessi percorsi orizzontali e verticali guidati dalla geometria. I test di
  protocollo/partizione coprono Pro; la validazione numerica e prestazionale
  multi-Mac con modello reale resta in sospeso.

Il beneficio per nodo è la riduzione dell'I/O degli esperti: ogni worker fa
streaming di circa `1/N` degli esperti instradati, rendendo più probabile che
il working set caldo resti in RAM. Il prefill gira a chunk, per default 32
token per frame distribuito. L'inoltro opzionale worker-verso-worker può
ridurre i salti di rete, ma richiede l'indirizzo di ritorno LAN del
coordinatore. Tutte le impostazioni del cluster (lista dei peer, bit di
attivazione, inoltro e cosa trasporta l'ASSIGN) sono documentate nel
[Riferimento di configurazione](#distribuito-settings--scheda-worker).

Il protocollo v11 implementa il **parallelismo verticale degli esperti**: il
coordinatore esegue la dorsale densa completa e partiziona tutti gli esperti
instradati di ogni layer su maschere worker con prefisso di lunghezza (256/32
byte per Flash, 384/48 byte per Pro). Le raccolte da SSD dei worker possono
così girare in parallelo, al costo di circa un round-trip di rete per layer
instradato. Questa modalità richiede un collegamento cablato con RTT sotto
circa 1 ms; è intenzionalmente inadatta al Wi-Fi.
Vedi [`docs/INFERENZA-DISTRIBUITA.md`](docs/INFERENZA-DISTRIBUITA.it.md).

## Struttura del repository

```text
Makefile / Package.swift / project.yml
Sources/
  DS4Core/        GGUF, campionamento, ispezione dell'architettura e contratti
                  portabili; tokenizer e formati di chat posseduti dal modello
  DS4Metal/       runtime/kernel Metal condivisi più backend concreti dei modelli;
                  decoder DeepSeek, cache degli esperti e loader dei pesi GGUF
  DS4Engine/      selezione del backend, InferenceService, cache KV su disco,
                  strumenti, agenti, download dei modelli e runtime distribuito
  DS4Demo/        demo CLI: bring-up di Metal, audit GGUF, streaming di token
  DwarfStar/      app SwiftUI, una cartella di funzionalità per scheda
metal/            sorgente di verità dei kernel Metal
templates/        riscrittura Jinja commentata del template di chat del modello
scripts/          strumenti di analisi GGUF e helper per incorporare i kernel
docs/             guide di architettura, operatività, sviluppo e validazione
packaging/        assemblaggio del bundle .app e input per la firma
Tests/            test unitari e di parità
```

Ogni cartella di sorgenti, test e operativa ha un `README.md` locale che
spiega proprietà, dipendenze, regole di estensione e come si collega al resto
del sistema.

La mappa completa dei sorgenti e le regole di collocazione sono in
[`docs/STRUTTURA-PROGETTO.md`](docs/STRUTTURA-PROGETTO.it.md).
La matrice di supporto e le regole per aggiungere un backend sono in
[`docs/ARCHITETTURE-SUPPORTATE.md`](docs/ARCHITETTURE-SUPPORTATE.it.md).

## Demo CLI

```sh
swift run DS4Demo
swift run DS4Demo <model.gguf> 4
swift run DS4Demo <model.gguf> 32 "Explain RoPE briefly"
```

Sintassi completa:

```sh
swift run DS4Demo [gguf-path] [maxNew] [prompt]
```

| Argomento | Default | Significato |
|---|---|---|
| `gguf-path` | nessuno | File GGUF da aprire. Se omesso, la demo esegue solo il bring-up di Metal e il self-test della GPU. |
| `maxNew` | `4` | Numero di token da generare. `0` esegue solo lo smoke test forward a un token. |
| `prompt` | `"ciao come stai? rispondi in 1 parola"` | Prompt utente renderizzato attraverso il template di chat del modello. Poiché gli argomenti sono posizionali, passare `maxNew` prima di passare un prompt. |

La demo onora ogni variabile d'ambiente del motore elencata nel
[Riferimento di configurazione](#riferimento-di-configurazione) qui sotto; il
flusso di lavoro di profiling e la prosa sulle singole regolazioni vivono in
[`Sources/DS4Demo/README.md`](Sources/DS4Demo/README.it.md).
## Riferimento di configurazione

Tutto ciò che è configurabile nel progetto, in un unico posto. La
configurazione vive su tre livelli:

> Le regolazioni `DS4_*` esistenti di prestazioni e memoria descrivono il
> backend DeepSeek V4 operativo, salvo dove una riga dice esplicitamente che è
> comune. La GUI ora vincola quei controlli alla capacità del backend; Qwen
> riceverà impostazioni indipendenti quando il suo runtime sarà implementato.

1. **Impostazioni GUI** — persistite in `UserDefaults` e applicate dall'app.
   La maggior parte funziona esportando la variabile d'ambiente del motore
   corrispondente all'avvio e a ogni modifica, quindi la GUI è la fonte di
   verità per quelle regolazioni dentro l'app.
2. **Variabili d'ambiente del motore (`DS4_*`)** — lette direttamente dal
   motore. Le stesse regolazioni funzionano nell'app, nella demo CLI e nei
   test. In un processo puro (demo/test) valgono i default del motore
   riportati sotto; nell'app, le impostazioni GUI sovrascrivono le principali
   all'avvio.
3. **Configurazione locale ai pannelli** — server HTTP, cluster distribuito,
   server MCP, agenti e download dei modelli, ciascuno configurato nella
   propria scheda.

Salvo indicazione contraria, le regolazioni del motore vengono lette **al
caricamento del modello**: cambiarle, poi (ri)caricare il modello. Le
eccezioni ricaricabili a caldo sono `DS4_PREFILL_UNION`, `DS4_PREFILL_CHUNK` e
`DS4_PREFILL_ROUTE_BATCH`, che vengono rilette a ogni chiamata di prefill.

### Impostazioni GUI (scheda Settings)

#### Modello e motore

| Impostazione | Chiave `UserDefaults` | Default | Cos'è / cosa fa |
|---|---|---|---|
| Percorso del modello | `DS4ModelPath` | build di sviluppo: `<DS4_ROOT>/gguf/DeepSeek-V4-Flash-…-imatrix.gguf`; bundle dell'app: vuoto | Il GGUF che il motore condiviso carica. **Scarica…** seleziona un modello del catalogo eseguibile, conservato in Application Support senza bisogno di un bookmark sandbox. Per un modello esterno usare **Browse**: il selettore valida il supporto del runtime e concede un bookmark security-scoped (persistito come `ds4.modelBookmark` / `ds4.modelDirBookmark`). |
| Dimensione del contesto | `DS4ContextSize` | a scaglioni di RAM: 4096 (<24 GB), 8192 (<80 GB), 32768 (oltre) | Massimo numero di posizioni KV (`maxKeys`). Le cache KV e gli scratch scalano con esso — un contesto da 1M alloca decine di GB di KV e affama la page cache degli esperti, motivo per cui il default segue la RAM. Lo stepper arriva comunque fino a 1M per le macchine che possono permetterselo. |
| Modalità | `DS4EngineMode` | `Local` | `Local` esegue il singolo motore in-process; `Distributed` instrada chat/benchmark attraverso il coordinatore configurato nella sezione Distribuito. |
| Thinking (interruttore in Chat) | non persistito | off | Abilita la gestione dei token di ragionamento per il turno successivo: la catena di pensiero del modello va in streaming in una sezione richiudibile. |

#### Campionamento (chat)

| Impostazione | Chiave `UserDefaults` | Default | Cos'è / cosa fa |
|---|---|---|---|
| Temperatura | `DS4Temperature` | `0.6` | Temperatura della softmax. Più bassa = più concentrata e meno deriva — aiuta sul modello a 2 bit quantizzato in modo aggressivo. |
| Penalità di ripetizione | `DS4RepPenalty` | `1.1` | Valori >1 scoraggiano le ripetizioni e rompono il collasso in loop di ripetizione in cui cadono i modelli quantizzati. `1.0` = disattivata. |
| top-k | fisso | `40` | Non esposto nella GUI. Campionare sull'intero vocabolario da 129k di DeepSeek permette a un singolo token rumoroso di coda di trascinare un'intera risposta in cinese a temperature normali; il tetto a 40 taglia la coda senza danneggiare la varietà. (La demo CLI è greedy per default ed espone le proprie regolazioni env del sampler; il server HTTP accetta valori per richiesta.) |

#### Interruttori memoria / I/O

Ogni interruttore persiste una chiave `UserDefaults` ed esporta la variabile
del motore indicata; tutti si applicano al **successivo caricamento del
modello**. I default sono il profilo veloce misurato per 16 GB (una
migrazione una tantum allinea le installazioni vecchie; il pulsante
**Align to fast demo config** lo riapplica).

Provenienza dei benchmark: salvo diversa indicazione nella riga, le cifre
comparative di questo preset sono state misurate su un **MacBook Pro M1 Pro
con 16 GB di memoria unificata**, usando il modello DeepSeek-V4 Flash 2-bit
imatrix. Il preset base è stato validato il **2026-07-13** e la sua cache
mixed-quant il **2026-07-16**. Sono misurazioni di riferimento, non garanzie
per un chip, un SSD, una pressione di memoria, una build del modello o un
prompt diversi.

| Impostazione | Chiave `UserDefaults` | Variabile del motore | Default app | Cos'è / cosa fa |
|---|---|---|---|---|
| Slot cache esperti | `DS4ExpertCacheSlots` | `DS4_EXPERT_CACHE_SLOTS` | `22` | Cache GPU LRU per layer per gli esperti instradati, ~6.9 MB wired per slot per layer sul modello a 2 bit. `0` = disattivata. Il confronto del 2026-07-13 su M1 Pro ha trovato 22 più veloce di 20 senza collasso di memoria osservato (circa 70% di hit-rate); senza `mlock` o il preset Q4 completo, pool più piccoli possono essere più sicuri. |
| Cache esperti mixed-quant | `DS4MultiQuantCache` | `DS4_MULTI_QUANT_CACHE` | on | I pool consapevoli del layer usano la reale dimensione dei record IQ2/Q4 di ogni layer instradato condividendo il budget totale in byte della cache storica. Sul GGUF Flash 37-IQ2/6-Q4, l'A/B esatto del 2026-07-16 ha migliorato il decode 2.11→2.72 tok/s (+28.9%) e ridotto le letture di esperti 930.1→640.4 MB/token (−31.1%) a parità di cache pianificata da 5.37 GiB; 64 token e 2,068,480 logits sono risultati identici bit a bit. Disattivata ripristina il precedente percorso a classe singola con bypass fuori classe. |
| Cache KV su disco | `DS4DiskKV` | — (API del servizio) | on | Fa checkpoint su disco delle generazioni completate e ripristina i prefissi corrispondenti agli avvii a freddo, così ricaricare l'app non ri-esegue il prefill delle vecchie conversazioni. |
| Budget KV su disco | `DS4DiskKVBudgetKTok` | — (API del servizio) | `1000` (= 1M token) | Budget totale dei checkpoint in migliaia di token su tutte le conversazioni salvate (~22 KB/token sul modello a 2 bit, quindi 1M ≈ 22 GB su disco). La finestra viva resta `contextSize`. |
| Ring raw-KV | `DS4RawRing` | `DS4_RAW_RING` | on | Mantiene il KV grezzo solo per la finestra di sliding-attention a 128 righe invece dell'intero contesto, rendendo costante la RAM del raw-KV. È un `MTLBuffer` in memoria unificata condivisa, **non** una KV cache su disco. Una finestra avvolta viene riordinata e convertita F32→F16 da un solo dispatch GPU. La GUI lo tiene abilitato come vincolo fisso a bassa RAM; non riduce il KV compresso. |
| Read-ahead esperti | `DS4WillNeed` | `DS4_WILLNEED_EXPERTS` | on | `madvise(WILLNEED)` esattamente sugli esperti appena selezionati dal router, prima della raccolta. Non speculativo, numerica invariata. |
| Letture dirette esperti | `DS4ExpertPread` | `DS4_EXPERT_PREAD` | on | Legge gli slab degli esperti con `pread`+`F_NOCACHE` direttamente nei buffer di destinazione, aggirando la page cache così che ~1 GB/token di ricambio degli esperti smetta di sfrattare i pesi densi. Misurato ~+20% tok/s da solo sulla macchina di riferimento da 16 GB. Il fallback della proprietà pura è vincolato alla RAM, ma il preset veloce corrente lo abilita esplicitamente. |
| Streaming dei pesi densi | `DS4DenseStream` | `DS4_DENSE_STREAM` | on | Fa streaming dei tensori densi di ogni layer attraverso un ring di staging invece di tenerne ~6 GB residenti, sovrapponendo il lavoro di SSD e GPU. Numerica identica; il preset veloce corrente lo abilita esplicitamente. |
| Pinning dei buffer caldi | `DS4MLock` | `DS4_MLOCK` | on | `mlock()` best-effort dei buffer residenti caldi (pool degli esperti, testa residente, staging denso e, quando abilitato, ring raw-KV; ~3.3 GB ai default). Impedisce a macOS di comprimere i buffer tra un token e l'altro (misurato −38% ms/token su 16 GB). |
| Cache attention Q4 | `DS4DenseQ4` | `DS4_DENSE_Q4` | on | **Lossy.** Riquantizza le tre gigantesche proiezioni di attention Q8→Q4_K al caricamento e le mantiene residenti (~1.4 GB), togliendo ~4.6 GB/token dallo stream SSD. L'output greedy può divergere pur restando coerente. Richiede lo streaming denso; il primo caricamento scrive una cache `.q4dense`. |
| Cache q/kv Q4 | `DS4QkvQ4` | `DS4_QKV_Q4` | on | **Lossy; richiede `DS4_DENSE_Q4`.** Riquantizza anche le rimanenti proiezioni di attention q_a e kv, mantenendole residenti e togliendo circa altri 0.7 GB/token dallo stream denso. |
| Cache FFN condivisa Q4 | `DS4SharedQ4` | `DS4_SHARED_Q4` | on | **Lossy; richiede `DS4_DENSE_Q4`.** Riquantizza le proiezioni gate/up/down dell'esperto condiviso e le mantiene residenti, lasciando più banda SSD alle raccolte degli esperti instradati. |
| Expert bundle | `DS4ExpertBundle` | `DS4_EXPERT_BUNDLE` | on | File sidecar con gli slab gate/up/down di ogni esperto contigui: un miss di cache diventa una singola lettura sequenziale da ~7 MB invece di tre sparse (raccolta misurata 2.7 → 4.8 GB/s). Stessi byte, stessa numerica. Costruito una volta accanto al GGUF (o in Application Support); servono ~73 GB liberi e viene saltato con un log quando lo spazio scarseggia. Un pulsante in Settings lo costruisce/verifica su richiesta. |
| Caricamento esperti MetalIO | `DS4MetalIO` | `DS4_MTLIO` | on | Carica intervalli dell'expert bundle direttamente nei buffer Metal della slot-cache durante il decode. L'app esporta `DS4_MTLIO_MIN_GBS=4.0`: due finestre sostenute da 64 MiB sotto 4 GB/s fanno scattare il fallback a `pread`. Il prefill continua a usare `pread` parallelo. |
| Profiling route/attn | `DS4ProfileRoute` | `DS4_PROFILE_ROUTE` | off | Divide `route/attn` in cinque sottofasi temporizzate nel log del motore. Solo diagnostico: le sync extra rallentano la generazione, quindi leggere i rapporti, non i tok/s assoluti. |
| Union esperti in prefill | `DS4PrefillUnion` | `DS4_PREFILL_UNION` | `256` | Massimo di esperti raggruppati per lettura layer-major in prefill. Ricaricabile a caldo; il benchmark rapido confronta 192/256 e quello completo misura anche 64. Il preset dell'app è 256; il default indipendente del motore puro resta 192. |
| Chunk di prefill | `DS4PrefillChunk` | `DS4_PREFILL_CHUNK` | `512` | Token per chunk di prefill. Ogni chunk ricarica i pesi densi di tutti i layer, quindi chunk più grandi ammortizzano quel costo fisso sui prompt lunghi. Ricaricabile a caldo; confrontato in benchmark con 1024. |
| Route batch di prefill | `DS4PrefillRouteBatch` | `DS4_PREFILL_ROUTE_BATCH` | `32` | Token i cui dispatch di route/attention condividono un solo command buffer. Ricaricabile a caldo; il benchmark completo misura 16/32/64/128 e richiede un miglioramento >2% prima di sostituire il valore persistito. |

Altre otto regolazioni sono persistite senza interruttore GUI (impostarle via
`defaults write` per gli esperimenti; l'auto-tune qui sotto le esplora
automaticamente):

| Chiave `UserDefaults` | Variabile del motore | Default app | Cos'è / cosa fa |
|---|---|---|---|
| `DS4ExpertCacheUniform` | `DS4_EXPERT_CACHE_UNIFORM` | off | Disattivata ridistribuisce il budget fisso in byte della cache a partire dal profilo d'uso congelato; attivata assegna lo stesso numero di slot a ogni layer instradato. Il tuner rigoroso confronta entrambe le politiche solo quando è disponibile un seed d'uso. |
| `DS4ExpertLookahead` | `DS4_EXPERT_LOOKAHEAD` | `0` | Look-ahead speculativo della slot-cache: mentre il layer *i* calcola, pre-riempie il pool del layer *i+1* con i suoi top-N esperti secondo il prior d'uso. `0` = vengono prefetchati solo i layer 0–2 instradati via hash (la cui selezione è esatta). |
| `DS4PreadSplit` | `DS4_PREAD_SPLIT` | `4` | Numero di range concorrenti di `pread` degli esperti. Cambia la profondità della coda NVMe senza cambiare i byte dei tensori; l'ottimo dipende dall'SSD, quindi il tuner spazza l'intera piccola griglia. |
| `DS4DenseAhead` | `DS4_DENSE_AHEAD` | `2` | Profondità di read-ahead del ring di staging (1–3) per lo streaming denso. `2` tiene in volo i layer i+1 e i+2 (+1.5% misurato, ~150 MB per slot extra). |
| `DS4AsyncFFN` | `DS4_ASYNC_FFN` | on | Pipeline asincrona della FFN instradata (+10% misurato, output identico al token — correttezza garantita dalla coda Metal in-order). `false` è il paracadute di debug che ripristina le attese sincrone. |
| `DS4Q8NSG` | `DS4_Q8_NSG` | `4` | Simdgroup per threadgroup per i matvec densi Q8_0. Cambia l'occupancy e la partizione delle somme parziali in virgola mobile, quindi i bit bassi possono differire; 4 è il migliore su M1 Pro, mentre GPU più larghe (Max/Ultra) possono preferire 6–8. Per questo il tuner GUI a qualità esatta lo lascia fisso; solo il tuner di processo con `--allow-numeric` lo esplora. |
| `DS4MoeNSG` | `DS4_MOE_NSG` | `4` | Simdgroup per threadgroup per i kernel id MoE/Q4 instradati. Partiziona le righe di output anziché la riduzione K; l'occupancy dipende dalla GPU, quindi l'auto-tune lo misura indipendentemente dall'NSG Q8. |
| `DS4DenseQ4NSG` | `DS4_DENSE_Q4_NSG` | `4` | Occupancy indipendente di partizione delle righe per i kernel densi Q4. L'app lo persiste separatamente; i processi con motore puro ereditano ancora `DS4_MOE_NSG` quando questa variabile non è impostata esplicitamente. |

I pesi densi residenti (`DS4_RESIDENT_DENSE`) **non hanno interruttore GUI**:
l'app li abilita automaticamente sulle macchine con ≥24 GB e cancella
qualsiasi valore persistito obsoleto (su 16 GB rallentano l'app reale; lo
streaming denso ha comunque la precedenza). La regolazione env resta per la
demo.

Esempio — cambiare una regolazione nascosta su una build di sviluppo, poi
ricaricare il modello nell'app:

```sh
defaults write com.dwarfstar.app DS4DenseAhead -int 3     # build xcodegen/Xcode
defaults write org.ds4.dwarfstar DS4DenseAhead -int 3     # bundle di `make app`
```

#### Benchmark e auto-tune (pulsanti in Settings)

- **Benchmark di prefill — rapido** (~3 min): riscalda il motore caricato, poi
  confronta `DS4_PREFILL_UNION=192/256` su un contesto di 256 token con 4
  token di decode. Esclude intenzionalmente 64 perché il rumore delle
  esecuzioni brevi può favorire un valore noto per causare riletture eccessive
  a lunghezze di prompt realistiche.
- **Benchmark di prefill — completo** (~15 min): confronta union 64/192/256 su
  512 token di contesto con 8 token di decode; confronta chunk 512/1024 su
  1024 token di contesto; poi confronta route batch 16/32/64/128 su 512 token
  di contesto. Un vincitore di route batch sostituisce il valore corrente solo
  quando è più veloce di oltre il 2%. Entrambe le modalità applicano e
  persistono i loro vincitori. La chat non è disponibile durante l'esecuzione
  perché il motore è un actor seriale.
- **Auto-tune della macchina da record** (può comunque richiedere ore): esegue
  fino a due passate di coordinate-ascent ma misura ogni configurazione esatta
  al massimo una volta. La radice calda già caricata viene misurata con una
  sonda che ripristina il prompt; le visite successive sono hit di cache senza
  ricaricare il modello. Il campione valido completo con la mediana più alta
  di velocità di decode greedy è il record di confronto — decode, prefill,
  qualità e metriche di risorse non vengono mai mescolati tra esecuzioni.
  Sostituirlo richiede un risultato di decode strettamente superiore, token
  generati esatti e hash dei logit sull'intero vocabolario rispetto alla
  radice immutabile, prefill entro −8%, stabilità tail/head ≥0.75, il
  pavimento di RAM immutabile e non più di 128 MiB di swapout stazionario. Gli
  slot della cache esperti usano una scala direzionale che sale per prima:
  dopo una vittoria viene misurato solo il valore di manifesto immediatamente
  superiore, e la prima sconfitta chiude quel parametro senza testare valori
  maggiori; il vicino inferiore viene provato solo se il passo iniziale verso
  l'alto perde. In modalità vincolata a bassa RAM vengono esplorati solo
  candidati neutri o riduttivi per la memoria. I valori candidati restano
  locali al processo; gli UserDefaults vengono scritti solo dopo che il
  vincitore è installato, il suo agente attivo è riscaldato e la sonda finale
  di swap da 128 MiB con ripristino del prompt riesce. Una chat non vuota
  viene marcata per il re-prefill della cronologia dopo ciascuna delle due
  sonde. Una transazione durevole ripristina lo snapshot iniziale completo
  dopo un'interruzione. I report usano lo schema 7 ed etichettano questo
  metodo `cached-high-water`. Evitare le ripetizioni elimina la stima del
  rumore bilanciata per ordine: un record fortunato può rendere le promozioni
  successive conservativamente più difficili, ma non può ammettere un
  candidato che manchi i cancelli di qualità, RAM, stabilità, prefill o swap.

Per l'equivalente ricerca isolata per processo su più regolazioni
interagenti, usare
[`scripts/metal_autotune.py`](scripts/metal_autotune.py) e il
[`runbook di autotuning multi-parametro`](docs/AUTOTUNING-METAL.it.md). Confronta
i logit sull'intero vocabolario, conferma le prestazioni in ordine ABBA, fa
checkpoint di ogni decisione ed emette un `final-env.sh` sourceable. La GUI
segue gli stessi principi di promozione con ricaricamenti in-process
completamente attesi e, allo stesso modo, non promuove modifiche lossy del
modello.

#### Posizioni su disco

| Percorso | Contenuto |
|---|---|
| `~/Library/Application Support/DwarfStar/kv-cache/` | Checkpoint disk-KV (chat + server HTTP). |
| `~/Library/Application Support/DwarfStar/models/` | GGUF del catalogo scaricati dalla GUI; i trasferimenti incompleti usano `<name>.part` e metadati di ripresa. |
| `~/Library/Application Support/DwarfStar/q4-cache/` | Cache di riquantizzazione `.q4dense`: ~1.4 GB per il trio denso-Q4 di base e più grandi quando sono incluse le proiezioni QKV/condivise. L'app sandboxed non può scrivere accanto al GGUF. |
| `~/Library/Application Support/DwarfStar/expert-bundle/` | Sidecar expert-bundle costruiti quando la cartella del modello non è scrivibile (decine di GB). |
| `~/Library/Application Support/DwarfStar/autotune/<timestamp>/` | `report.md` e `results.json` schema-7 dell'auto-tune GUI da record, incluse le prove accettate, in cache e respinte. |
| `~/Library/Application Support/DwarfStar/dist-models/` | File ricevuti via trasferimento distribuito (GGUF, sidecar), indicizzati per manifest SHA-256. |
| `~/Library/Application Support/DwarfStar/github-projects/` | Repository importati dallo strumento `github_clone`, una cartella `<owner>-<repo>` ciascuno (sostituita al re-clone). |
| `~/Library/Application Support/DwarfStar/expert-usage-<model>-<agent>.json` | Imatrix d'uso per modello e per agente (cronologia di instradamento usata per pre-riscaldare la cache degli esperti). |
### Variabili d'ambiente del motore (`DS4_*`)

I default qui sotto sono i **default del motore per un processo puro** (demo
CLI, test). Dentro l'app, le impostazioni GUI di cui sopra possiedono le
regolazioni sovrapposte. Le regolazioni "on"/"off" marcate `=0 disabilita`
sono attive per default. Ogni riga dichiara il proprio comportamento numerico:
le modifiche di I/O/layout possono preservare i valori, i percorsi di fusione
o MM possono cambiare l'ordine di accumulazione in virgola mobile, e la
riquantizzazione o un numero ridotto di esperti attivi sono deliberatamente
lossy. Assumere l'identità bit a bit solo dove è dichiarata. Le indicazioni di
flusso di lavoro ("quale regolazione provare per prima") sono in
[`Sources/DS4Demo/README.md`](Sources/DS4Demo/README.it.md).

#### Percorsi e integrazione

| Variabile | Valori / default | Cos'è / cosa fa |
|---|---|---|
| `DS4_ROOT` | percorso; default `/Users/oppog/Downloads/ds4-main` | Radice del progetto in modalità sviluppo: dove le build non pacchettizzate cercano i modelli in `gguf/` e il percorso del modello predefinito. Irrilevante per una `.app` pacchettizzata. |
| `DS4_USAGE_FILE` | percorso o `off`; default `<gguf>.usage.json` | Posizione demo/CLI della imatrix d'uso (le scelte storiche di esperti del router, usate per pre-riscaldare la cache). L'app la ignora e mantiene file per modello/per agente in Application Support. `off` = esecuzione a freddo. |
| `DS4_DEMO_CONTEXT` | int `1024...1000000`; default `4096` | Capacità di contesto configurata solo per la demo (`maxKeys`). Usare lo stesso prompt corto con due valori per misurare l'overhead di capacità mentre il KV vivo resta ugualmente vuoto. |
| `DS4_DEMO_LIVE_CONTEXT` | conteggio esatto di token; non impostata | Frontiera di prefill sintetica solo per la demo. Crea un contesto incorniciato di esattamente N token da token di prompt ordinari affiancati, consentendo misurazioni a 10K o altre dimensioni di KV vivo senza un corpus esterno. |
| `DS4_PROMPT_FILE` | percorso di file; non impostata | Solo demo: legge il prompt da questo file UTF-8, scavalcando il prompt posizionale. Equivalente a un `@/path/file` posizionale, ma evita lo splitting degli argomenti di shell per i prompt lunghi. `~` viene espansa. |
| `DS4_Q4_CACHE_DIR` | directory; default: accanto al GGUF | Dove le cache di riquantizzazione `.q4dense` vengono lette/scritte. L'app la imposta ad Application Support (la sandbox non può scrivere accanto al modello). |
| `DS4_BUNDLE_DIR` | directory; default: accanto al GGUF | Directory di fallback per il sidecar `.expbundle` (viene sempre provato prima un file fratello del GGUF). L'app la imposta ad Application Support. |
| `DS4_SEARCH_URL` | template di URL contenente `%@`; default `https://html.duckduckgo.com/html/?q=%@` | Endpoint usato dallo strumento integrato `web_search`; la query viene percent-encoded in `%@` e i risultati vengono estratti da HTML in stile DuckDuckGo. |
| `HF_TOKEN` | token; default: `~/.cache/huggingface/token` | Token Hugging Face usato dal downloader dei modelli per i download autenticati (processi puri: demo/test). Nell'app, un token salvato in Settings → Hugging Face (Keychain) ha precedenza su questa variabile. |

#### Diagnostica (orientata alla demo)

| Variabile | Valori / default | Cos'è / cosa fa |
|---|---|---|
| `DS4_TYPES_ONLY` | `=1`; off | Modalità audit GGUF: stampa i dtype dei tensori critici, gli id speciali del tokenizer e la tokenizzazione del prompt, poi esce prima di costruire il decoder. |
| `DS4_DIAG` | `=1`; off | Diagnostica completa dello streaming: stampa le regolazioni attive, misura la banda dell'SSD, poi riporta l'instradamento per layer, la concentrazione degli esperti, l'allocazione della cache e la banda di raccolta rispetto al tetto dell'SSD. |
| `DS4_WARMUP` | int ≥0; `0` (`min(4,maxNew-1)` con DIAG) | Esclude i primi N token generati dal profilo di decode (costi di cache fredda e di cablaggio). |
| `DS4_PROFILE_ROUTE` | `=1`; off | Divide `route/attn` in sottofasi temporizzate (compressore, Q/KV, attention, output). Aggiunge overhead di sync — leggere i rapporti, non i tok/s. |
| `DS4_PROMPT_MAX_CHARS` | int; `12000` | Solo demo: limite di troncamento per i prompt `@/path/file` affinché prompt più generazione stiano in `DS4_DEMO_CONTEXT`. |

#### Campionamento della demo e decode auto-speculativo

Queste variabili riguardano solo `DS4Demo`. Il default resta greedy così le
esecuzioni storiche di prestazioni restano comparabili.

| Variabile | Valori / default | Cos'è / cosa fa |
|---|---|---|
| `DS4_DEMO_TEMPERATURE` | float; `0` | Temperatura di campionamento. `0` seleziona l'argmax greedy; valori sopra zero abilitano il campionamento. |
| `DS4_DEMO_TOP_K` | int; `0` | Top-k della demo. `0` significa il percorso a vocabolario intero del sampler; normalmente si abbina un valore finito a una temperatura >0. |
| `DS4_DEMO_TOP_P` | float; `1` | Soglia nucleus usata durante il campionamento. |
| `DS4_DEMO_MIN_P` | float; `0` | Soglia min-p relativa usata durante il campionamento. |
| `DS4_DEMO_REPEAT_PENALTY` | float; `1` | Penalità di ripetizione; valori >1 penalizzano gli id emessi di recente. |
| `DS4_DEMO_REPEAT_LAST_N` | int ≥0; `64` | Numero di id recenti considerati dalla penalità di ripetizione. |
| `DS4_SPEC_K` | int; non impostata/`0`/`1` = off, ≥2 abilita | Finestra greedy auto-speculativa. I candidati di draft vengono riavvolti e verificati in un unico batch a configurazione completa; viene disabilitata automaticamente quando sono attivi il campionamento a temperatura o la penalità di ripetizione. |
| `DS4_SPEC_DRAFT` | `experts` (default) / `ngram` | Sorgente del draft per la finestra speculativa. `experts` è lo storico draft del modello a esperti ridotti. `ngram` è il prompt-lookup: i candidati vengono copiati dall'occorrenza più recente del suffisso corrente nella trascrizione — zero forward di draft, i round degradano a un forward normale quando nulla corrisponde. Stessa verifica batch a configurazione completa e stessa parità greedy in entrambi i casi. |
| `DS4_SPEC_DRAFT_EXPERTS` | int; `2` | Esperti instradati attivi usati dalla passata di draft (solo `DS4_SPEC_DRAFT=experts`), vincolati ad almeno 1 e sotto il numero completo di esperti configurato. Cambia il costo del draft, non la regola di verifica a configurazione completa. |
| `DS4_SPEC_VERIFY_BATCH` | `=0` disabilita; on | Verifica speculativa a batch (fase V1): il route/attention della finestra va in UN solo command buffer per layer e le FFN instradate vengono servite dalla slot-cache degli esperti per token — una sync per layer invece di una per token. Stessi dispatch nello stesso ordine per token: identico al token. I layer con l'indexer NSA attivo nella finestra o con esperti attivi ridotti ricadono sul percorso per token; anche i quant fuori classe ricadono, a meno che `DS4_MULTI_QUANT_CACHE=1` non fornisca il loro pool dimensionato correttamente. `=0` ripristina la storica verifica per token per gli A/B. |
| `DS4_MTP_GGUF` | percorso, o `=1`; non impostata | Fase speculativa MTP M1 (solo diagnostica): apre un GGUF sidecar MTP fornito esplicitamente (id accessorio interno `mtp`), stampa l'inventario dei tensori e valida l'interfaccia di draft rispetto a vocab/nEmbd del modello principale. `=1` cerca `*MTP*.gguf` accanto al modello principale. MTP non è mostrato nel catalogo GUI dei modelli principali e non influenza ancora il decode. |

#### Streaming e cache degli esperti

| Variabile | Valori / default | Cos'è / cosa fa |
|---|---|---|
| `DS4_ACTIVE_EXPERTS` | `1...6`; `6` | **Cambia l'output.** Usa meno esperti instradati per token dei 6 addestrati: meno I/O e tempo di raccolta a un costo di qualità. Modalità degradata a bassa RAM / sonda del costo di I/O degli esperti. |
| `DS4_EXPERT_CACHE_SLOTS` | int; `0` (off) | Cache GPU LRU per layer per gli esperti (~6.9 MB wired per slot per layer a 2 bit). 8 è il minimo effettivo; il preset dell'app usa 22. |
| `DS4_EXPERT_CACHE_UNIFORM` | `=1`; off | Disabilita la ridistribuzione degli slot guidata dall'uso (per default i layer con instradamento concentrato ricevono più slot a parità di budget). Per gli A/B. |
| `DS4_MULTI_QUANT_CACHE` | `=1`; off | A/B della cache mixed-quant: crea pool IQ2/Q4 dimensionati correttamente entro il budget totale effettivo in byte della cache storica. Numerica esatta; off ripristina il precedente bypass fuori classe. Il percorso distribuito a shard di esperti resta sul layout di cache storico. |
| `DS4_EXPERT_LOOKAHEAD` | int; `0` | Pre-riempimento speculativo del pool del layer i+1 con i suoi top-N esperti da prior d'uso durante il calcolo del layer i. I layer hash 0–2 vengono sempre prefetchati esattamente. Richiede la slot cache. |
| `DS4_EXPERT_PREAD` | `=1`; off | Letture di esperti con `pread`+`F_NOCACHE` che aggirano la page cache, impedendo al ricambio degli esperti di sfrattare i pesi densi. Spesso la singola regolazione migliore su 16 GB. |
| `DS4_PREAD_SPLIT` | `1...8`; `1` | Con l'expert pread: divide ogni slab in N letture di range concorrenti allineate a 16 KB per aumentare la profondità della coda NVMe (il disco raggiunge il picco solo con ~24 richieste in volo). Stessi byte. |
| `DS4_EXPERT_BUNDLE` | `=1`; off | Costruisce/riusa il sidecar `<gguf>.expbundle` con slab per esperto contigui: una lettura sequenziale da ~7 MB per miss invece di tre sparse. Duplica la regione degli esperti su disco (decine di GB). |
| `DS4_MTLIO` | `=1`; off | Caricamento veloce di risorse Metal dall'expert bundle direttamente nelle destinazioni `MTLBuffer` della slot-cache durante il decode. Il prefill resta deliberatamente sul consolidato percorso `pread` parallelo. Sperimentale: fare benchmark contro pread su ogni Mac. |
| `DS4_MTLIO_MIN_GBS` | float; `1.5` | Interruttore di sicurezza MetalIO: i tempi vengono aggregati in finestre da 64 MiB e due finestre consecutive sotto questa banda fanno scattare il fallback permanente a `pread`. Piccoli batch e stalli isolati non disabilitano più MetalIO. Il preset dell'app esporta esplicitamente `4.0`; il default del motore puro resta 1.5. |
| `DS4_POOL_INTERLEAVE` | `=0` disabilita; on | Layout dei pool della slot-cache con gate/up/down contigui per slot, così un miss del bundle è UNA sola pread. `=0` ripristina lo storico layout a 3 buffer per i controlli di parità. |
| `DS4_WILLNEED_EXPERTS` | `=0` disabilita; on | `madvise(WILLNEED)` esattamente sugli esperti selezionati dal router, subito prima della raccolta. Non speculativo. |
| `DS4_PREFETCH` | `=1`; off | Read-ahead `madvise` dei pesi non instradati del layer SUCCESSIVO durante il calcolo del layer corrente. Può aiutare o nuocere a seconda della saturazione dell'SSD. |
| `DS4_PREFETCH_EXPERTS` | int; `0` | Con `DS4_PREFETCH=1`: prefetcha anche N esperti probabili dal prior d'uso. Speculativo. |

#### Pesi densi, attention e KV

| Variabile | Valori / default | Cos'è / cosa fa |
|---|---|---|
| `DS4_DENSE_STREAM` | `=1`; off | Fa streaming dei pesi densi per layer attraverso un ring a 2 slot `pread`+`F_NOCACHE`, un layer in anticipo (~300 MB di staging invece di ~6 GB residenti). Numerica identica; ha precedenza su `DS4_RESIDENT_DENSE`. |
| `DS4_DENSE_AHEAD` | `1...3`; `1` | Profondità di read-ahead del ring di staging denso (richiede lo streaming denso). Ogni slot extra costa ~150 MB e contende con l'I/O degli esperti sullo stesso disco. |
| `DS4_RESIDENT_DENSE` | `=1`; off | Copia ~5 GB di pesi non-esperti in buffer wired invece di viste mmap sfrattabili. Aiuta su 24/32 GB; può nuocere su 16 GB. |
| `DS4_RESIDENT_COMP` | `=0` disabilita; on | Mantiene le quattro proiezioni del compressore NSA residenti invece che in streaming. Vengono lette a ogni token su 41 dei 43 layer Flash e su tutti i 61 layer Pro; la cifra di ~0.6 GB è specifica di Flash. `=0` è il fallback per RAM strette. |
| `DS4_COMP_Q8` | `=1`; off (richiede compressori residenti) | **Lossy, sperimentale.** Riquantizza le quattro proiezioni residenti del compressore F16→Q8_0, dimezzando circa la loro RAM e le letture GPU per token. La prima esecuzione scrive una cache validata `.q8comp.Lx-y`; confrontare la qualità prima di abilitarla in modo permanente. |
| `DS4_LAZY_IDX` | `=0` disabilita; on | Tiene le proiezioni di scoring dell'indexer fuori dallo stream denso per token finché il contesto **usato** non raggiunge il confine dello sparse top-K, poi le carica una volta sola in buffer residenti. Evita ~360 MB/token di traffico SSD prematuro anche quando il contesto configurato è grande. Lossless; `=0` ripristina lo storico comportamento sempre in streaming. |
| `DS4_MLOCK` | `=1`; off | `mlock()` best-effort dei buffer residenti caldi (pool degli esperti, testa residente, staging denso e il ring raw-KV se abilitato). Impedisce al compressore di memoria di macOS di comprimere buffer usati una volta per token. |
| `DS4_DENSE_Q4` | `=1`; off (richiede `DS4_DENSE_STREAM=1`) | **Lossy.** Riquantizza le tre gigantesche proiezioni di attention Q8→Q4_K, le mantiene residenti (~1.4 GB), togliendo ~4.6 GB/token di traffico SSD. Il primo caricamento scrive `<gguf>.q4dense` (o `DS4_Q4_CACHE_DIR`); cancellarlo per forzare una nuova riquantizzazione. |
| `DS4_QKV_Q4` | `=1`; off (richiede `DS4_DENSE_Q4=1`) | **Lossy.** Riquantizza anche le proiezioni q_a e kv a Q4_K e le mantiene residenti, togliendo dallo stream denso i restanti slab medi di attention. |
| `DS4_SHARED_Q4` | `=1`; off (richiede `DS4_DENSE_Q4=1`) | **Lossy.** Riquantizza anche le proiezioni FFN dell'esperto condiviso a Q4_K e le mantiene residenti, liberando banda del disco per la raccolta degli esperti. |
| `DS4_RAW_RING` | `=1`; off | KV grezzo mantenuto in un ring `nSWA` (128 righe) in memoria Metal condivisa: memoria raw-KV costante per contesti lunghi. Non è sostenuto da SSD ed è indipendente dai checkpoint Disk KV. Quando la finestra cronologica avvolge, un solo dispatch GPU 2D la riordina e la converte F32→F16. Non riduce il KV compresso. |

#### Prefill

| Variabile | Valori / default | Cos'è / cosa fa |
|---|---|---|
| `DS4_PREFILL_CHUNK` | int; `512` | Token per chunk di prefill (ricaricabile a caldo). Chunk più grandi ammortizzano sui prompt lunghi il ricaricamento completo dei pesi densi per chunk. |
| `DS4_PREFILL_UNION` | int ≥6; `192` | Massimo di esperti raggruppati per lettura layer-major di prefill (ricaricabile a caldo). Governa i byte/token del prefill; `64` ha causato riletture eccessive nel benchmark di riferimento. Questo è il **default del motore puro**; il preset corrente dell'app esporta 256. Costa circa 1.3 GB ×2 transitori alle union più grandi. |
| `DS4_PREFILL_FFN_BATCH` | `=0` disabilita; on | Codifica tutte le FFN dei token di un gruppo in un solo command buffer Metal invece di uno per token (la sync per token era un costo di prefill dominante). Numerica identica. |
| `DS4_PREFILL_ROUTE_BATCH` | int; `32` (`0`/`1` off) | Raggruppa il route/attention di fino a N token consecutivi in un solo command buffer, tagliando le sync della fase A di N volte. Numerica identica; i token con indexer attivo ricadono sul percorso per token. |
| `DS4_GPU_INDEXER_TOPK` | `=0` disabilita; on | Mantiene punteggio dell'indexer NSA a contesto lungo → maschera esatta top-K → attention in un solo command buffer GPU. Rimuove un readback/attesa CPU per ogni layer con indexer attivo; `=0` ripristina l'heap CPU per le diagnostiche di parità. |
| `DS4_PREFILL_MM` | `=1`; off (opt-in) | **Cambia l'ordine di accumulazione** (vicino ma non identico bit a bit). FFN di prefill instradata + condivisa attraverso kernel matrice-matrice: i pesi degli esperti vengono letti una volta per tile per tutti i token del gruppo. Questo percorso quantizzato ottimizzato resta limitato alle forme Flash; tenerlo off per Pro Q2 così viene usato il fallback matvec supportato. |

#### Kernel e numerica

| Variabile | Valori / default | Cos'è / cosa fa |
|---|---|---|
| `DS4_FUSED_MOE` | `=0` disabilita; on | Kernel MoE fusi (2 dispatch invece di 5). `=0` seleziona il percorso non fuso per gli A/B numerici; l'arrotondamento può differire. |
| `DS4_FUSED_HC` | `=0` disabilita; on | Coda HC-reduce fusa (split+collapse+RMSNorm in un solo dispatch, ~170 dispatch/token risparmiati). Stessa matematica, differenza di ±1 ulp nell'ordine di riduzione. |
| `DS4_DENSE_Q4_KERNEL` | `=0` disabilita; on | Matvec Q4_K dedicato a matrice singola per le proiezioni dense/condivise residenti. Usa la stessa dequantizzazione e riduzione dello storico wrapper MoE `k=1`, ma rimuove il lavoro di expert-ID/stride. `=0` ripristina il wrapper per gli A/B di parità bit a bit. |
| `DS4_FUSED_ROUTER_PROBS` | `=0` disabilita; on | `sqrt(softplus(logit))` del router vettorizzato in un solo dispatch invece di due passate complete sulla riga di probabilità (256 valori Flash o 384 Pro). `=0` ripristina il percorso a due dispatch per i test di parità. |
| `DS4_FUSED_ROUTER_FINALIZE` | `=0` disabilita; on | Fonde la selezione top-6 del router e la normalizzazione bit-identica dei pesi selezionati, rimuovendo un dispatch per layer instradato (fino a 43 layer Flash o 61 Pro per token). |
| `DS4_FUSED_COMP_PROJ` | `=0` disabilita; on | Accoppia i matvec KV+gate del compressore in un solo dispatch sia per F16 sia per `DS4_COMP_Q8`, condividendo le letture delle attivazioni. Stesso ordine di riduzione per matrice; `0` ripristina i due dispatch. |
| `DS4_ASYNC_FFN` | `=0` disabilita; on | Commit asincrono della FFN instradata: l'attesa del route del layer successivo atterra su una FFN già in esecuzione (+10% misurato, identico al token). `=0` è il percorso di debug sincrono. |
| `DS4_ASYNC_ROUTE` | `=0` disabilita; on | Commit asincrono del route in decode: la FFN dell'esperto condiviso viene committata subito dietro il route, così la GPU concatena route→FFN senza pause di encode mentre la CPU attende la selezione degli esperti, e il join CPU sulla FFN condivisa prima dell'encode della FFN instradata viene saltato (coda in-order + hazard tracking — l'argomento di `DS4_ASYNC_FFN`). Identico al token; `DS4_PROFILE_ROUTE` forza il percorso sincrono. `=0` ripristina il route completamente sincrono per gli A/B. |
| `DS4_FAST_SAMPLER` | `=0` disabilita; on | Percorso rapido del sampler a vocabolario intero (`top_k<=0` con `min_p>0`, il default di server/subagent): raccoglie solo i candidati raggiungibili dalla camminata min-p invece di costruire e ordinare tutti i ~129k per token (ms di CPU risparmiati per token). Stessa selezione e stesso stream RNG; solo l'ordinamento dei pareggi esatti di logit (già non specificato nell'ordinamento completo) può differire. `=0` ripristina la storica costruzione completa. |
| `DS4_Q8_NSG` | `1...8`; `4` | Simdgroup per threadgroup per i matvec densi Q8_0. Valori diversi ripartizionano la riduzione K e possono cambiare gli ultimi bit in virgola mobile; spazzare 2/4/6/8 per macchina e validare l'output quando la parità esatta conta. |
| `DS4_MOE_NSG` | `1...8`; `4` | Simdgroup per threadgroup per i kernel id MoE/Q4 instradati. Partiziona le righe di output ed è regolato indipendentemente dallo split K del Q8 denso. |
| `DS4_DENSE_Q4_NSG` | `1...8`; eredita `DS4_MOE_NSG` | Simdgroup per threadgroup per le proiezioni Q4_K dense/raggruppate residenti. Partizionato per righe e bit-identico; separato dall'occupancy degli esperti instradati per la regolazione per GPU. |
| `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` | `64/128/256/512/1024/2048/4096`; `1024` | Numero di righe di KV compresso oltre il quale l'attention in decode passa dalla scansione densa al percorso sparso dell'indexer NSA (al di sotto, il setup del top-k costa più di quanto risparmi). Imposta anche il confine vivo per il caricamento one-shot dello scorer di `DS4_LAZY_IDX`. |
| `DS4_ADAPTIVE_SPLITK` | `=0` disabilita; on | Profondità split-K adattiva per la FlashAttention di decode: per `rows = righe grezze + righe compresse`, dispatcha esattamente `nwg = min(32, max(1, ceil(rows/32)))` workgroup invece di arrotondare a una potenza di due o usare sempre 32. Così 128 righe usano 4 workgroup e 129 ne usano 5, evitando il vecchio gradino 4→8. I contesti corti smettono di scrivere e leggere parziali vuoti; `nwg == 1` salta il dispatch di riduzione perché il kernel vec si auto-normalizza. Le profondità arbitrarie 1…32 sono supportate e preservano il calcolo previsto. `=0` ripristina il dispatch fisso a profondità 32 per gli A/B. |
| `DS4_VECTOR_COPY` | `=1` abilita; off | Copie contigue F32/F16 sperimentali impacchettate a quattro. Sono esatte bit a bit, comprese le code scalari e il trasporto dei bit F16, ma l'A/B a modello intero bilanciato per ordine su M1 Pro è risultato leggermente più lento in decode. La copia generica resta il default. |
| `DS4_FLASH_KV_STAGE` | `=1` abilita; off | Fonde la raccolta del raw-ring, lo staging F32→F16 della cache compressa e il padding parziale di K/V/mask in un solo dispatch. È risultato esatto bit a bit nei test GPU con e senza wrap e su 2,197,760 logits a modello intero. Su M1 Pro ha guadagnato circa il 2% in prefill ed è stato neutro in decode, quindi resta opt-in in attesa di misurazioni su altre GPU Apple e contesti vivi più lunghi. |
| `DS4_ROPE_PAIR` | `=1` abilita; off | Specializzazione RoPE in-place a sole coppie. Per il decode a un token ricostruisce anche le posizioni affini sulla GPU a meno che `DS4_ROPE_AFFINE=0`. I percorsi baseline, pair e affine sono risultati bit-identici nei casi normale, YaRN, inverso, decode e prefill; l'A/B end-to-end su M1 Pro non ha mostrato guadagni in decode. |
| `DS4_ROPE_AFFINE` | `=0` disabilita le posizioni affini; on quando pair è abilitato | Mantiene `DS4_ROPE_PAIR` attivo ma ripristina l'array di posizioni fornito dall'host. Non ha effetto mentre `DS4_ROPE_PAIR` è off. |

Esempio — il profilo misurato a bassa RAM nella demo (ciò che l'app applica
per default):

```sh
DS4_EXPERT_CACHE_SLOTS=22 DS4_RAW_RING=0 DS4_WILLNEED_EXPERTS=1 \
DS4_EXPERT_PREAD=1 DS4_DENSE_STREAM=1 DS4_MLOCK=1 \
DS4_DENSE_Q4=1 DS4_QKV_Q4=1 DS4_SHARED_Q4=1 \
DS4_PREFILL_UNION=256 DS4_PREFILL_CHUNK=512 DS4_PREFILL_ROUTE_BATCH=32 \
DS4_EXPERT_BUNDLE=1 DS4_MTLIO=1 DS4_MTLIO_MIN_GBS=4.0 \
DS4_POOL_INTERLEAVE=1 DS4_PREFILL_FFN_BATCH=1 DS4_GPU_INDEXER_TOPK=1 \
DS4_DENSE_Q4_KERNEL=1 DS4_FUSED_ROUTER_PROBS=1 \
DS4_FUSED_ROUTER_FINALIZE=1 DS4_FUSED_COMP_PROJ=1 \
DS4_EXPERT_LOOKAHEAD=0 DS4_DENSE_AHEAD=2 DS4_ASYNC_FFN=1 \
DS4_RESIDENT_DENSE=0 DS4_RESIDENT_COMP=1 DS4_LAZY_IDX=1 DS4_PROFILE_ROUTE=0 \
DS4_Q8_NSG=4 DS4_MOE_NSG=4 DS4_DENSE_Q4_NSG=4 \
  swift run DS4Demo /path/model.gguf 32 "Explain RoPE briefly."
```
### Server HTTP (scheda Server)

La configurazione si imposta nella scheda Server prima di premere Start (in
memoria, non persistita). Il server espone il singolo motore condiviso
caricato in Settings.

| Impostazione | Default | Cos'è / cosa fa |
|---|---|---|
| Host | `127.0.0.1` | Indirizzo di bind. Loopback per default; mettere TLS davanti prima di fare bind oltre il loopback — il listener stesso è HTTP in chiaro. |
| Porta | `8000` | Porta TCP. L'URL base diventa `http://<host>:<port>/v1`. |
| Max token per risposta | `1024` (64–8192) | Tetto usato quando il corpo di una richiesta non specifica `max_tokens`. |
| CORS | off | Aggiunge `Access-Control-Allow-Origin: *` per i client browser. |
| Chiave API | vuota (nessuna autenticazione) | Quando impostata, ogni richiesta `/v1` deve inviare `Authorization: Bearer <key>` (stile OpenAI) o `x-api-key: <key>` (stile Anthropic); altrimenti 401. |

L'id del modello è il basename del GGUF caricato (non c'è cambio di modello
per richiesta). I corpi delle richieste sono limitati a 32 MB; le richieste
vanno in timeout dopo 60 s.

Parametri per richiesta (mappati sul sampler del motore):

| Campo del corpo | Endpoint | Effetto |
|---|---|---|
| `temperature`, `top_p`, `top_k` | tutti | Controlli di campionamento standard. |
| `min_p`, `seed` | `chat/completions` | Soglia min-p; seed deterministico. |
| `max_tokens` / `max_completion_tokens` / `max_output_tokens` | tutti / chat / Responses | Tetto di generazione (ripiega sul Max token del server). |
| `stream` | tutti | Streaming SSE (default `false`). |
| `reasoning_effort` (chat), `reasoning.effort` (Responses), `thinking.type:"enabled"` (Anthropic) | come elencato | `high`/`xhigh`/`medium` abilitano il thinking; `low`/`minimal`/assente lo disabilitano. |
| `tools`, `messages`/`input`/`prompt` | secondo l'API | Specifiche degli strumenti e contenuto della conversazione. |

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-key" \
  -d '{"model":"deepseek-v4-flash","stream":true,"temperature":0.3,
       "max_tokens":256,
       "messages":[{"role":"user","content":"Hello"}]}'
```

### Distribuito (Settings + scheda Worker)

Il lato **worker** ha una sola impostazione: la **porta** di ascolto (default
`9100`, non persistita). I worker partono inattivi; tutto il resto viene
assegnato dal coordinatore al momento della connessione.

Il lato **coordinatore** (Settings, in memoria):

| Impostazione | Default | Cos'è / cosa fa |
|---|---|---|
| Workers (un `host:port` per riga) | `127.0.0.1:9100` | La lista dei peer, **in ordine di layer**: i layer vengono partizionati in modo contiguo su di essa e l'ultima fetta esegue anche la testa di output. |
| Bit di attivazione | `32` (32/16/8) | Larghezza sul filo dello stato nascosto HC (`nHC × nEmbd` float) che attraversa i worker a ogni token: float32, float16 o int8 scalato per tagliare la banda LAN. |
| Chunk prefill | `32` (1–256) | Token per frame di prefill distribuito. |
| Max token per risposta | `512` (16–4096) | Tetto di generazione per turno distribuito. |
| Inoltro worker-verso-worker | off | I worker inoltrano le attivazioni direttamente a valle invece di passare per il coordinatore; il worker terminale risponde all'indirizzo di ritorno. |
| Host / porta di ritorno | vuoto / `9099` | L'IP LAN del coordinatore e la porta del listener per le risposte inoltrate (richiesti quando l'inoltro è attivo). |
| Split verticale | off | Usa il parallelismo degli esperti del protocollo v11: questo Mac esegue la dorsale densa e i worker possiedono maschere di esperti dimensionate sulla geometria su tutti i layer. Richiede un RTT cablato sotto circa 1 ms. |

L'ASSIGN inviato a ogni worker trasporta il GGUF (percorso, nome e —
protocollo v5 — il file stesso più l'expert-bundle e i sidecar `.q4dense`,
trasmessi in chunk da 4 MB con SHA-256 inline e conservati sotto
`Application Support/DwarfStar/dist-models`), la dimensione del contesto, il
budget della cache esperti, il budget disk-KV, la fetta di layer, la imatrix
d'uso per il pre-riscaldamento e le regolazioni di prestazioni del
coordinatore. La whitelist delle regolazioni inoltrate è un **confine di
sicurezza su quali nomi di ambiente possono essere impostati**, non una
promessa che ogni configurazione consentita sia bit-identica:
`DS4_DENSE_STREAM`, `DS4_DENSE_AHEAD`, `DS4_MLOCK`, `DS4_EXPERT_PREAD`,
`DS4_PREAD_SPLIT`, `DS4_WILLNEED_EXPERTS`, `DS4_ASYNC_FFN`,
`DS4_EXPERT_LOOKAHEAD`, `DS4_Q8_NSG`, `DS4_LAZY_IDX`, `DS4_RESIDENT_COMP`,
`DS4_FUSED_HC`, `DS4_FUSED_MOE`, `DS4_RAW_RING`, `DS4_EXPERT_CACHE_UNIFORM`,
`DS4_POOL_INTERLEAVE` e le cinque regolazioni `DS4_PREFILL_*`. Il lossy
`DS4_DENSE_Q4` viaggia invece come campo tipizzato dell'ASSIGN, insieme al suo
file di cache. Le regolazioni di I/O e scheduling preservano la matematica
prevista, ma i kernel fusi possono cambiare l'ordine di riduzione in virgola
mobile e `DS4_PREFILL_MM` cambia esplicitamente l'ordine di accumulazione;
perciò impostazioni miste non sono sempre identiche bit per bit anche quando
la qualità è invariata. Il coordinatore legge anche `DS4ExpertCacheSlots`,
`DS4DiskKV`, `DS4DiskKVBudgetKTok`, `DS4ExpertBundle` e `DS4DenseQ4` dalle
impostazioni locali per costruire l'ASSIGN, e persiste l'agente della chat
distribuita come `DS4SelectedAgentDist`.

I frame sono TCP in chiaro senza autenticazione (protocollo v11, magic
`DS4D`): usare la modalità distribuita solo su reti fidate.

Setup minimo a due Mac: sul Mac worker aprire **Worker**, mantenere la porta
`9100`, premere Start; sul Mac coordinatore impostare Mode = Distributed,
elencare `<worker-ip>:9100` in Workers, premere Connect e chattare.

### Server MCP (scheda MCP)

Server [Model Context Protocol](https://modelcontextprotocol.io) esterni.
La lista è persistita in `UserDefaults` sotto `DS4MCPServers`; import/export
usa il JSON standard `mcpServers` (compatibile Claude Desktop / Cursor /
VS Code — si importa anche un oggetto puro `{"<name>": {…}}`):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": { "MY_VAR": "value" }
    },
    "remote-api": {
      "url": "https://example.com/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

| Campo | Trasporto | Cos'è / cosa fa |
|---|---|---|
| `command` | stdio | Eseguibile lanciato come processo figlio (risolto via PATH, inclusi `/usr/local/bin` e `/opt/homebrew/bin`). La sua presenza seleziona il trasporto stdio. |
| `args` | stdio | Argomenti da riga di comando (default `[]`). |
| `env` | stdio | Variabili d'ambiente extra fuse sopra l'ambiente ereditato. |
| `url` | HTTP | Endpoint Streamable-HTTP (`http://`/`https://`); la sua presenza seleziona il trasporto HTTP. |
| `headers` | HTTP | Header di richiesta extra inviati verbatim (es. `Authorization`). |
| `enabled` | entrambi | Interruttore on/off persistito per server (default `true`). |

Gli strumenti dei server connessi compaiono come `mcp_<server>_<tool>`
accanto agli integrati, nel foglio Tool della chat e nelle liste di strumenti
degli agenti.

### Download dei modelli

Il foglio **Scarica…** in Settings (e nella vista pre-caricamento) legge
l'unico catalogo tipizzato posseduto da `DS4Engine`; nomi di file, confini dei
pacchetti, digest SHA-256 e disponibilità a runtime non sono duplicati in
SwiftUI. Scarica nativamente dai repository Hugging Face per target
`antirez/deepseek-v4-gguf` e `antirez/glm-5.2-gguf` in
`~/Library/Application Support/DwarfStar/models/`, senza `curl`, `hf` né un
processo helper.

Il foglio riporta **Non scaricato**, **Parziale** o **Installato**, lo spazio
libero, il progresso per file/pacchetto e se una voce è eseguibile. Un nome di
file di catalogo esatto già trovato come file regolare leggibile e non vuoto
nella directory gestita, nelle directory di modelli di sviluppo configurate o
nella directory del modello corrente viene riusato sul posto; quando il
catalogo fissa una dimensione esatta, anche quel conteggio di byte deve
corrispondere. Un trasferimento annullato o interrotto conserva
`<filename>.part` e lo riprende con HTTP Range; un server che ignora Range
causa un riavvio sicuro invece dell'accodamento di byte incompatibili. I nuovi
trasferimenti vengono accettati solo dopo la verifica del conteggio finale di
byte e dello SHA-256 fissato dal catalogo, poi il `.part` viene rinominato
atomicamente.

La precedenza di autenticazione è: token salvato in **Settings → Hugging
Face** (Keychain di macOS, mostrato solo in forma oscurata) > `HF_TOKEN` >
`~/.cache/huggingface/token`. I file pubblici possono essere scaricati senza
token.

| Id catalogo | Artefatto | Runtime dopo il download | ~Dimensione |
|---|---|---|---:|
| `q2-imatrix` | DeepSeek V4 Flash, imatrix IQ2XXS/Q2_K | selezionabile ed eseguibile | 87 GB |
| `q2-q4-imatrix` | DeepSeek V4 Flash, esperti misti IQ2XXS/Q4_K | selezionabile ed eseguibile | 98 GB |
| `q4-imatrix` | DeepSeek V4 Flash, esperti Q4_K | selezionabile ed eseguibile | 165 GB |
| `pro-q2-imatrix` | DeepSeek V4 Pro, GGUF Q2 singolo | selezionabile ed eseguibile localmente | 465 GB |
| `pro-q4-split` | DeepSeek V4 Pro, due shard Q4 (`00…30` e `31…output`) | solo download; nessuno dei due shard è un modello locale selezionabile | 900 GB |
| `glm-5.2-iq2-xxs` | GLM 5.2, esperti instradati IQ2_XXS | solo download; runtime GLM non ancora implementato | 211 GB |
| `glm-5.2-q2-k` | GLM 5.2, esperti instradati Q2_K | solo download; runtime GLM non ancora implementato | 262 GB |
| `glm-5.2-q4-k` | GLM 5.2, esperti instradati Q4_K | solo download; runtime GLM non ancora implementato | 434 GB |

Le tre voci Flash e la voce Pro Q2 a file singolo compaiono nella selezione
automatica dei modelli locali. Pro Q4 resta visibile nel foglio di download
con la sua esplicita ragione di indisponibilità; scaricare quel pacchetto
diviso non cambia mai il modello attivo. Le tre varianti GLM 5.2 seguono la
stessa regola solo-download: sono mostrate nel foglio ma escluse dalla
scoperta automatica e dalla selezione dei modelli finché il runtime `glm-dsa`
non sarà implementato. Il supporto di detector, tokenizer e schema non
scavalca questo cancello. Il compagno MTP è deliberatamente fuori dal catalogo
GUI dei modelli principali e non è mostrato come riga di modello
selezionabile/scaricabile.

## Packaging di una `.app`

```sh
make app          # -> build/DwarfStar.app, release, firma ad-hoc
open build/DwarfStar.app
```

`make app` usa [`packaging/make_app.sh`](packaging/make_app.sh) e firma il
bundle **senza gli entitlement di App Sandbox**. È intenzionale per il
pacchetto locale ad-hoc: ha accesso normale al filesystem e `NSOpenPanel` può
selezionare un GGUF senza restrizioni Powerbox/bookmark. Impostare
`DS4_SIGN_IDENTITY` cambia l'identità di firma ma non aggiunge gli
entitlement sandbox.

Per una build di distribuzione sandboxed, generare/compilare invece il
progetto Xcode; `project.yml` applica `packaging/DwarfStar.entitlements` con
le capacità di client/server di rete e di file/bookmark selezionati
dall'utente. Firmare quella build con Developer ID e notarizzarla.

## Stato

Funzionante e verificato su un MacBook Pro M1 Pro 16 GB; il preset di
prestazioni è stato benchmarkato l'ultima volta il 2026-07-13 con il modello
DeepSeek-V4 Flash 2-bit imatrix:

- caricamento del modello e chat in streaming sul GGUF a 2 bit;
- gestione di thinking/ragionamento;
- riuso del KV multi-turno;
- allegati di file di testo;
- chiamate di strumenti DSML con gli integrati;
- agenti e profili di esperti per agente;
- libreria dei progetti;
- pannello di tuning e controlli della cache esperti;
- cache disk-KV, attiva per default;
- profilo veloce a bassa RAM: pread diretto degli esperti, streaming denso,
  `mlock`, set completo di proiezioni Q4, expert bundle + MetalIO e cache
  esperti a 22 slot;
- server HTTP nativo, verificato per lo streaming OpenAI `chat/completions` e
  Anthropic `messages`.

Implementato, ma che richiede ancora una validazione più ampia sul
dispositivo:

- `/v1/responses`;
- sub-agenti a contesto isolato con snapshot/ripristino del KV principale e
  cache KV a chiave di contenuto per file/progetto;
- gli agenti predefiniti più recenti: Orchestrator, LaTeX, Documentation;
- protocollo di inferenza distribuita, worker/coordinatore, UI e benchmark;
- chat/benchmark con parallelismo verticale degli esperti;
- parità numerica più ampia ed esecuzioni distribuite multi-Mac.

## Limiti noti

Sui sistemi da 16 GB il decode è limitato dall'I/O. È la fisica dello
streaming da SSD di un modello MoE da 284B; l'inferenza distribuita è la
mitigazione prevista. La memoria KV scala comunque con il contesto, quindi
mantenere il contesto modesto sui sistemi a bassa RAM. Il ring raw-KV riduce
solo la cache grezza, non ogni riga di KV compresso. Server, inferenza
distribuita, diagnostica e benchmark sono pannelli nativi in-process; non
restano pannelli UI guidati da sottoprocessi.
