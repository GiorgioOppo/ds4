# Documentazione DwarfStar (DS4-gui)

Front-end nativo **Swift / SwiftUI** per il motore di inferenza **DeepSeek V4
(ds4 / DwarfStar)** su macOS Apple Silicon.

Questo documento descrive **come è fatto** il progetto e **come si usa**, sia
dal lato sviluppatore (la demo CLI `DS4Demo`) sia dal lato utente finale
(l'app SwiftUI `DwarfStar`). Per i dettagli di build e packaging fai
riferimento anche al [`README.md`](../README.md) nella radice del progetto.

> 🛠️ Per il funzionamento interno del motore (encoder/tokenizer, decoder,
> attenzione MLA, MoE, compressore NSA, streaming dei pesi, sampler,
> quantizzazione) vedi [`ARCHITETTURA-MOTORE.md`](ARCHITETTURA-MOTORE.md).

---

## Indice

1. [Panoramica](#1-panoramica)
2. [Architettura a livelli](#2-architettura-a-livelli)
3. [Concetti chiave del motore](#3-concetti-chiave-del-motore)
4. [La demo CLI: `DS4Demo`](#4-la-demo-cli-ds4demo)
5. [L'app SwiftUI: `DwarfStar`](#5-lapp-swiftui-dwarfstar)
6. [Flusso utente passo-passo](#6-flusso-utente-passo-passo)
7. [Configurazione automatica e preset hardware](#7-configurazione-automatica-e-preset-hardware)
8. [Modalità di streaming e memoria](#8-modalità-di-streaming-e-memoria)
9. [Pannelli avanzati](#9-pannelli-avanzati)
10. [Build, esecuzione e packaging](#10-build-esecuzione-e-packaging)
11. [Risoluzione dei problemi](#11-risoluzione-dei-problemi)
12. [Glossario](#12-glossario)

---

## 1. Panoramica

DwarfStar permette di eseguire **localmente** il modello DeepSeek V4 (un
Mixture-of-Experts da centinaia di miliardi di parametri) su un Mac Apple
Silicon, anche con RAM molto inferiore alla dimensione del modello, grazie a
una pipeline Metal completamente reimplementata in **Swift puro** e a tecniche
di *streaming* dei pesi per-layer ed expert-cache.

Il progetto offre due punti d'ingresso:

| Punto d'ingresso | Tipo | A chi serve | Cosa fa |
|---|---|---|---|
| **`DS4Demo`** | CLI | sviluppatori / verifica | Avvia il runtime Metal, esegue un self-test GPU e — dato un GGUF — genera token in streaming dal modello reale. |
| **`DwarfStar`** | App SwiftUI | utente finale | Chat grafica con tool/agenti/progetti, ragionamento collassabile, tuning della cache esperti, server HTTP nativo (OpenAI+Anthropic), inferenza distribuita su più Mac, diagnostica. |

Entrambi condividono lo stesso motore Swift (`DS4Core` + `DS4Metal`), quindi
ciò che la demo dimostra a riga di comando è esattamente ciò che l'app esegue
sotto il cofano.

---

## 2. Architettura a livelli

```
┌───────────────────────────────────────────────────────────────┐
│  DwarfStarApp (SwiftUI)   ← GUI: chat, agenti, progetti,       │
│        │                    tuning, server, distribuito, diag  │
│  DS4Engine (Swift)        ← InferenceService (actor):          │
│        │                    prompt → stream di eventi          │
│        │                    (reasoning/text/toolStream/toolCall)│
│        │                    + ToolRegistry, agenti, progetti,  │
│        │                    LocalServer (in DwarfStar), Dist*  │
│  ┌─────┴───────────────────────────────────────────────┐      │
│  │ DS4Core (Swift puro)   DS4Metal (Swift puro)         │      │
│  │  • GGUF parser          • MetalRuntime (kernel)      │      │
│  │  • Tokenizer            • GPUTensor / GraphContext   │      │
│  │  • Sampler              • DSV4Decoder                │      │
│  │  • ModelShape           • StreamingDecoder           │      │
│  │  • KV cache / SSD plan  • GGUFWeights loader         │      │
│  └─────────────────────────────────────────────────────┘      │
│                          │                                     │
│        metal/*.metal  (kernel sorgente, incorporati nel binario)│
└───────────────────────────────────────────────────────────────┘
```

Punti importanti:

- **Nessun link esterno.** `DS4Core` e `DS4Metal` sono Swift puro: niente
  motore C, niente libreria statica precompilata. Questo rende il progetto
  apribile e compilabile in un progetto Xcode pulito (`DwarfStar.xcodeproj`).
- **I kernel Metal sono incorporati nel binario** tramite
  `Sources/DS4Metal/Runtime/KernelSources.swift`, generato da `metal/*.metal`
  con `make embed-kernels`. A runtime **non serve** la cartella `metal/`.
- L'intero stack è **puro Swift**: la GUI usa il motore via `DS4Engine`. Non c'è
  più alcun bridge verso il motore C originale (rimosso dal progetto).

### Mappa dei sorgenti

| Cartella | Contenuto |
|---|---|
| `Sources/DS4Core/` | Parser GGUF, tokenizer, sampler, forma del modello, piani KV/SSD. |
| `Sources/DS4Metal/` | Runtime Metal, tensori GPU, kernel, grafo di calcolo, decoder, loader pesi. |
| `Sources/DS4Engine/` | `InferenceService` (la facciata async usata dalla GUI), downloader, diagnostica. |
| `Sources/DS4Demo/` | La demo CLI (`main.swift`). |
| `Sources/DwarfStar/` | L'app SwiftUI (App, Chat, Server, Bench, Diagnostics, Models, Support). |
| `metal/` | Sorgenti dei kernel Metal (fonte di verità, poi incorporati). |
| `Tests/DS4CoreTests/` | Suite di test sui kernel, sul grafo, tokenizer, GGUF, sampler, ecc. |

---

## 3. Concetti chiave del motore

- **Compilazione kernel a runtime.** I kernel Metal vengono compilati
  all'avvio del `MetalRuntime`. La demo li conta e lo segnala
  (`… N kernels compiled`).
- **Un solo motore per processo.** Il backend Metal mantiene stato globale
  (device, queue, pipeline, cache degli expert): il modello si carica una
  volta sola. Per questo **l'app non esegue contemporaneamente** la chat
  in-process e il subprocess `ds4-server` (caricherebbe i pesi due volte →
  rischio OOM sui quant grandi).
- **StreamingDecoder (load/compute/evict per-layer).** Permette al modello da
  ~164 GB di girare in 16 GB di RAM: i pesi non-routed sono viste `mmap`
  no-copy (residenti via page cache, evictabili), e per ogni token si
  "raccolgono" soltanto i **6 expert selezionati su 256** per layer (≈ 6/256
  dell'I/O degli expert). È il modello `--ssd-streaming` del motore C.
- **Rilevamento quant.** `GGUFWeights.detectMoEQuant` legge dal GGUF lo schema
  di quantizzazione (es. `Q4_K`+`Q8` oppure `IQ2_XXS`/`Q2_K`+`F16`) e configura
  il decoder di conseguenza: un disallineamento produrrebbe output spazzatura.
- **Template di chat e specials.** Il `Tokenizer` conosce i token speciali del
  protocollo DS4 (`bos`, `eos`, `user`, `assistant`, fine-`<think>`) e produce
  il prompt di chat con `encodeChatPrompt(system:prompt:think:)`.

---

## 4. La demo CLI: `DS4Demo`

`DS4Demo` (`Sources/DS4Demo/main.swift`) è una CLI minimale che pilota il
motore Swift puro **senza link esterni**. Serve a verificare la pipeline e a
dimostrare la generazione reale.

### Utilizzo

```sh
cd DS4-gui

# 1) Solo bring-up: porta su il runtime Metal + self-test GPU
swift run DS4Demo

# 2) Genera token da un modello reale (I/O pesante)
swift run DS4Demo <percorso.gguf> [maxNew] ["prompt"]
```

Argomenti:

| Posizione | Significato | Default |
|---|---|---|
| `argv[1]` | percorso al file `.gguf` | — (se assente, solo bring-up) |
| `argv[2]` (`maxNew`) | numero di token da generare | `4` |
| `argv[3]` | prompt utente | `"ciao come stai? rispondi in 1 parola"` |

### Cosa fa, passo per passo

1. **Bring-up Metal** — crea `MetalRuntime()` (kernel già incorporati nel
   binario) e stampa device e numero di kernel compilati:
   ```
   DS4Demo: Metal runtime up on Apple M…, N kernels compiled
   ```
2. **Self-test GPU** — `runTouchSelfTest()` verifica che la GPU calcoli
   correttamente:
   ```
   DS4Demo: GPU self-test PASSED
   ```
   Se non viene passato alcun GGUF, la demo termina qui (utile come smoke test
   del solo runtime).
3. **Apertura del GGUF** — `GGUFModel(path:…)` con mapping Metal, no prefetch.
4. **Audit dei tipi (opzionale)** — se la variabile d'ambiente
   `DS4_TYPES_ONLY=1` è impostata, la demo stampa i dtype dei tensori per-layer
   (expert, router, attention, hyper-connection, ecc.), i token speciali del
   tokenizer e la tokenizzazione di un prompt di esempio, poi esce. Serve a
   diagnosticare disallineamenti di quantizzazione.
5. **Configurazione quant + RoPE** — rileva lo schema MoE e imposta i parametri
   RoPE, poi costruisce lo `StreamingDecoder` expert-cached/mapped.
6. **Forward singolo di prova** — esegue un `forward(token:0,…)` e riporta tempo,
   finitezza dei logit e argmax:
   ```
   DS4Demo: 1 forward in 12.3s — logits[…] finite=YES argmax=… (logit …)
   ```
7. **Generazione di chat reale** (se `maxNew > 0`):
   - **Prefill**: tokenizza il prompt con il template di chat e processa ogni
     token, con progressione per-token (`prefill 3/11 …s`).
   - **Decode**: campiona greedy (`temperature 0`), si ferma su `eos`, e
     **stream** dei byte di ogni token su `stdout` man mano che vengono
     prodotti (come `./ds4`). Alla fine stampa il throughput:
     ```
     Risposta: …
     DS4Demo: N tokens in Ts (X tok/s)
     ```

> Nota: i messaggi di stato (`log(...)`) vanno su **stderr**, mentre la
> risposta del modello va su **stdout**: puoi reindirizzarli separatamente.

---

## 5. L'app SwiftUI: `DwarfStar`

L'app (`Sources/DwarfStar/`) è una finestra macOS con una **sidebar**
(`RootView` + `AppSection`):

| Sezione | Icona | Scopo |
|---|---|---|
| **Chat** | fumetti | Conversazione in streaming (tool, agenti, progetto, riuso KV). Picker **Locale** (motore in-process) / **Distribuito** (coordina il cluster di worker). |
| **Agenti** | persone | Ruoli: visualizza/modifica i prompt di definizione e i tool di ogni agente, creane di nuovi, scegli l'attivo, export/import JSON. |
| **Progetto** | cartella | Libreria di progetti salvati (memoria separata dalla chat): importa più cartelle, attivane una; l'agente la esplora con i tool `project_*` (solo le parti lette entrano in conversazione). |
| **Tuning** | slider | Cache esperti (persistenti+dinamici) e profilo d'uso ("imatrix d'uso") per-agente. Il fine-tuning dei pesi non è possibile on-device. |
| **Server** | rack | Server HTTP **nativo in-process**, compatibile OpenAI + Anthropic (§9). |
| **Worker** | cpu | Questo Mac come worker del cluster distribuito (slice di layer); il coordinatore è in Chat → Distribuito (§9). |
| **Benchmark** | tachimetro | Nativo: prefill + generazione (token/s) del motore in-process a contesti crescenti, con grafico. |
| **Diagnostica** | stetoscopio | Tokenizzazione di un testo, chat template + token speciali (tipo/atomicità), console del motore. |

Il punto d'ingresso è `DwarfStarApp` (`@main`): crea lo `ChatStore`
condiviso, installa la cattura dello stderr del motore (`EngineLog`) e apre la
finestra (min 860×600).

### Il modello di stato della chat: `ChatStore`

`ChatStore` è il view-model `@MainActor @Observable` che possiede il
`InferenceService` (un *actor*) e rispecchia il suo output nello stato
osservabile della UI. Macchina a stati (`Phase`):

```
needsModel ──load()──▶ loading ──ok──▶ ready
     ▲                    │
     └────────────────────┴──errore──▶ failed(messaggio)
```

- `needsModel` / `failed` → mostra la schermata di caricamento (`ModelLoadView`).
- `loading` → mostra spinner "Caricamento del modello…".
- `ready` → mostra la chat (`ChatView`).

Funzioni principali: `scanModels()` (scansione GGUF), `applyRecommendedPreset()`
(configura per la RAM), `load()` (apre il modello fuori dal main thread),
`send()` (invia il messaggio e fa lo streaming della risposta), `stop()`
(annulla la generazione), `newChat()` (azzera la conversazione).

### Lo streaming degli eventi

`InferenceService.send(...)` restituisce un `AsyncThrowingStream<GenEvent>`. Lo
`ChatStore` consuma tre tipi di evento e li smista nell'ultimo messaggio:

| Evento | Destinazione UI |
|---|---|
| `.reasoning(String)` | blocco "Ragionamento" collassabile (chain-of-thought). |
| `.text(String)` | bolla di risposta visibile. |
| `.progress(String)` | barra di stato (es. `prefill 3/11`, `12 token · 1.4 tok/s`). |

---

## 6. Flusso utente passo-passo

Questo è il percorso tipico di un utente nell'app `DwarfStar`.

### Passo 1 — Selezione e configurazione del modello

All'avvio (fase `needsModel`) appare la **schermata di caricamento**
(`ModelLoadView`), un `Form` con le seguenti sezioni:

1. **Modelli disponibili** — elenco dei GGUF trovati in `scriptDir` e
   `scriptDir/gguf`, ciascuno con nome e dimensione; si seleziona con un tap
   (radio button). I pulsanti in testata permettono di **ricaricare** la
   scansione (↻) e aprire la sheet di **download** (`Scarica…`).
2. **Configurazione automatica** — un pulsante *"Configura per la tua RAM (N GB)"*
   applica il preset consigliato (vedi §7) e mostra una nota esplicativa.
3. **Percorsi** — campo di testo per il percorso GGUF (modificabile a mano) e
   pulsante **Sfoglia…**. Con l'**App Sandbox** attiva (build firmata/da Xcode) il
   modello va selezionato dal pannello *Sfoglia…*: l'app ottiene l'accesso
   *security-scoped* al file e ne salva un **bookmark**, così riapre lo stesso
   GGUF ai lanci successivi senza richiederlo. Un percorso digitato a mano non
   è leggibile sotto sandbox (la scansione automatica delle cartelle è anch'essa
   limitata). Senza sandbox (`swift run DwarfStar`) funzionano anche i percorsi
   digitati e la scansione.
4. **Memoria** — nota informativa: lo streaming da SSD è sempre attivo, senza
   opzioni da configurare (vedi §8).
5. **Contesto e system prompt** — stepper per la dimensione del contesto
   (1024…1.000.000 token) e campo per un system prompt opzionale.
6. **Avvisi di memoria** — se il modello rischia di non entrare in RAM, appare
   un warning arancione (`MemoryInfo.loadWarning`).
7. **Carica modello** — pulsante (scorciatoia ⌘↩) che avvia `load()`; sotto è
   mostrata la RAM di sistema rilevata.

### Passo 2 — Caricamento

Premendo **Carica modello** la fase passa a `loading`: appare uno spinner con
*"Mappa il GGUF e compila i kernel Metal. Può richiedere alcuni secondi."* Il
caricamento avviene in un `Task.detached`, quindi la UI resta reattiva. Al
termine la fase diventa `ready` (oppure `failed` con il messaggio d'errore
mostrato nella schermata di caricamento).

### Passo 3 — Conversazione

In fase `ready` appare la **`ChatView`**, divisa in tre parti:

- **Header** — nome del modello e metadati: *N layer · k-bit · ctx … · KV ~…*.
  A destra il toggle **Thinking** e il pulsante **Nuova chat**.
- **Transcript** — l'elenco scorrevole dei messaggi (`MessageRow`):
  - messaggi utente allineati a destra, risposte assistant a sinistra;
  - se presente, il **Ragionamento** è un blocco collassabile (`DisclosureGroup`
    con icona cervello) sopra la risposta;
  - mentre la risposta è vuota, appare un piccolo spinner;
  - lo scroll segue automaticamente l'ultimo token prodotto.
- **Composer** — sopra, la barra di **stato** live (spinner + testo
  `prefill …` / `… tok/s`) durante la generazione; sotto, il campo di testo
  (invio con ↩) e il pulsante **invia** (▲) che diventa **stop** (■ rosso)
  durante la generazione.

#### Cosa succede premendo invio (`send()`)

1. Il testo viene aggiunto come messaggio utente; si crea un messaggio
   assistant vuoto.
2. `isGenerating = true`; il pulsante diventa **Stop**.
3. Si apre lo stream con `thinkMode` (`.high` se Thinking è attivo, altrimenti
   `.none`), `SamplingParams()` di default e `maxTokens: 4096`.
4. Gli eventi affluiscono: il ragionamento riempie il blocco collassabile, il
   testo riempie la bolla, un'eventuale tool-call scorre live come markup
   grezzo (`.toolStream`) finché non diventa card, il progress aggiorna la
   barra di stato.
5. In caso d'errore, viene appeso il messaggio d'errore e — se presente — la
   coda del log del motore (`EngineLog.shared.tail()`).
6. Alla fine `isGenerating = false` e la barra di stato si svuota.

#### Thinking, Stop, Nuova chat

- **Thinking** — attiva la catena di pensiero: il prompt termina con `<think>`,
  il decoder emette `.reasoning` finché non incontra il token di fine-think,
  poi passa a `.text`.
- **Stop** — `stop()` annulla il `Task` di generazione (cancellazione
  cooperativa: il loop di decode controlla `Task.checkCancellation()`).
- **Nuova chat** — `newChat()` svuota i messaggi e resetta la conversazione nel
  servizio (mantenendo il system prompt configurato).

#### Tool (function calling)

Il pulsante **Tool** nell'header apre un foglio dove abiliti il function calling
e scegli quali **tool integrati** esporre al modello. I built-in di demo sono:

- **`now`** — data/ora corrente (ISO-8601), auto-eseguibile;
- **`calculator`** — valuta un'espressione aritmetica (`+ - * / ( )`),
  auto-eseguibile in modo sicuro (input ristretto a cifre/operatori);
- **`add` / `subtract` / `multiply`** — operano su due numeri `a` e `b`
  (somma, sottrazione, moltiplicazione); accettano numeri JSON o stringhe
  numeriche;
- **`project_list` / `project_read` / `project_search`** — esplorano il
  **progetto attivo** (tab Progetto): elenco file, lettura paginata con numeri
  di riga, ricerca testuale. Sono i tool dell'agente *coding*.

Nel foglio **Tool** c'è anche il toggle **"Dichiarazione compatta"** (attivo di
default): invece dello schema completo dei tool manda solo `nome(parametri)` + una
riga di formato → **meno token di prefill** (utile sull'inferenza locale, dove ogni
token pesa), al piccolo costo di discostarsi dal testo di addestramento. Disattivalo
per il formato pieno se la qualità delle chiamate ne risente (il toggle è l'unica
fonte di verità: non esiste più una variabile d'ambiente).

Tutti i built-in sono abilitati di default e vengono eseguiti automaticamente.

Flusso di una chiamata a tool:

1. Con i tool abilitati, l'`InferenceService` **dichiara** i tool nel prompt
   (blocco "## Tools") e il modello, invece di rispondere, può emettere una
   **tool-call** nel formato **DSML/XML** del paper (`<｜DSML｜tool_calls>…`). Il
   decoder riconosce il token `｜DSML｜` e bufferizza la chiamata invece di
   mostrarla come testo.
2. Mentre la chiamata viene generata, la GUI mostra il **markup grezzo in
   streaming** in un riquadro tratteggiato ("Generazione chiamata tool…"); alla
   chiusura del blocco il riquadro è sostituito dalla card arancione
   **"Chiamata tool"** con nome e argomenti JSON.
3. **Esecuzione**: i tool integrati vengono eseguiti **automaticamente**
   (`ToolRegistry.execute`) e il risultato compare come bolla verde; per tool
   non integrati si apre il foglio **"Risultati dei tool"** dove inserisci il
   risultato a mano.
4. Il risultato viene reimmesso nella conversazione (`provideToolResults`) e il
   modello produce la risposta finale — o **altre** tool-call: il loop si ripete
   finché non resta che testo.

Il formato è quello **autorevole del paper DeepSeek-V4** (Table 4): XML basato sul
token `｜DSML｜`. Il `tokenizer.chat_template` grezzo del GGUF è ispezionabile dal
pannello **Diagnostica** ("Mostra chat template + formato tool") o via
`InferenceService.chatTemplate()`. Vedi
[`ARCHITETTURA-MOTORE.md`](ARCHITETTURA-MOTORE.md) §14 per i dettagli.

> Nota: il multi-turno **riusa la KV cache** (design append-only). Il servizio
> tiene gli id esatti già nella KV (`committedIds`) e a ogni turno fa il prefill
> **solo del suffisso nuovo** (chiusura `<eos>` del turno precedente + nuovo
> turno utente/tool-result + apertura assistant). Reasoning e tool-call restano
> verbatim nella KV. Se una generazione viene interrotta (Stop/errore), la KV
> viene marcata *dirty* e il turno successivo la **ricostruisce** dagli id
> committati (il compressore NSA è ricorrente e non può tornare indietro).

---

## 7. Configurazione automatica e preset hardware

Il pulsante *"Configura per la tua RAM"* invoca `applyRecommendedPreset()`, che
usa `HardwarePresets.forRAM(...)` per scegliere un preset conservativo in base
alla RAM fisica rilevata. Quando il preset preferisce una quant a 2 bit,
l'app seleziona automaticamente un modello 2-bit già su disco, oppure suggerisce
come scaricarlo.

| RAM rilevata | Contesto | Quant consigliata | Note |
|---|---|---|---|
| **< 24 GB** | 4096 | 2-bit | Sotto il minimo del progetto (64 GB): pesi streamati da SSD (lento ma funziona). |
| **24–80 GB** | 8192 | 2-bit | Gran parte dei pesi resta in page cache. |
| **80–200 GB** | 32768 | 2-bit | Modello interamente in RAM. |
| **> 200 GB** | 32768 | Q4 | Anche la quant Q4 entra in RAM. |

`HardwarePresets.isTwoBit(...)` riconosce i nomi 2-bit (`iq2`, `q2k`, `-q2`,
`q2-`). Il preset compila i campi della schermata di caricamento; l'utente può
sempre modificarli manualmente prima di premere **Carica modello**.

---

## 8. Memoria e streaming

Nel motore Swift **lo streaming da SSD è sempre attivo** e non richiede
configurazione: i pesi non-routed sono viste `mmap` no-copy (residenti via page
cache quando la RAM basta, evictabili quando non basta) e per ogni token vengono
letti **solo i 6 expert selezionati**. Se il modello entra in RAM, la page cache
lo tiene residente automaticamente — non esistono più i toggle del motore C
(streaming on/off, RAM minima, per-layer eviction), che non avevano effetto.

`MemoryInfo.loadWarning(...)` calcola un avviso preventivo:

- modello **> RAM** → funziona, ma i pesi vengono riletti da SSD a ogni token
  (decine di secondi/token);
- modello **> 4× RAM** → alto rischio OOM, perché le parti non-routed e la KV
  cache devono comunque stare in RAM.

Dopo il caricamento, l'header della chat mostra una stima dell'impronta della
**KV cache** (`nLayer × contesto × headDim × F32`).

---

## 9. Pannelli avanzati

### Server (`ServerView` + `LocalServer`) — HTTP nativo in-process

Server HTTP **nativo** (Network.framework): espone il modello su un endpoint
compatibile **OpenAI e Anthropic**, senza alcun sottoprocesso. Carica un
**proprio** `InferenceService` in-process: i pesi GGUF sono viste mmap no-copy,
quindi la page cache del SO li **condivide** col motore della chat — niente
seconda copia dei pesi in RAM (solo KV cache e scratch sono separati).

- configurazione: GGUF (con **Sfoglia**, sandbox), host/porta, contesto,
  max token per risposta, toggle **CORS**;
- **Avvia / Ferma** con stato "In ascolto", esempio `curl` pronto, log live;
- richieste **serializzate** (modello singolo): una generazione alla volta
  (`RequestGate`); ogni richiesta è **stateless** (la lista messaggi completa
  viene ri-renderizzata, semantica OpenAI — `InferenceService.complete`).

Endpoint:

| Metodo | Path | API |
|---|---|---|
| GET | `/v1/models`, `/v1/models/{id}` | OpenAI |
| POST | `/v1/chat/completions` | OpenAI chat (SSE `chat.completion.chunk` con `stream:true`) |
| POST | `/v1/responses` | OpenAI Responses (lifecycle eventi `response.*` completo) |
| POST | `/v1/completions` | OpenAI completamento legacy |
| POST | `/v1/messages` | Anthropic Messages (`message_start` → `content_block_*` → `message_stop`) |

Il parsing delle richieste (`ChatRequestParser`) mappa i JSON delle tre API nei
tipi del motore (`ChatTurn`/`ToolSpec`/`SamplingParams`); `reasoning_effort` /
`thinking` / `reasoning.effort` abilitano il thinking. Il sandbox richiede
l'entitlement `com.apple.security.network.server` (già nel progetto).

### Distribuito (`DistributedView` + `DS4Engine/Distributed/*`)

Inferenza **distribuita su più Mac** (pipeline parallelism a range di layer,
modellata su `ds4_distributed.c`):

- ogni **worker** possiede uno slice contiguo di layer (`primo…ultimo`) con i
  **suoi soli** pesi caldi e il **suo solo** shard di KV/compressore (allocati
  esclusivamente per lo slice);
- il **coordinatore** possiede embedding, sampling e prompt; valida che la
  *route* copra tutti i layer del modello (Flash 43, Pro 61) in modo contiguo, poi per token fa scorrere lo
  **stato HC** (`nHC×nEmbd` float) attraverso i worker in ordine di layer;
- lo streaming da SSD resta attivo su ogni nodo: il vantaggio è che ogni worker
  tocca solo ~1/N degli esperti → working set caldo più piccolo → meno
  page-fault;
- **prefill a chunk** (default 32 token/frame, configurabile): il round-trip di
  rete si paga per chunk, non per token;
- **inoltro worker→worker** opzionale: lo stato HC passa direttamente tra i
  worker e il terminale risponde al listener di ritorno del coordinatore
  (serve l'IP LAN del coordinatore) — metà degli hop;
- **bit attivazioni** 32/16/8 per ridurre la banda (parti da 32 per la
  correttezza).

Uso: su ogni Mac-worker apri la scheda **Worker**, imposta porta e range di
layer e avvia; sul coordinatore (Chat → **Distribuito**) elenca i worker
(`host:porta`, uno per riga, in ordine di layer), **Connetti** e chatta. Il GGUF
deve essere presente su ogni Mac (selezionato con Sfoglia per il sandbox).

> **Permesso rete locale (macOS 15+)**: al primo uso il sistema mostra il
> prompt "DwarfStar vuole accedere alla rete locale" (testo da
> `NSLocalNetworkUsageDescription`, impostato in `project.yml`). Va concesso su
> OGNI Mac del cluster — coordinatore e worker — altrimenti le connessioni LAN
> vengono negate silenziosamente (timeout in Connetti). Verificabile in
> Impostazioni di Sistema → Privacy e Sicurezza → Rete locale.

### Benchmark (`BenchView` + `BenchController`) — nativo

Benchmark **in-process** (niente sottoprocesso): carica una propria copia del
modello (pesi mmap condivisi) e misura, a contesti crescenti, il throughput di
**prefill** e **generazione** (`InferenceService.benchmark(contextTokens:
genTokens:)`: prefilla un prompt sintetico di N token e decodifica `genTokens`,
cronometrando le due fasi). Configurazione: modello (+Sfoglia), contesto max
caricato, frontiere (start/max/passo) e token generati per punto. Il grafico
(Swift Charts) traccia le due serie token/s sull'asse dei token di contesto; un
log mostra l'avanzamento. Caricamento ed esecuzione off-main, risultati in
streaming via `AsyncStream`; **Ferma** annulla tra una frontiera e l'altra.

### Diagnostica (`DiagnosticsView` + `DiagnosticsController`)

- **Tokenizzazione nativa** (puro Swift, senza subprocess): dato un GGUF e un
  testo, mostra come il testo — inclusi gli special del protocollo DS4 — viene
  mappato in token;
- **Console motore** (`EngineConsole`): vista live dello **stderr** catturato
  del motore (diagnostica Metal/kernel), utile per capire errori di
  compilazione kernel o di esecuzione.

### Download modelli (`DownloadView` + `DownloadRunner`)

Sheet raggiungibile da **Scarica…** nella schermata di caricamento:

- download **nativo** da Hugging Face in `scriptDir/gguf` (i download parziali
  riprendono automaticamente);
- elenco di target predefiniti (`ModelCatalog.downloadTargets`) con titolo e
  dettaglio, ciascuno con pulsante **Scarica**;
- **barra di progresso** + log live; pulsanti **Annulla** e **Chiudi** (alla
  chiusura ri-scansiona i modelli disponibili).

---

## 10. Build, esecuzione e packaging

```sh
cd DS4-gui

# Progetto Xcode pulito (motore Swift puro, nessun link esterno)
make xcodeproj          # (ri)genera DwarfStar.xcodeproj via xcodegen
make xcode              # genera + apre in Xcode

# Demo da riga di comando
swift run DS4Demo                         # bring-up Metal + self-test GPU
swift run DS4Demo <model.gguf> 4          # + stream di 4 token

# App SwiftUI completa
swift run DwarfStar

# Kernel: rigenera l'embedding dopo aver modificato un .metal
make embed-kernels      # metal/*.metal → Sources/DS4Metal/Runtime/KernelSources.swift

# Packaging .app
make app                # build/DwarfStar.app (release + metal/ in Resources, firma ad-hoc)
open build/DwarfStar.app
```

`packaging/make_app.sh` costruisce l'eseguibile release, assembla il bundle,
copia i kernel `metal/` in `Contents/Resources/metal`, opzionalmente include i
binari `ds4*` sotto `Resources/bin` e firma ad-hoc l'app per l'esecuzione
locale. Per la distribuzione, firma con un'identità Developer ID e notarizza
(vedi README). Inserisci un `AppIcon.icns` in `packaging/` per dare un'icona
all'app.

### Risoluzione dei percorsi (`AppEnvironment`)

`AppEnvironment` risolve i percorsi di default in modo che lo stesso codice
funzioni sia in sviluppo (percorsi assoluti nel progetto ds4 upstream, override
con la variabile `DS4_ROOT`) sia in un `.app` (Resources del bundle). In bundle
non c'è alcun modello pre-incluso: l'utente ne seleziona uno.

---

## 11. Risoluzione dei problemi

| Sintomo | Causa probabile | Rimedio |
|---|---|---|
| *"Nessun GGUF trovato…"* nella schermata di caricamento | nessun modello in `scriptDir` o `scriptDir/gguf` | premi **Scarica…** (es. target `q2-imatrix`) o copia un `.gguf` nella cartella, poi ↻. |
| Warning arancione di memoria | modello più grande della RAM | riduci il **contesto** o usa una quant 2-bit (lo streaming SSD è già attivo). |
| Prefill fallisce / processo killed (OOM) | configurazione che non entra in RAM | applica il preset consigliato per la tua RAM; non eseguire chat e `ds4-server` insieme. |
| Output spazzatura | schema di quantizzazione disallineato | verifica con `DS4_TYPES_ONLY=1 swift run DS4Demo <gguf>`; usa un GGUF coerente con l'engine. |
| Errori Metal/kernel | compilazione kernel a runtime fallita | apri **Diagnostica → Console motore** per leggere lo stderr del motore. |
| Server non parte | binario `ds4-server` mancante | compila i binari nel progetto padre (`make`) e imposta il percorso corretto. |

---

## 12. Glossario

- **GGUF** — formato di file dei pesi del modello (quantizzati).
- **MoE (Mixture-of-Experts)** — il modello attiva solo pochi "expert" per
  token (qui 6 su 256 per layer), riducendo il calcolo per token.
- **Quant (quantizzazione)** — rappresentazione compressa dei pesi (es. 2-bit
  `IQ2_XXS`/`Q2_K`, 4-bit `Q4_K`, 8-bit `Q8`). Meno bit = meno RAM, meno qualità.
- **KV cache** — memoria delle chiavi/valori dell'attenzione, cresce con il
  contesto e occupa RAM.
- **Prefill / Decode** — fase di elaborazione del prompt vs. fase di generazione
  token-per-token.
- **Streaming SSD / per-layer** — tecniche per non tenere tutti i pesi in RAM,
  leggendoli dal disco (mmap) o evictandoli layer per layer.
- **Thinking / Ragionamento** — chain-of-thought del modello, mostrata in un
  blocco collassabile separato dalla risposta finale.
```
