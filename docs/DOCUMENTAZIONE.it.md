[English](DOCUMENTAZIONE.md) | **Italiano**

# Documentazione DwarfStar

Questa è la guida per utenti e sviluppatori di DwarfStar: cosa fa l'app, come è
organizzato il motore DS4 in puro Swift, come si usano la GUI e la CLI e come i
pannelli avanzati si integrano tra loro.

DwarfStar è al tempo stesso:

- un'applicazione SwiftUI per macOS con caricamento dei modelli consapevole
  dell'architettura;
- un runtime Metal in puro Swift il cui backend operativo oggi è DeepSeek V4,
  usato dall'app, dalla demo CLI, dal server HTTP nativo, dal pannello
  benchmark, dalla diagnostica e dai worker distribuiti.

Qwen è attualmente riconosciuto dai metadati GGUF ma non eseguibile. GLM 5.2 è
catalogato per il download ripristinabile con verifica di integrità ed è
eseguibile end-to-end col proprio backend streaming: detector registrato,
contratto GGUF rigoroso, frontend nativo tokenizer/chat e motore Metal con
gather degli esperti per token (chat, demo, server, benchmark). Vedi la
[matrice di supporto](ARCHITETTURE-SUPPORTATE.it.md) per la distinzione precisa
tra modelli riconosciuti e supportati.

Per i dettagli di più basso livello del motore, vedi
[`ARCHITETTURA-MOTORE.md`](ARCHITETTURA-MOTORE.it.md). Per la conformità di
esportazione per l'App Store, vedi [`CRITTOGRAFIA.md`](CRITTOGRAFIA.it.md).
L'indice della documentazione in [`README.md`](README.it.md) collega le guide
tematiche su inferenza, Metal, esecuzione distribuita, GUI/server,
configurazione e test.

## Indice

1. [Panoramica](#1-panoramica)
2. [Architettura a livelli](#2-architettura-a-livelli)
3. [Concetti fondamentali del motore](#3-concetti-fondamentali-del-motore)
4. [Demo CLI: `DS4Demo`](#4-demo-cli-ds4demo)
5. [App SwiftUI: `DwarfStar`](#5-app-swiftui-dwarfstar)
6. [Flusso di lavoro dell'utente](#6-flusso-di-lavoro-dellutente)
7. [Configurazione automatica e preset hardware](#7-configurazione-automatica-e-preset-hardware)
8. [Memoria, streaming e default della GUI](#8-memoria-streaming-e-default-della-gui)
9. [Pannelli avanzati](#9-pannelli-avanzati)
10. [Build, esecuzione e packaging](#10-build-esecuzione-e-packaging)
11. [Risoluzione dei problemi](#11-risoluzione-dei-problemi)
12. [Glossario](#12-glossario)

## 1. Panoramica

DwarfStar esegue attualmente DeepSeek V4 Flash e il profilo Pro Q2 a file
singolo in locale su Apple Silicon, facendo streaming da SSD dei pesi MoE
instradati. Per ogni token viene toccato solo un piccolo sottoinsieme
instradato del modello selezionato. Pro Q4 resta solo scaricabile perché il suo
artefatto di catalogo è diviso in due shard GGUF; le tre quantizzazioni
monolitiche di GLM 5.2 sono selezionabili ed eseguibili col backend streaming
GLM, con gather degli esperti per token da SSD; il Pro Q2
distribuito usa percorsi orizzontali e a shard di esperti consapevoli della
geometria, con la validazione numerica multi-Mac su modello reale ancora in
sospeso. La selezione del backend è esplicita, quindi una futura
implementazione Qwen potrà fornire tokenizer, mappatura dei tensori e decoder
propri senza modificare l'hot path DeepSeek. Il progetto evita il vecchio
approccio a sottoprocesso: l'app SwiftUI dialoga direttamente con il motore
Swift tramite actor e stream asincroni.

La stessa implementazione del motore alimenta:

- la tab Chat interattiva;
- il tool calling e i sub-agent isolati;
- il server HTTP nativo compatibile OpenAI/Anthropic;
- il coordinatore distribuito e il runtime dei worker;
- i pannelli benchmark e diagnostica;
- l'eseguibile a riga di comando `DS4Demo` per bring-up e profiling.

Il vincolo di progettazione importante è che l'app deve comportarsi come la
demo CLI: stesso parser del modello, stesso tokenizer, stesso grafo di decode,
stessa logica di sampling, stesse assunzioni di streaming. La GUI aggiunge
gestione dello stato e comodità, non una seconda implementazione di inferenza.
In modalità GUI locale, Chat, Server e Benchmark riusano il singolo
`InferenceService` caricato in Settings, così l'app non raddoppia i buffer Q4
residenti, i buffer `mlock`ati o lo scratch GPU sui Mac con poca RAM.

## 2. Architettura a livelli

```text
DwarfStar (SwiftUI)
  Chat · Settings · Agents · Project · Tuning · Server · Worker · Benchmark · Diagnostics
        |
DS4Engine
  actor InferenceService · server HTTP · tool · agenti · ProjectCache
  DiskKVStore · coordinatore/worker distribuito · downloader dei modelli
        |
DS4Core + DS4Metal
  parser GGUF mmap · tokenizer · sampler · forma del modello · rendering/parser DSML
  runtime Metal · GPUTensor · contesto del grafo · kernel · StreamingDecoder
        |
metal/*.metal
  Fonte di verità dei kernel, incorporata in Swift al momento della build
```

### Mappa dei sorgenti

| Percorso | Responsabilità |
|---|---|
| `Sources/DS4Core/` | Swift puro, nessuna dipendenza da Metal: parser GGUF, tokenizer, sampler, forma del modello, rendering della chat, parsing DSML. |
| `Sources/DS4Metal/` | Runtime Metal, tensori, wrapper dei kernel, grafo di decode, streaming decoder, slot-cache degli esperti, caricamento dei pesi no-copy. |
| `Sources/DS4Engine/` | Livello di servizio applicativo: actor di inferenza, registro dei tool, disk KV, downloader, runtime distribuito. |
| `Sources/DS4Demo/` | Demo CLI ed eseguibile diagnostico per bring-up, audit GGUF e streaming di token. |
| `Sources/DwarfStar/` | Applicazione SwiftUI, organizzata con una cartella per tab o area funzionale. |
| `metal/` | Sorgente modificabile dei kernel Metal. `make embed-kernels` rigenera la versione Swift incorporata. |
| `templates/` | Copia commentata del chat template del modello per revisione e documentazione. |
| `scripts/` | Helper di analisi GGUF e script di embedding dei kernel. |
| `docs/` | Documentazione dettagliata del progetto. |
| `Tests/` | Test di unità, kernel, grafo, parser e servizi. |

## 3. Concetti fondamentali del motore

### Streaming da SSD

DwarfStar usa lo streaming da SSD come modalità di caricamento del modello
predefinita e unica. I pesi non instradati sono viste `mmap` e possono essere
tenuti caldi dalla page cache del sistema operativo. Gli esperti instradati
vengono raccolti per token: solo gli esperti selezionati per il layer corrente
vengono copiati o letti nel percorso GPU.

Lo scopo non è far "entrare" in RAM un modello da 284B. Lo scopo è mantenere il
working set attivo abbastanza piccolo da permettere alla macchina di avanzare
token per token.

### MoE ed esperti instradati

DeepSeek V4 Flash e Pro sono profili Mixture-of-Experts. Il router seleziona un
insieme top-k di esperti per ogni token e layer. Il numero attivo predefinito è
6. Flash instrada tra 256 esperti con scala 1.5; Pro instrada tra 384 con scala
2.5. Ridurre il numero di esperti attivi con `DS4_ACTIVE_EXPERTS` riduce l'I/O
ma cambia la qualità del modello e va trattato come una modalità degradata o un
esperimento diagnostico.

I primi tre layer sono instradati per hash: la selezione degli esperti deriva
dall'id del token tramite la tabella `ffn_gate_tid2eid`, esattamente come nel
riferimento C, non dallo stato nascosto. Questo è importante per il prefetch
degli esperti (la selezione è nota prima del calcolo) e per gli shard
distribuiti che coprono quei layer.

Al caricamento il motore valida i metadati GGUF nello stesso modo del loader C
(un port di `config_validate_model`) e seleziona il profilo dichiarato. Il
runtime locale costruisce una geometria immutabile da quella configurazione:
Flash usa 43 layer e 256 esperti, mentre Pro usa 61 layer e 384. Il router Pro
riempie con padding un dispatch bitonico a 512 lane sopra l'esperto 383, quindi
non applica mai la geometria Flash né legge oltre la riga di probabilità Pro.
Anche la parte numerica segue il riferimento C: gli epsilon di RMSNorm e
Hyper-Connection sono per default `1e-6`, in accordo con `DS4_DEFAULT_RMS_EPS`
e `DS4_DEFAULT_HC_EPS` in `ds4.c`.

### Riuso della KV

Le conversazioni sono append-only. Il servizio tiene traccia esattamente di
quali id di token sono già stati committati nella KV del decoder. Un nuovo
turno utente esegue il prefill solo del nuovo suffisso. Se una generazione
viene interrotta, il servizio può ricostruire la KV dagli id committati prima
di continuare, perché tutto il testo committato è noto esattamente.

### Tool Calling

Il modello usa DSML, un formato di tool call in stile XML aperto dal token di
controllo `｜DSML｜`. DwarfStar rende le dichiarazioni dei tool attraverso la
stessa superficie di template su cui il modello è stato addestrato e analizza
le chiamate generate con `ToolCallParser`.

I tool integrati vengono eseguiti automaticamente. Le chiamate non integrate
possono essere presentate per risultati manuali. Le chiamate ai sub-agent sono
intercettate dal motore ed eseguite in un contesto di decoder isolato.

### Imatrix di utilizzo degli esperti

Il motore registra quali esperti sono stati selezionati dal router. Questo
profilo di utilizzo è persistito per modello e per agente. Serve a
pre-riscaldare gli slot della cache degli esperti e, quando l'allocazione
guidata dall'uso è attiva, ad assegnare più slot ai layer il cui instradamento
è più concentrato.

## 4. Demo CLI: `DS4Demo`

`DS4Demo` è un harness minimale del motore. Evita intenzionalmente la GUI ed è
utile quando serve un percorso di misura pulito.

### Utilizzo

```sh
swift run DS4Demo
swift run DS4Demo /path/model.gguf 4
swift run DS4Demo /path/model.gguf 32 "Explain RoPE briefly"
```

La prima forma avvia solo Metal ed esegue il self-test. La seconda e la terza
aprono un GGUF, eseguono uno smoke test forward e generano token se `maxNew` è
maggiore di zero.

### Comportamento passo per passo

1. Crea `MetalRuntime`, compilando i kernel incorporati da `KernelSources.swift`.
2. Esegue il self-test GPU.
3. Se non è stato passato un percorso GGUF, esce.
4. Apre il GGUF con mmap no-copy e senza prefetch CPU.
5. Facoltativamente esegue la diagnostica disco `DS4_DIAG` e il report sulla
   presenza di MTP.
6. Facoltativamente esegue l'audit dtype/tokenizer `DS4_TYPES_ONLY` ed esce.
7. Rileva la quantizzazione MoE instradata e i layer a precisione mista.
8. Crea `StreamingDecoder.fromGGUFExpertCachedMapped`.
9. Carica o inizializza l'imatrix di utilizzo.
10. Esegue un forward di un token e verifica che i logit siano finiti.
11. Se `maxNew > 0`, tokenizza il prompt con il chat template del modello.
12. Esegue il prefill dei token del prompt layer-major.
13. Decodifica in greedy i token, invia i byte in streaming su `stdout` e
    registra i tempi su `stderr`.
14. Salva l'imatrix di utilizzo e stampa il profilo di decode.

Vedi `Sources/DS4Demo/README.md` per esempi diagnostici e il
[Riferimento di configurazione](../README.it.md#riferimento-di-configurazione) nel
README di radice per l'elenco completo delle variabili d'ambiente `DS4_*`.

## 5. App SwiftUI: `DwarfStar`

L'app è un wrapper con stato attorno allo stesso motore. Aggiunge selezione del
modello, bookmark security-scoped, chat persistenti, tool, agenti,
indicizzazione di progetto, modalità server, modalità distribuita, diagnostica
e pannelli di tuning.

### Tab dell'app

| Tab | Ruolo |
|---|---|
| **Chat** | Conversazione principale locale o distribuita. |
| **Settings** | Modello condiviso, contesto, modalità di esecuzione, manopole di memoria/I/O, caricamento locale, route distribuita. |
| **Agents** | Ruoli modificabili con prompt, icone e allow-list di tool. |
| **MCP** | Configura server MCP esterni i cui tool estendono quelli integrati. |
| **Project** | Libreria dei progetti importati e cache del progetto attivo. |
| **Tuning** | Controlli della slot-cache degli esperti e imatrix di utilizzo. |
| **Server** | Server HTTP nativo compatibile OpenAI/Anthropic. |
| **Worker** | Esegue questo Mac come worker distribuito; il coordinatore assegna GGUF, impostazioni e fetta di layer. |
| **Benchmark** | Grafici nativi di throughput e di accuratezza next-token teacher-forced. |
| **Diagnostics** | Dump del tokenizer e ispezione di chat template/formato dei tool. |

### Stato della chat: `ChatStore`

`ChatStore` è il view-model principale della chat locale. Possiede o fa da
proxy a:

- impostazioni condivise dell'app (`modelPath`, `contextSize`);
- l'`InferenceService` locale;
- sessioni di chat persistenti;
- allegati di file di testo in attesa;
- agente selezionato e impostazioni dei tool;
- impostazioni di sampling;
- manopole di memoria come expert bundle, pread degli esperti, streaming dei
  pesi densi, proiezioni di attenzione Q4, `mlock`, disk KV, ring raw-KV e
  slot della cache degli esperti;
- informazioni di tuning e controlli sull'uso degli esperti;
- stato di generazione live e consumo dello stream.

Le chat persistenti sono salvate come file JSON in Application Support. Una
chat riaperta è visibile immediatamente, ma la KV del motore potrebbe non
contenere ancora quella conversazione. Al prossimo invio, `ChatStore` esegue un
re-prime dalla cronologia visibile; il disk KV può ripristinare i prefissi già
noti.

### Streaming di eventi

`InferenceService` restituisce eventi asincroni: stato, token di ragionamento,
testo visibile, testo dello stream dei tool, tool call completate ed errori.
`ChatStore` riflette questi eventi in righe `UIMessage`. Le tool call possono
attivare l'esecuzione automatica dei tool, la raccolta manuale dei risultati o
l'esecuzione di sub-agent.

## 6. Flusso di lavoro dell'utente

### Passo 1 — Selezionare e configurare il modello

Apri **Settings** e scegli un GGUF con **Browse**. Nelle build sandboxed questo
è obbligatorio, perché il picker concede un bookmark security-scoped. Il
bookmark viene salvato, così lo stesso modello può riaprirsi al prossimo avvio.

Le build sandboxed possono leggere SOLO il file `.gguf` scelto: le cache
sidecar che gli stanno accanto (`.q4dense`, `.expbundle`, per esempio costruite
dalla demo CLI) sono invisibili, quindi l'app le ricreerebbe nel proprio
container — un primo caricamento lento e gigabyte duplicati.
**Grant Model Folder Access…** concede l'intera cartella del modello, così quei
sidecar vengono riusati direttamente; farlo una volta per ogni cartella di
modello è raccomandato.

Seleziona una dimensione di contesto. Il default è consapevole della RAM: i
sistemi con poca RAM hanno come default un contesto piccolo, perché la memoria
KV cresce con il contesto e compete con la page cache necessaria allo streaming
degli esperti.

Impostazioni facoltative di memoria e I/O:

- **Slot della cache degli esperti:** budget LRU di esperti residenti su GPU
  per layer.
- **Esperti via pread diretto:** bypassa la page cache per gli slab degli
  esperti.
- **Sidecar expert bundle:** salva gli slab gate/up/down di ogni esperto in
  modo contiguo, così un miss di cache è una singola lettura sequenziale. Il
  sidecar duplica la regione degli esperti su disco; le build sandboxed lo
  creano in Application Support quando non possono scrivere accanto al GGUF.
- **Streaming dei pesi densi:** invia in streaming i pesi dei layer densi
  attraverso un piccolo ring di staging un layer avanti rispetto al calcolo. È
  l'alternativa a bassa RAM preferita rispetto al tenere residenti tutti i pesi
  densi.
- **Blocco dei buffer caldi in RAM:** `mlock()` best-effort per i buffer caldi
  residenti, così il compressore di memoria non trasforma ogni token in una
  lenta rilettura.
- **Proiezioni di attenzione Q4:** percorso di velocità lossy che riquantizza a
  Q4_K le proiezioni di attenzione più grandi e le mette in cache in
  Application Support.
- **Disk KV:** salva checkpoint dei prefissi KV su disco per il riuso tra
  sessioni e server.
- **Ring raw-KV:** mantiene la raw KV a finestra scorrevole a dimensione
  costante.
- **Look-ahead degli esperti:** esegue il prefill speculativo degli slot di
  cache del layer successivo mentre il layer corrente calcola (0 = solo layer
  hash, che sono sempre prefetchati esattamente).

Ogni manopola dietro questi interruttori, più quelle solo-CLI, è documentata
nel [Riferimento di configurazione](../README.it.md#riferimento-di-configurazione) del
README di radice, che è il riferimento completo dei parametri.

### Passo 2 — Caricare

Premi **Load Model**. L'app apre il GGUF, inizializza Metal, costruisce il
decoder, applica le manopole d'ambiente di memoria/I/O, configura il disk KV,
applica l'agente selezionato e passa allo stato pronto. Il tempo di caricamento
dipende dalla dimensione del modello e dal fatto che debbano essere creati
sidecar o cache una tantum, in particolare l'expert bundle e la cache Q4 densa.

### Passo 3 — Chattare

In **Chat**, digita un messaggio e premi Invio. Lo store:

1. incorpora nel turno utente gli eventuali allegati di file di testo in
   attesa;
2. aggiunge il messaggio utente visibile alla trascrizione;
3. invia al motore un suffisso incrementale oppure una cronologia re-primed;
4. riceve in streaming ragionamento, testo visibile e markup delle tool call;
5. esegue i tool integrati quando serve;
6. persiste la sessione di chat e aggiorna l'uso del contesto.

Il sampling della chat usa la temperatura e la penalità di ripetizione
configurate con un top-k fisso di 40 (il default di llama.cpp). Campionare
sull'intero vocabolario DeepSeek da 129k lasciava che la coda rumorosa di un
modello a 2 bit scegliesse occasionalmente un token cinese e trascinasse
l'intera risposta in cinese; il tetto rimuove quella coda senza penalizzare la
varietà. Il server HTTP continua a rispettare il `top_k` per richiesta.

### Thinking, Stop e New Chat

**Thinking** attiva/disattiva il rendering del prompt consapevole del
ragionamento. I token di ragionamento sono mostrati in un blocco richiudibile.
**Stop** annulla la generazione corrente. **New Chat** avvia una nuova
conversazione persistita con l'agente corrente.

### Tool

Il foglio dei tool controlla se i tool vengono dichiarati e quali integrati
sono abilitati. Le dichiarazioni compatte riducono i token di prefill inviando
una dichiarazione `name(parameters)` più corta, mentre le dichiarazioni
complete restano più vicine al template di addestramento.

I risultati dei tool vengono reinseriti nella conversazione come turni
tool-result lato utente. I tool integrati vengono eseguiti automaticamente. Ai
tool sconosciuti si può rispondere manualmente.

### Analizzare un repository GitHub dalla chat

Gli agenti Coding e Code espongono `github_clone`: il modello scarica un
repository GitHub PUBBLICO come tarball HTTPS (la richiesta è vincolata a
`codeload.github.com`, owner/nome/ref sono validati rigorosamente e l'archivio
ha un tetto di dimensione), lo estrae in
`Application Support/DwarfStar/github-projects` — eventuali symlink
nell'archivio vengono rimossi, e i tool di progetto rifiutano indipendentemente
i percorsi che risolvono fuori dalla radice — e lo importa nell'indice dei
progetti come progetto ATTIVO, sostituendo quello corrente. Il risultato del
tool è deliberatamente piccolo: un albero di directory con i conteggi dei file
più i file di documentazione da leggere per primi (README, file `.md` di
radice, `docs/`).

Da lì il modello lavora in modo parsimonioso con il contesto, allo stesso modo
di una cartella importata dalla tab Project: struttura via
`project_tree`/`project_list`, nomi dei file via `project_find`, ricerca nel
codice via `project_search` (facoltativamente limitata a una sottocartella) e
letture mirate via `project_read` in blocchi con tetto di righe. Nulla entra
nella KV della conversazione finché un tool non restituisce effettivamente
contenuto, quindi un repository grande può essere analizzato senza prefillarlo
nel contesto. `github_clone` non è concedibile ai sub-agent perché sostituisce
il progetto attivo condiviso.

I repository clonati sono anche cittadini di prima classe nella GUI: appaiono
automaticamente nella **tab Project** (e nei menu Project della chat e della
modalità distribuita) come voci di libreria contrassegnate da un'icona di
download e denominate `<owner>-<repo> (GitHub)`, già attive subito dopo il
clone. Da lì puoi passare avanti e indietro tra cloni e cartelle importate
dall'utente con **Activate**, esattamente come qualsiasi altro progetto.
Rimuovere una voce clonata elimina la copia da
`Application Support/DwarfStar/github-projects` (può essere riclonata in
qualsiasi momento); rimuovere una cartella importata dall'utente non tocca mai
il disco.

### Server MCP

La tab **MCP** collega l'app, come CLIENT Model Context Protocol, a server di
tool esterni. Ogni server configurato è:

- **stdio** — l'app avvia il server (`npx`, `uvx` o qualsiasi eseguibile) come
  processo figlio e parla JSON-RPC delimitato da newline su stdin/stdout;
- **HTTP** — trasporto Streamable-HTTP: il JSON-RPC viene inviato in POST
  all'URL del server (con header facoltativi come un token Authorization), per
  server remoti o locali che la sandbox non può avviare.

Alla connessione l'app esegue l'handshake `initialize` e recupera
`tools/list`. Ogni tool del server appare quindi accanto a quelli integrati —
nel foglio Tool della chat sotto "MCP Tools" e nell'elenco dei tool di ogni
agente — con il nome namespaced `mcp_<server>_<tool>` (es. `mcp_fs_read_file`).
Quando il modello ne chiama uno, l'app inoltra `tools/call` al server e
reimmette il risultato testuale nella conversazione; gli errori e i timeout del
server tornano come risultati di errore a cui il modello può reagire. Ai
sub-agent non possono essere concessi tool MCP.

Le configurazioni persistono tra i lanci e possono essere importate/esportate
nel JSON standard `{"mcpServers": …}` condiviso con Claude Desktop, Cursor e
VS Code. Nota per le build sandboxed (App Store): i processi figli stdio
ereditano la sandbox dell'app, quindi i server che necessitano di ampio accesso
a file o rete dovrebbero girare fuori dall'app ed essere raggiunti via HTTP. Le
build di sviluppo (`swift run`, `make app`) non sono sandboxed. Il codice lato
motore vive in `Sources/DS4Engine/Tools/MCP/` (`MCPManager`, `MCPClient`,
trasporti); il pannello è
`Sources/DwarfStar/Features/Settings/Views/MCPServersView.swift`.

### Sub-agent

`subagent_run` delega una domanda mirata a un contesto di decoder isolato. Il
sub-agent può ricevere un file di progetto, l'intera mappa del progetto o un
compito semplice. Usa una propria cache di prefissi KV con chiave sul
contenuto, può eseguire tool internamente e restituisce una risposta concisa.
La chat principale memorizza solo la chiamata e la risposta.

### Allegati di file di testo

Il composer può importare file di testo. I file vengono decodificati come UTF-8
o Latin-1 e incorporati nel prossimo prompt utente tra delimitatori chiari. La
trascrizione mostra badge con i nomi dei file; il contenuto completo del file
entra nel modello solo quando invii.

## 7. Configurazione automatica e preset hardware

L'app rileva la RAM fisica e propone default conservativi:

| Fascia di RAM | Default |
|---|---|
| `<24 GB` | Quantizzazione a 2 bit, contesto 4096, streaming da SSD previsto. |
| `<80 GB` | Quantizzazione a 2 bit, contesto 8192, la maggior parte dei pesi caldi può restare in page cache. |
| `<200 GB` | Quantizzazione a 2 bit, contesto 32768. |
| RAM molto grande | Q4 può entrare; contesto 32768. |

Queste fasce di RAM sono raccomandazioni di capacità, non profili di parità
numerica: la quantizzazione del modello selezionato influisce sulla qualità, e
i percorsi di velocità Q4 facoltativi qui sotto sono deliberatamente lossy. Il
loro scopo è ridurre la pressione sulla memoria. Puoi comunque alzare
manualmente il contesto fino a 1M. La capacità KV/compressore mantiene quel
limite superiore, mentre lo scratch di attenzione dipendente dal contesto
cresce a partire dall'**high-water mark dei token usati** ed è limitato da
esso: selezionare un limite grande non impegna più lo scratch massimo alla
prima domanda.

Oltre ai preset statici, Settings offre tre strumenti guidati dalle misure che
regolano il motore per la macchina reale. Tutti e tre richiedono un modello
caricato, rendono la chat inutilizzabile mentre girano e stampano un report nel
pannello e nel log del motore.

### Benchmark di prefill (rapido ~3 min / completo ~10 min)

La sezione **Benchmark** in Settings misura le due manopole di prefill che il
motore rilegge a ogni chiamata di prefill — `DS4_PREFILL_UNION` (numero massimo
di esperti per gruppo in una union di prefill) e `DS4_PREFILL_CHUNK` (token per
chunk) — poi applica e persiste la combinazione più veloce:

- **Quick** confronta union 192 vs 256 su un prefill da 256 token. Salta
  deliberatamente union 64: a scala corta il rumore può farla vincere, mentre
  sui prefill reali è il valore catastrofico.
- **Full** confronta union 64/192/256 su un prefill da 512 token, poi chunk
  512 vs 1024 con la union vincente su un prefill da 1024 token (sotto i 512
  token un secondo chunk non esiste, quindi la modalità quick non può
  misurarlo).

I valori applicati sono mostrati nella riga "Attivi (prefill)".

### Auto-tune per macchina basato sul record

**Auto-tune macchina** usa fino a due passate di coordinate-ascent per cercare
i controlli di caricamento a qualità esatta per questo chip, questa RAM e
questo SSD. Il manifest copre la policy di cache a quantizzazione mista, gli
slot esperti limitati dalla RAM, l'allocazione guidata dall'uso rispetto a
quella uniforme, la profondità di coda del `pread` degli esperti, il read-ahead
denso, l'FFN asincrono, il look-ahead degli esperti e le occupancy NSG a
partizione di righe per MoE/dense-Q4. La NSG Q8 resta fissa perché cambiare la
sua partizione della riduzione K può cambiare i bit bassi Float32; solo il
profilo numerico opt-in del tuner di processo la esplora. I controlli ordinati
usano una camminata direzionale prima verso l'alto. In particolare, gli slot
della cache esperti avanzano un passo di manifest alla volta: dopo una vittoria
il vicino inferiore viene saltato e la camminata continua verso l'alto; la
prima sconfitta ferma quel parametro senza sondare valori più grandi. Il vicino
inferiore viene provato solo se il passo iniziale verso l'alto perde. Le
griglie hardware non monotone vengono comunque esplorate per intero.

- La radice calda già caricata viene misurata una volta con una sonda che
  ripristina il prompt. Poiché la sua KV sintetica sostituisce la KV della
  conversazione, una chat non vuota viene contrassegnata per il re-prefill
  automatico della cronologia al prossimo invio.
  Ogni candidato unico viene caricato, riscaldato e misurato indipendentemente
  al massimo una volta; le visite successive riusano la cache locale della run
  senza un altro reload.
- Il decode è greedy. La metrica primaria è la mediana della velocità dei token
  a regime, non un p99 per token. L'osservazione valida completa con la mediana
  di decode più alta è il record. Un candidato deve essere strettamente più
  veloce, e il prefill non può regredire di oltre l'8%; i campi di run separate
  non vengono mai combinati.
- Gli id dei token generati e ogni frame di logit a vocabolario completo
  catturato devono essere bit-exact sia rispetto all'incumbent corrente sia
  rispetto alla radice iniziale immutabile. Questo gate cumulativo sulla radice
  impedisce che piccole variazioni numeriche si accumulino attraverso
  promozioni successive.
- La stabilità deve restare almeno 0.75 secondo il rapporto coda/testa del
  decode. Una prova viene rifiutata dopo più di 128 MiB di swapout nella sua
  finestra misurata a regime. Init a freddo, warmup e il primer scartato
  formano una finestra di setup diagnostica separata e non consumano quel
  budget a regime. Una barriera di assestamento fail-closed dopo il primer
  ancora la finestra misurata; campioni del contatore di swap mancanti,
  azzerati o non validi invalidano l'osservazione. Il pavimento normale di RAM
  libera è il 10%; quando la radice già caricata è sotto di esso ma conserva
  almeno 512 MiB, la modalità vincolata fissa il pavimento al maggiore tra
  512 MiB e la radice meno un punto percentuale, e vieta delta residenti
  stimati positivi. Il ripristino della radice e quei candidati limitati
  conservano la riserva known-loadable di 512 MiB dopo il teardown; i candidati
  sconosciuti o che accrescono la memoria richiedono comunque 12%/1.5 GiB prima
  dell'init. Il profilo di utilizzo per modello/per agente è congelato per
  l'intera run, e il ring Raw-KV resta abilitato come vincolo fisso a bassa
  RAM.

Non c'è un ciclo di ripetibilità della baseline né una ri-misura finale del
throughput. Questo elimina la costosa sequenza di reload A→B→B→A e rende una
configurazione rivisitata un `CACHE HIT`. La prima firma di qualità a logit
completi e il pavimento RAM della run restano immutabili. Rimuovere le
ripetizioni rimuove anche la stima di bias d'ordine/rumore: un record alto
fortunato può produrre falsi negativi conservativi, ma i gate fail-closed su
qualità exact-root, prefill, stabilità, RAM e swap restano.

Le impostazioni candidate vengono applicate solo all'ambiente del processo
durante la ricerca; non toccano UserDefaults. La configurazione finale viene
persistita una sola volta, solo dopo che il suo record in cache è stato
validato e il vincitore completa un warmup riuscito con l'agente attivo. Prima
del commit, il motore completamente pronto deve anche superare una sonda steady
post-warmup con lo stesso limite di 128 MiB; il suo swap di setup è registrato
separatamente. L'adozione è supportata da transazione: il recovery all'avvio
ripristina lo snapshot iniziale completo se l'installazione o il commit vengono
interrotti. **Stop** annulla la run e ripristina il motore iniziale. Le run
complete e parziali scrivono `report.md` e `results.json` in
`Application Support/DwarfStar/autotune`, e Settings può mostrare il report nel
Finder.

### "Align to fast demo config" e migrazioni una tantum

Il pulsante **Align to fast demo config** applica lo snapshot del preset
misurato del **2026-07-13**, con il successivo allineamento a bassa RAM: 22
slot esperti/layer; pread, dense streaming, `mlock`, bundle e MetalIO ON; Q4
completo (`DS4_DENSE_Q4`, `DS4_QKV_Q4`, `DS4_SHARED_Q4`) ON;
union/chunk/route-batch di prefill `256/512/32`; pread split 4; dense-ahead 2;
look-ahead 0; NSG Q8, MoE e dense-Q4 4; raw ring ON. L'app esegue anche
migrazioni una tantum che spostano i default persistiti da build sperimentali
più vecchie a quello snapshot. Le modifiche manuali successive e i risultati di
auto-tune specifici della macchina completamente validati restano autorevoli.

## 8. Memoria, streaming e default della GUI

La tabella seguente è uno **snapshot del 2026-07-13** del preset GUI misurato
per un M1 Pro con 16 GB e RAM libera sufficiente. È un punto di partenza, non
un'affermazione che gli stessi valori siano ottimali su ogni GPU Apple o SSD:

| Impostazione | Default GUI | Perché |
|---|---|---|
| Cache esperti | `22` slot/layer | Punto misurato che ha mantenuto caldi più esperti instradati senza collasso di memoria osservato sulla macchina dello snapshot. |
| Pread esperti | ON sotto 24 GB di RAM | Bypassa la page cache per gli slab degli esperti, così i pesi densi non vengono sfrattati. |
| Expert bundle | ON | Trasforma le letture sparse dei miss degli esperti in un singolo burst contiguo. |
| Streaming dei pesi densi | ON sotto 24 GB di RAM | Usa un piccolo ring di staging invece di un set denso residente multi-GB. |
| `mlock` dei buffer caldi | ON | Evita il churn del compressore di memoria sui buffer Metal condivisi caldi. |
| MetalIO | ON | Tenta il caricamento diretto file→buffer Metal e ripiega automaticamente quando il suo throughput misurato è sotto la soglia configurata. |
| Q4 completo | `DENSE_Q4`, `QKV_Q4`, `SHARED_Q4` ON | Preset deliberatamente lossy che copre le grandi proiezioni di attenzione, q_a/kv e le proiezioni degli esperti condivisi. |
| Raggruppamento prefill | union `256`, chunk `512`, route batch `32` | Snapshot di raggruppamento misurato; il benchmark di Settings può sostituire questi valori hot-reloadable. |
| Disk KV | ON | Riusa i prefissi noti tra chat, ricarichi e richieste server. |
| Ring raw-KV | ON | Vincolo GUI fisso a bassa RAM: la raw KV resta limitata alla finestra scorrevole NSA. Questo non cambia il default del motore puro. |
| Pipeline FFN asincrona | ON | Committa l'FFN instradato in modo asincrono, così la GPU non si svuota più tra i layer; la parità è stata verificata per lo snapshot citato, non promessa come contratto bit-exact universale. |
| Dense-stream ahead | `2` | Il ring di staging legge un layer avanti rispetto al calcolo. |
| Look-ahead esperti | `0` | Prefill speculativo misurato neutro; i layer hash sono sempre prefetchati esattamente. |

La maggior parte di questi valori è persistita e si applica al prossimo
caricamento del modello. L'app esegue migrazioni una tantum dai default
sperimentali più vecchi a questo profilo misurato (vedi la Sezione 7); le
modifiche future dell'utente sono preservate. L'elenco completo delle manopole
d'ambiente del motore, incluse quelle solo-CLI, vive nel
[Riferimento di configurazione](../README.it.md#riferimento-di-configurazione) del
README di radice.

### Page cache vs buffer wired

L'mmap no-copy lascia decidere al sistema operativo cosa resta residente. È
eccellente quando la RAM è sufficiente, ma sui sistemi piccoli lo streaming
degli esperti può sfrattare i pesi densi. `DS4_EXPERT_PREAD=1` bypassa la page
cache per gli slab degli esperti, mentre `DS4_DENSE_STREAM=1` invia in
streaming i tensori dei layer densi attraverso un piccolo ring di staging.
`DS4_RESIDENT_DENSE=1` è ancora disponibile per esperimenti CLI e sistemi
ricchi di RAM, ma l'app decide automaticamente il denso residente in base alla
RAM e preferisce il dense streaming sui sistemi a bassa RAM.

### Expert bundle

`DS4_EXPERT_BUNDLE=1` usa un sidecar in cui gli slab gate, up e down di ogni
esperto instradato sono contigui. La parte numerica non cambia: il sidecar è
una copia riordinata di byte già presenti nel GGUF. Il compromesso è lo spazio
su disco. Su un modello Flash a 2 bit il sidecar può essere di decine di GB,
quindi l'app controlla lo spazio e registra nel log quando la creazione viene
saltata.

Il pannello Settings può generare il bundle immediatamente. Nelle build
sandboxed, l'app prova prima a riusare un `<model>.expbundle` leggibile accanto
al GGUF; se non può scrivere lì, lo costruisce in
`Application Support/DwarfStar/expert-bundle`.

### Dense streaming, `mlock` e cache Q4 densa

Il dense streaming legge il layer `i+1` mentre la GPU calcola il layer `i`. Usa
`pread + F_NOCACHE`, evita il churn della page cache e ha precedenza sui pesi
densi residenti. `DS4_DENSE_AHEAD` controlla la profondità del ring di staging
(default 2 = un layer avanti, l'ottimo misurato; l'auto-tune esplora 1-3).

Due raffinamenti lossless sono ON per default dentro il dense stream:

- `DS4_RESIDENT_COMP` mantiene residenti le quattro proiezioni del compressore
  NSA invece di riceverle in streaming — vengono lette a ogni token su 41/43
  layer Flash e su tutti i 61 layer Pro; la stima di ~0.6 GB è specifica di
  Flash. È la singola rilettura ripetuta più densa nello stream. Stessi byte,
  numerica identica; `=0` ripristina lo streaming completo come ripiego quando
  la RAM è stretta.
- `DS4_LAZY_IDX` rinvia le proiezioni di SCORING dell'indexer in base al
  **contesto effettivamente usato**, non alla capacità di contesto configurata.
  Prima del confine sparse sono omesse dal dense stream; alla prima attivazione
  vengono caricate una volta in buffer residenti e poi riusate. Questo evita
  circa 360 MB/token di letture SSD premature su Flash anche quando `maxKeys` è
  grande. Il compressore ricorrente dell'indexer resta attivo dall'inizio.

Lo scratch di attenzione/indexer dipendente dal contesto segue la stessa policy
live. Parte dalle righe raw della finestra scorrevole più le righe compresse
emesse, cresce geometricamente solo a un nuovo high-water mark e non supera mai
il tetto configurato. Il riuso dei buffer mantiene il percorso normale del
token privo di allocazioni dopo ogni passo di crescita.

Se il percorso sparse dell'indexer si attivi o meno è governato da
`DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` (default 1024, stesso override
d'ambiente e valori ammessi del C): il decode mantiene l'attenzione densa su
tutte le righe compresse finché il loro numero non supera la soglia, perché
intorno alla frontiera dei ~2K il setup score/top-k del percorso sparse costa
più della scansione densa più piccola. Alla soglia predefinita 1024 e top-K
512, 4096 e 4099 chiavi live restano sul percorso denso; 4100 è il primo numero
di chiavi live che può attivare il caricamento dello scorer.

`DS4_MLOCK=1` blocca i buffer caldi in modalità best-effort. Include i pool
della cache esperti, lo staging del dense stream, i buffer residenti della
testa di output, i buffer densi Q4 e il ring raw-KV quando `DS4_RAW_RING=1`. Il
mancato blocco non è fatale; significa solo che macOS può comprimere o paginare
quei buffer.

`DS4_DENSE_Q4=1` è un percorso di velocità lossy. Riquantizza a Q4_K le più
grandi proiezioni di attenzione Q8, mette in cache il risultato in
`Application Support/DwarfStar/q4-cache` e mantiene residenti i buffer ridotti.
La conversione del primo avvio salva checkpoint parziali della cache tra i
batch, quindi un caricamento interrotto a metà riquantizzazione riprende dai
tensori completati al prossimo avvio invece di ripartire da zero (il progresso
ed eventuali errori di scrittura della cache sono registrati nel log del motore
come righe `DS4 q4cache:`). Disabilitalo quando vuoi il comportamento più
vicino al full-Q8 anziché il massimo throughput su singola macchina.

### Pipeline FFN asincrona e manopole dei kernel GPU

`DS4_ASYNC_FFN=1` (default ON) committa il lavoro dell'FFN instradato in modo
asincrono, così la GPU non si svuota tra i layer. La coda Metal è in-order,
quindi l'output è identico token per token al percorso sincrono — misurato +10%
su M1 Pro. Impostarlo a `0` ripristina le attese sincrone storiche ed è tenuto
come paracadute di debug.

`DS4_FUSED_HC` (default ON) fonde la coda di riduzione della Hyper-Connection —
split + collapse + RMSNorm — in un solo dispatch invece di tre; gira due volte
per layer, risparmiando ~170 dispatch/token. Stessa matematica; cambia solo
l'ordine di riduzione della RMSNorm (classe ±1 ulp). `=0` ripristina il
percorso non fuso.

`DS4_Q8_NSG` imposta i simdgroup per threadgroup nei matvec Q8. Cambia
l'occupancy e il modo in cui le somme parziali partizionano la riduzione K:
viene valutata la stessa operazione di matrice, ma gli ultimi bit
floating-point non sono garantiti identici. L'ottimo dipende dal numero di core
GPU: 4 è il riferimento (il migliore su M1 Pro); GPU più larghe (Max/Ultra)
possono preferire 6-8. Il motore lo rilegge a ogni caricamento del modello, ed
è così che l'auto-tune lo esplora.

### KV cache

La memoria KV cresce con il contesto. Il disk KV salva checkpoint di prefissi,
così richieste o sessioni successive con lo stesso prefisso possono
ripristinare invece di rifare il prefill. È particolarmente utile per richieste
HTTP stateless che rispediscono lo stesso prefisso di conversazione.

I checkpoint vengono spostati tra disco e buffer KV in batch per layer, in
entrambe le direzioni, così un ripristino non materializza mai l'intero file in
RAM: ogni layer viene letto (F_NOCACHE), importato nel decoder e liberato prima
che il successivo venga caricato — il picco di memoria è un layer invece di ~3×
la dimensione del checkpoint del vecchio percorso che caricava tutto. Il
salvataggio è simmetrico: lo writer in background è l'unico proprietario dello
snapshot esportato e rilascia i buffer di ogni layer appena questi sono
scritti, così la RAM scende durante la scrittura invece di trattenere l'intero
checkpoint fino alla fine.

### Ring raw-KV

L'attenzione a finestra scorrevole NSA legge solo le righe raw recenti.
`DS4_RAW_RING=1` mantiene la raw KV in un `MTLBuffer` di `nSWA` righe in
memoria condivisa/unificata, riducendo la RAM della raw KV. Non è una cache su
disco: il Disk KV salva e ripristina separatamente i checkpoint di prefissi
completati. Il ring inoltre non rimuove lo stato KV compresso.

Quando la finestra cronologica richiesta si avvolge attorno al ring fisico, un
singolo dispatch GPU 2D riordina le sue righe convertendo F32→F16 per
FlashAttention. La profondità split-K dell'attenzione si basa sulle righe raw e
compresse insieme ed è scelta esattamente come
`min(32, max(1, ceil(totalRows/32)))`. Non è arrotondata a una potenza di due:
128 righe usano 4 workgroup, mentre 129 ne usano 5. Disabilita
`DS4_ADAPTIVE_SPLITK` per confrontare con la profondità fissa storica di 32.

### Cache degli esperti

La slot-cache degli esperti è per layer. Ogni slot contiene un esperto. Su
Flash a 2 bit, uno slot costa circa 6.9 MB per layer. I budget di slot vanno
regolati con la tab Tuning: guarda l'hit-rate e la concentrazione per layer. Un
hit-rate basso significa che la cache non ripaga la sua memoria wired.

`DS4_EXPERT_LOOKAHEAD` (stepper in Settings, default 0) esegue il prefill degli
slot del layer successivo mentre il layer corrente calcola: esatto per i layer
instradati per hash (sempre attivo — la selezione per id di token è nota in
anticipo), top-N dal prior di utilizzo quando è maggiore di 0. L'I/O
speculativo gira solo nella finestra di inattività dell'SSD e cede il passo al
gather reale, quindi una previsione sbagliata spreca solo banda inattiva.

`DS4_PREAD_SPLIT` (default 1) divide ogni slab di esperto in N intervalli
disgiunti letti in concorrenza durante il riempimento diretto (`F_NOCACHE`)
della slot-cache, alzando la profondità di coda NVMe a parità di byte. Il
percorso storico a singola `pread` è il default; sono accettati valori fino
a 8.

### Profiling route/attenzione

`DS4_PROFILE_ROUTE=1` scompone route/attenzione del decode in sottofasi come
compressore, proiezione Q, proiezione KV, attenzione e output. Aggiunge
sincronizzazione extra dei command buffer, quindi usalo per capire rapporti e
colli di bottiglia, non per misurare la velocità assoluta.

## 9. Pannelli avanzati

### Server (`ServerView` + `LocalServer`)

Il server è nativo e in-process. Espone il singolo motore condiviso caricato in
Settings: nessun sottoprocesso, nessuna seconda copia del modello e nessuna
allocazione separata di Q4 residente o `mlock`. `InferenceService` è un actor,
quindi chat, richieste server e chiamate di benchmark locale sono serializzate
in sicurezza.

Espone:

- `/v1/models`
- `/v1/chat/completions`
- `/v1/responses`
- `/v1/completions`
- `/v1/messages`

Carica il modello in Settings prima di avviare il server. Fermare il server
scollega soltanto il socket in ascolto; il motore di chat condiviso resta vivo.

### Distribuito (`DistributedView` + `DS4Engine/Distributed/*`)

La modalità distribuita divide i layer tra i worker, e il COORDINATORE
definisce il compito di ogni worker: la tab Worker avvia solo un listener
inattivo; alla connessione il coordinatore partiziona i layer sull'elenco dei
peer (in ordine, l'ultima fetta possiede la testa di output), costruisce e
invia il manifest dei file, trasferisce i GGUF/sidecar mancanti, poi invia
contesto, budget della cache esperti e fetta nell'assegnazione. I file ricevuti
vengono committati nello store gestito del worker. Un file già presente accanto
al model hint locale del worker può essere riusato, ma solo dopo che la sua
dimensione e il suo SHA-256 corrispondono al manifest del coordinatore; un nome
file uguale o un percorso del coordinatore accessibile da soli non bastano. Il
worker poi carica il suo motore e risponde ready. Settings configura l'elenco
dei peer del coordinatore, la larghezza in bit delle attivazioni, la dimensione
del chunk di prefill e l'inoltro facoltativo worker-to-worker. Chat rende la
conversazione distribuita quando la modalità dell'app è **Distributed**.

Le tool call distribuite vengono eseguite sul Mac coordinatore, quindi i tool
di progetto fanno riferimento al progetto attivo del coordinatore.

Robustezza (protocollo v2): il coordinatore valida la versione di protocollo di
ogni worker alla connessione; ogni turno di chat o run di benchmark porta un id
di `session` che i worker riecheggiano nei loro risultati, così una risposta
lasciata in un buffer di socket da un turno fermato viene scartata invece di
corrompere il successivo. I worker validano ogni frame WORK (dimensione del
payload, limiti dei layer, posizione) prima di eseguirlo e servono un turno
alla volta — un coordinatore concorrente riceve un errore esplicito. Stop si
propaga al task di generazione del cluster e ha effetto al prossimo confine di
chunk. I frame sono TCP in chiaro senza autenticazione: usa la modalità
distribuita solo su reti fidate.

Distribuzione dei file (protocollo v5): i worker non hanno bisogno di file in
anticipo. Dopo l'handshake di versione il coordinatore offre un manifest —
nome, dimensione e SHA-256 del GGUF e (quando abilitati) del sidecar expert
bundle e della cache di riquantizzazione Q4 (i file derivati viaggiano invece
di essere ricostruiti su ogni worker). Ogni worker controlla prima il proprio
store gestito (`Application Support/DwarfStar/dist-models`), poi i candidati
locali con lo stesso nome; entrambi i percorsi di riuso richiedono che
dimensione e hash del manifest corrispondano. Richiede e memorizza solo le voci
mancanti. Il trasferimento usa chunk da 4 MB e accumula l'hash inline, così le
connessioni successive possono saltare il contenuto già verificato. La
decisione on/off del sidecar viaggia nell'ASSIGN, come ogni altra impostazione.

Continuità della KV (protocollo v4): i turni non rifanno più il prefill
dell'intera conversazione ogni volta. Il coordinatore riusa il prefisso in
memoria committato dall'ultimo turno pulito quando la conversazione
ri-renderizzata lo estende esattamente; a freddo (o quando l'impostazione
disk-KV del coordinatore è attiva) negozia un ripristino su tutti gli shard —
ogni worker mantiene checkpoint su disco con chiave sulla fetta, salvati dopo i
turni puliti e ripristinati solo quando OGNI shard detiene lo stesso prefisso;
qualsiasi discrepanza ripiega su un prefill a freddo. L'ASSIGN porta anche
l'imatrix di utilizzo, così ogni worker pre-riscalda la sua slot-cache di
esperti (e persiste tra le sessioni il proprio profilo raffinato per fetta).

Instradamento per id di token (protocollo v7): i frame WORK portano gli id dei
token del chunk. I primi tre layer instradano gli esperti per id di token
(`ffn_gate_tid2eid`), quindi uno shard che li copre non può instradare dal solo
stato HC.

Trasferimenti ripristinabili (protocollo v8): l'offerta dei file porta un
elenco di checkpoint a hash concatenato per file — uno SHA-256 ogni 256 MB,
ciascuno ripiegato sul precedente, così il checkpoint `k` si impegna
sull'intero prefisso (un GGUF da 70 GB aggiunge ~9 KB all'offerta). Un worker
mantiene il suo file `.part` attraverso disconnessioni E sessioni, lo valida
blocco per blocco contro la catena, lo tronca all'ultimo checkpoint valido e
risponde con un offset di ripresa per file; il coordinatore riprende lo
streaming da lì. Gli errori di trasporto durante il setup di un peer vengono
ritentati fino a 3 volte (gli errori semantici — versione non corrispondente,
fetta errata, errori riportati dal worker — no), e ogni tentativo rispedisce al
massimo 256 MB.

Manopole di prestazioni del coordinatore (protocollo v9): l'ASSIGN porta
l'ambiente di prestazioni `DS4_*` misurato del coordinatore, ristretto a una
whitelist su entrambi i lati (`Dist.perfKnobKeys` in
`Sources/DS4Engine/Distributed/Protocol/Core/Dist.swift`), così il wire non può
mai impostare variabili d'ambiente arbitrarie su un worker. Questa whitelist è
un confine di sicurezza dell'ambiente, **non** una garanzia di parità numerica:
le manopole ammesse di fusione, batching e prefill-MM possono cambiare l'ordine
di riduzione/accumulazione e quindi gli ultimi bit floating-point,
potenzialmente cambiando una continuazione campionata. La decisione
deliberatamente lossy `DS4_DENSE_Q4` viaggia separatamente come campo tipizzato
con la sua cache. Prima della v9 un worker con i default di fabbrica girava
senza dense streaming/`mlock`/pread e misurava 0.37 tok/s dove lo stesso
hardware faceva 2.7 in locale — trasportare la configurazione misurata allinea
il lavoro, ma non promette output bit-identico tra hardware o percorsi di
esecuzione diversi.

Parallelismo di esperti (protocollo v11): oltre alla pipeline orizzontale di
layer, il coordinatore può tenere localmente il backbone denso completo e
assegnare a ogni worker una maschera di proprietà con prefisso di lunghezza
sugli esperti instradati di ogni layer: 256 bit per Flash o 384 per Pro. Per
ogni layer instradato invia `expertWork` ai proprietari selezionati dal router
e somma le loro risposte `expertSum`. Il percorso worker è implementato da
`ExpertShard`; la chat verticale e un benchmark dedicato sono esposti dalla
feature Distributed. Le fette orizzontali validano analogamente 43 layer Flash
o 61 layer Pro. Questa topologia richiede un RTT cablato sotto circa 1 ms
perché introduce circa un round-trip per layer instradato. Il GGUF Pro Q2
completo è accettato; il Pro Q4 diviso resta solo scaricabile finché non esiste
un loader multi-shard. Vedi
[`INFERENZA-DISTRIBUITA.md`](INFERENZA-DISTRIBUITA.it.md) e
[`EXPERT_PARALLELISM.md`](EXPERT_PARALLELISM.it.md).

Il setup dei worker gira IN PARALLELO: il trasferimento dei file e il
caricamento del motore di ogni peer procedono insieme, così la route si attiva
in max(setup dei worker) invece che nella somma. Con i file già distribuiti
(dalla seconda connessione in poi), N worker diventano pronti nel tempo di uno;
a freddo i trasferimenti condividono la banda del coordinatore, ma il
caricamento di un worker si sovrappone comunque ai trasferimenti degli altri.
L'ordine della route resta l'ordine dell'elenco dei peer.

La cache di riquantizzazione Q4 completa trasferita dal coordinatore serve
anche direttamente le fette dei worker: i record della cache sono abbinati per
chiave (layer, campo), così uno shard carica il `.q4dense` completo senza
riquantizzare la sua fetta da zero (lo snapshot M1 Pro citato ha misurato circa
mezzo secondo; non è una garanzia hardware). Le scritture di fetta parziale
vanno in un file con suffisso `<cache>.L<lo>-<hi>` e non sovrascrivono mai la
cache completa.

### Benchmark

Il pannello benchmark offre due misure distinte. **Speed** misura il throughput
di prefill e generazione a dimensioni di contesto crescenti. **Correctness**
misura l'accuratezza next-token teacher-forced top-1, top-2 e top-3 su testo
fornito dall'utente su più pezzi di corpus con seed e traccia tutte e tre le
accuratezze per ogni pezzo. I pezzi hanno posizioni distinte del primo target
ma possono sovrapporsi. La loro lunghezza di contesto è campionata
uniformemente nell'intervallo min/max effettivo, e ognuno valuta fino al limite
per pezzo configurato. Il planner preferisce pezzi completi e usa una coda di
corpus più corta solo quando il conteggio richiesto lo impone. I tre candidati
sono token del vocabolario, non esperti MoE, e le metriche annidate soddisfano
sempre `top-1 <= top-2 <= top-3`. Correctness fa sempre avanzare il decoder con
il token di riferimento, non con la sua predizione, così una previsione errata
non cambia il contesto delle osservazioni successive.

Con il prefisso minimo di un token, una tokenizzazione che produce `N` token fa
esattamente `N - 1` predizioni. Più in generale, un prefisso non valutato di
`C` token lascia `N - C` target eleggibili prima che vengano applicati i limiti
selezionati di valutazione e contesto. Questo contratto legacy a pezzo singolo
resta disponibile tramite l'overload a contesto fisso. Il riepilogo multi-pezzo
riporta il piano effettivo, i token valutati, i conteggi di token corretti
top-k, le accuratezze, il troncamento e il throughput. L'accuratezza globale
divide i token corretti totali per i token valutati totali, così i pezzi corti
sono pesati correttamente anziché mediati come percentuali uguali. Il seed
rende il campionamento riproducibile ma non influisce sui logit del modello.
Questa è una metrica deterministica di continuazione, non un punteggio generale
di qualità semantica, perché diversi token successivi possono essere
linguisticamente validi.

In modalità Local il pannello riusa il motore condiviso caricato quando la chat
è inattiva; la run muta la KV e viene rifiutata mentre una generazione è
attiva. In modalità Distributed il benchmark di velocità riusa il coordinatore
connesso, quindi non deve sovrapporsi a una generazione di chat distribuita.

La serie di generazione nel grafico e nel report è il p99 a regime delle
velocità per token, non la media: la media è trascinata giù dai primi token
freddi, mentre il p99 è la velocità di crociera su cui la run si assesta (il
log stampa entrambi). Il grafico disegna una gridline fine da 0.1 t/s perché le
differenze A/B che contano stanno nell'intervallo 0.05-0.3 t/s.

Questo pannello misura a frontiere di contesto crescenti; la tab Settings ha i
propri pulsanti di benchmark e auto-tune che misurano e APPLICANO le manopole
di configurazione (Sezione 7). I valori attualmente applicati appaiono in
Settings nelle righe "Attivi (prefill)" e "Attivi (load)".

### Diagnostica

Diagnostics apre il GGUF solo per i metadati del tokenizer. Può fare il dump
degli id dei token, mostrare il chat template incorporato del modello e
riportare la presenza di tensori MTP; né il template Jinja né un componente MTP
vengono eseguiti da questo percorso diagnostico. Sostituisce il vecchio flusso
`ds4 --dump-tokens` basato su sottoprocesso.

### Download dei modelli

Il foglio di download usa il `ModelDownloader` nativo Swift: download HTTP
Range ripristinabili da Hugging Face, file di ripresa `.part` e verifica
SHA-256 fissata dal catalogo per i nuovi trasferimenti. I download della GUI
vanno nella directory scrivibile
`~/Library/Application Support/DwarfStar/models/`. I file di catalogo esatti
già presenti come file regolari non vuoti in directory di modelli note vengono
riusati senza richieste di rete; le voci con una dimensione esatta fissata
devono anche corrispondere a quel conteggio di byte. I file `.part` interrotti
vengono ripresi.

L'autenticazione si configura in **Settings → Hugging Face**: incolla un token
di sola lettura da `huggingface.co/settings/tokens` e premi Save. Il token è
salvato nel **Keychain** di macOS (mai in UserDefaults), mostrato in seguito
solo in forma oscurata e inviato dal downloader come `Authorization: Bearer` —
necessario per i repository gated/privati e per evitare i limiti di frequenza
anonimi. Remove lo elimina dal Keychain. Quando nessun token è salvato, il
downloader ripiega sulla variabile d'ambiente `HF_TOKEN`, poi su
`~/.cache/huggingface/token`; il foglio di download mostra quale sorgente, se
presente, è attiva.

Il catalogo espone tre voci Flash e la voce Pro Q2 a file singolo come
scaricabili, selezionabili ed eseguibili in locale. Il pacchetto Pro Q4 a due
shard è visibile/scaricabile ma esplicitamente `downloadOnly`; nessuno dei due
shard diventa un modello locale indipendente. Le voci GLM 5.2 IQ2_XXS, Q2_K e
Q4_K da `antirez/glm-5.2-gguf` sono scaricabili con revisione, conteggio di
byte e SHA-256 fissati, e sono selezionabili ed eseguibili col backend
streaming GLM. La
distribuzione del Pro resta in verifica. MTP è un accessorio fuori dal catalogo
GUI del modello principale, e nessun percorso di caricamento attuale lo
consuma. Il **Browse** manuale resta disponibile, ma valida il GGUF con il
selettore di runtime prima di cambiare il modello attivo.

## 10. Build, esecuzione e packaging

```sh
make
make xcodeproj
swift run DS4Demo
swift run DwarfStar
make embed-kernels
make app
```

`make embed-kernels` deve essere eseguito dopo aver modificato i file in
`metal/`. Rigenera
`Sources/DS4Metal/Runtime/Generated/KernelSources.swift`, che è ciò che l'app e
la CLI compilano nel binario.

`make app` costruisce `build/DwarfStar.app` e lo firma ad-hoc. Per la
distribuzione, firma con un Developer ID ed esegui la notarizzazione.

### Risoluzione dei percorsi

`AppEnvironment` risolve i percorsi di sviluppo e del bundle:

- in sviluppo, usa la radice di progetto configurata e la cartella GGUF;
- in una `.app`, risolve le risorse attraverso il bundle;
- i file di modello e di progetto selezionati dall'utente sono memorizzati
  tramite bookmark security-scoped.

## 11. Risoluzione dei problemi

| Sintomo | Causa probabile | Cosa provare |
|---|---|---|
| Il modello non si apre nell'app pacchettizzata | Manca l'accesso sandbox | Scegli il GGUF con **Browse** invece di digitare un percorso. |
| Il caricamento è rifiutato con "unsupported DeepSeek4 shape" | I metadati di shape non corrispondono né al profilo Flash completo né al Pro | Fai l'audit del file con `DS4_TYPES_ONLY=1`; un Pro Q2 a file singolo valido è supportato in locale, quindi questo errore indica metadati mancanti/non corrispondenti o un profilo diverso. |
| Uno shard Pro Q4 scaricato non può essere selezionato | Pro Q4 è un pacchetto a due shard | Usa il modello Pro Q2 a file singolo per l'esecuzione locale; il caricamento del Q4 diviso non è implementato. |
| Il primo caricamento ricostruisce sidecar che esistono già accanto al GGUF | La sandbox può leggere solo il file scelto | Usa **Grant Model Folder Access…** in Settings così i `.q4dense`/`.expbundle` accanto al modello vengono riusati. |
| L'output è privo di senso | Quantizzazione non corrispondente o GGUF sbagliato | Esegui `DS4_TYPES_ONLY=1 swift run DS4Demo <gguf>` e confronta i dtype attesi. |
| Decode molto lento su 16 GB | Dominano lo streaming di esperti da SSD o le riletture dense | Usa i default veloci della GUI: pread esperti, dense streaming, `mlock`, cache di attenzione Q4, expert bundle, contesto moderato. |
| Il primo caricamento richiede molto tempo | È in costruzione l'expert bundle o la cache Q4 densa | Osserva il log del motore. I caricamenti successivi riusano il sidecar/la cache. |
| Il caricamento sembra bloccato su "Riquantizzazione Q4 (solo il primo avvio)" con la CPU al massimo | La conversione una tantum Q8→Q4_K richiede minuti — o ORE in una build Debug non ottimizzata (il default di Run in Xcode, `swift build` semplice) | Osserva la percentuale accanto all'etichetta della fase (avanza a passi dello 0.1%) e il log del motore: una riga `DS4 q4cache: ATTENZIONE: build di DEBUG` significa che dovresti ricompilare in Release (`make app` o la configurazione Release di Xcode). I checkpoint parziali vengono scritti ogni 16 tensori, quindi anche un'uscita forzata riprende da dove si era fermata; se non appare mai un `.q4dense`, cerca errori di scrittura `DS4 q4cache:` (spazio su disco, cartella della cache). |
| L'expert bundle viene saltato | Spazio su disco scrivibile insufficiente o la sandbox non può scrivere accanto al modello | Usa la directory del bundle di Settings in Application Support o libera spazio su disco. |
| Il denso residente peggiora le cose | Pressione di memoria wired | Preferisci il dense streaming sui sistemi da 16 GB; il denso residente è automatico nella GUI ed è utile soprattutto su sistemi ricchi di RAM o nei test A/B da CLI. |
| La cache degli esperti non aiuta | Instradamento troppo uniforme o cache troppo piccola | Controlla hit-rate e concentrazione per layer in Tuning; confronta l'allocazione uniforme con quella guidata dall'uso. |
| La chat distribuita non si connette | Route incompleta o worker non avviati | Avvia prima i worker e assicurati che le fette coprano ogni layer in modo contiguo. |
| La connessione distribuita fallisce con una versione non corrispondente | Coordinatore e worker eseguono build diverse | Aggiorna ogni Mac alla stessa build di DwarfStar; la versione di protocollo deve corrispondere esattamente. |
| Trasferimento file distribuito interrotto | Intoppo di rete a metà trasferimento | Niente da fare: il worker conserva il suo `.part`, il setup ritenta fino a 3 volte e ogni tentativo riprende dall'ultimo checkpoint da 256 MB. |
| L'auto-tune rifiuta la singola baseline calda o un record candidato | Pressione di memoria, gate di swap/stabilità o candidato troppo grande | Libera RAM (chiudi altre app) e riesegui l'intero tuner; all'interno di una run una configurazione non viene deliberatamente mai misurata due volte. |
| Il server funziona ma la chat rallenta | Risorse GPU/SSD condivise | Evita la generazione simultanea di chat e server sullo stesso Mac. |
| La build non può scrivere la cache Swift/clang nella sandbox | Cache della toolchain fuori dalle radici scrivibili | Compila fuori dalla sandbox gestita o configura percorsi di cache scrivibili. |

## 12. Glossario

| Termine | Significato |
|---|---|
| GGUF | Formato contenitore dei modelli usato dal motore DS4. |
| MoE | Mixture-of-Experts; esperti FFN instradati selezionati per token/layer. |
| Esperto instradato | Tensore di esperto selezionato dal router per il token corrente. |
| Pesi non instradati | Pesi densi sempre necessari, come le proiezioni di attenzione e l'FFN condiviso. |
| KV cache | Stato chiave/valore dell'attenzione accumulato sul contesto. |
| Ring raw-KV | Buffer raw KV a finestra scorrevole di dimensione costante. |
| NSA | Native Sparse Attention / percorso del compressore usato da DeepSeek-V4. |
| HC | Stato nascosto Hyper-Connection trasportato attraverso layer/worker. |
| DSML | Markup delle tool call DeepSeek aperto dal token `｜DSML｜`. |
| Imatrix di utilizzo | Tabella per layer della frequenza di instradamento degli esperti usata per il tuning della cache. |
| Layer hash | Uno dei primi 3 layer, i cui esperti sono selezionati per id di token tramite la tabella `tid2eid` invece che dal router. |
| Slot-cache | Cache LRU residente su GPU degli esperti caldi. |
| Expert bundle | Sidecar che salva gli slab di ogni esperto in modo contiguo per letture di miss più veloci. |
| Dense streaming | Percorso di staging per layer dei pesi densi che sovrappone le letture SSD al calcolo. |
| Cache Q4 densa | Copie Q4_K riquantizzate e messe in cache delle grandi proiezioni di attenzione. |
| `mlock` | Richiesta best-effort di mantenere i buffer caldi residenti e fuori dal compressore di memoria. |
| Prefill | Elaborazione dei token del prompt prima della generazione. |
| Decode | Generazione token per token dopo il prefill. |
