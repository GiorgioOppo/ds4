[English](STRUTTURA-PROGETTO.md) | **Italiano**

# Struttura del progetto

Questa guida descrive dove collocare nuovo codice e quali dipendenze sono
ammesse. La struttura separa i dati puri, l'esecuzione GPU, l'orchestrazione,
le funzionalità dell'app e la CLI, in modo che ogni modifica abbia un
proprietario chiaro.

## Moduli e dipendenze

```text
DS4Metal  = DS4Core
DS4Engine = DS4Core + DS4Metal
DwarfStar = DS4Core + DS4Engine
DS4Demo   = DS4Core + DS4Metal
```

In particolare:

| Target | Dipendenze | Contenuto |
|---|---|---|
| `DS4Core` | nessuna interna | Formati e contratti portabili, rilevamento architettura, sampling e componenti Core dei backend. |
| `DS4Metal` | `DS4Core` | Runtime e kernel Metal comuni; decoder, pesi e stato GPU separati per backend. |
| `DS4Engine` | `DS4Core`, `DS4Metal` | Selezione del backend, servizio di inferenza, strumenti, persistenza, download e distribuzione. |
| `DwarfStar` | `DS4Engine`, `DS4Core` | Interfaccia SwiftUI, stato delle feature e server HTTP esposto dall'app. |
| `DS4Demo` | `DS4Core`, `DS4Metal` | CLI diagnostica che usa direttamente il motore, senza lo strato applicativo. |

Le sottocartelle non sono moduli Swift. Servono a rendere visibile la
responsabilità del codice; i confini reali sono i target dichiarati in
`Package.swift`.

## Mappa delle responsabilità

### DS4Core: contratti portabili e frontend dei backend

- `Model/Common`: identificatore `general.architecture`, famiglia, descrittore,
  capacità e rilevamento senza dipendenze Metal.
- `Model/Backends/DeepSeekV4`: forma, profili Flash/Pro, default e validazione
  dei metadati `deepseek4.*`.
- `Model/Backends/Qwen`: punto di estensione documentato; nessuna forma Qwen è
  ancora implementata.
- `Conversation/Models`: turni, specifiche e chiamate tool condivise.
- `Conversation/Backends/DeepSeekV4`: renderer, markup DSML e parsing delle
  chiamate specifici del template DeepSeek.
- `Tokenization/Common`: primitive byte-level riutilizzabili.
- `Tokenization/Backends/DeepSeekV4`: tokenizer, token speciali e modalità
  thinking DeepSeek; il percorso Qwen resta separato e non operativo.
- `Formats/GGUF`: tipi GGUF, cursore binario e modello mappato in memoria.
- `Formats/KVCheckpoint`: involucro persistente comune; i payload restano dei
  singoli backend.
- `Formats/Quantization`: conversioni numeriche e quantizzazione CPU.
- `Generation`: sampling e politiche di scelta del token.
- `Storage` e `Diagnostics`: pianificazione cache/SSD e avanzamento del load.

Qui devono vivere strutture riutilizzabili senza Metal, rete o UI. Un tipo non
diventa comune solo perché due modelli hanno un concetto con lo stesso nome:
layout, token speciali e semantica devono essere realmente compatibili.

### DS4Metal: runtime comune e backend GPU concreti

- `Runtime/Core`: device, command queue, pipeline e `GPUTensor`.
- `Runtime/Generated`: sorgenti Metal incorporati e generati.
- `Kernels/<Area>`: wrapper Swift di operazioni riutilizzabili, raggruppati per
  attenzione, compressione, dense, MoE e tensori.
- `Graph/Core` e `Graph/Operations`: infrastruttura del grafo oggi condivisa
  con il backend DeepSeek; non ospita la selezione dell'architettura.
- `Model/Quantization`: descrittori di quantizzazione GPU comuni.
- `Backends/Common`: confine di esecuzione ad alto livello e regole che
  impediscono dispatch dinamico nel ciclo per layer.
- `Backends/DeepSeekV4/Architecture`: dimensioni Flash e parametri RoPE.
- `Backends/DeepSeekV4/Weights`, `Streaming`, `Experts`, `MTP`: mapping GGUF,
  pesi densi ed esperti, sidecar e streaming specifici DeepSeek.
- `Backends/DeepSeekV4/Decode`: esecuzione, generazione, attenzione, cache,
  KV, prefill, diagnostica, riferimento e stato del decoder concreto.
- `Backends/GLM52`: schema tensoriale, riferimenti DSA/IndexShare, il motore
  streaming (`GLM52ResidentModel` + `GLM52ChainedDecode`) e i kernel Metal
  per famiglia; eseguibile end-to-end (chat, demo, server, benchmark).
- `Backends/Laguna`: schema tensoriale delle ricette S 2.1 pubblicate e gate
  di runtime (spento); il decoder non è ancora portato.
- `Backends/Qwen`: placeholder documentato; nessun kernel o decoder fittizio.

Una struttura dati che contiene buffer o risorse Metal appartiene a questo
target. La scelta dell'architettura avviene prima di entrare nel percorso caldo:
non aggiungere `if qwen` a `StreamingDecoder` o ai suoi layer. Una scelta
applicativa su come usare il modello appartiene invece a `DS4Engine`.

### DS4Engine: API e funzionalità applicative

- `Runtime/Common`: ispezione del modello, selezione del backend, descrittore e
  capacità consumate dai client.
- `Runtime/Backends/DeepSeekV4`: costruzione e impostazioni del backend
  operativo senza cambiare il percorso caldo del decoder.
- `Runtime/Backends/Laguna`: registrazione delle capability dietro il gate di
  runtime Laguna; la selezione rifiuta la famiglia finché il decoder non
  arriva.
- `Runtime/Backends/Qwen`: errore esplicito e punto di estensione, non una
  implementazione simulata.
- `Inference/API`: DTO pubblici di richieste, eventi, risultati e benchmark.
- `Inference/Service`: actor principale e sue estensioni per conversazione,
  generazione e agenti.
- `Inference/Benchmark`, `Diagnostics`, `Subagents`, `Tuning`: funzionalità
  separate che compongono il servizio.
- `Distributed/Protocol`: dati del protocollo di rete divisi per framing,
  handshake, file, KV, work, esperti, codec e serializzazione.
- `Distributed/Coordinator`: stato centrale ed estensioni separate per
  connessioni, file, KV, chat, expert parallelism e benchmark.
- `Distributed/Worker`: stato del nodo ed estensioni raggruppate in
  `Assignments`, `Files`, `KV`, `Lifecycle`, `Serving` e `Concurrency`.
- `Distributed/Transport`, `Execution`, `Files`: rete, motore per nodo e
  distribuzione dei modelli, tenuti separati dai messaggi wire.
- `Tools/Core`: registro e contratti comuni; `Tools/Builtins`: un tool per file;
  `Tools/Integrations` e `Tools/MCP`: client e protocolli esterni.
- `Agents`, `ModelManagement`, `Persistence`, `Projects`: servizi applicativi
  con responsabilità autonome.

I tipi trasmessi su rete non devono dipendere dal coordinator o dal worker.
Il trasporto non deve definire la semantica dei messaggi. Le estensioni del
servizio seguono il nome `InferenceService+Responsabilita.swift`.
`InferenceService` resta la façade pubblica: ispeziona e seleziona una volta il
backend, poi conserva il decoder concreto. Demo, GUI e diagnostica non devono
duplicare euristiche basate sul nome del file.

### DwarfStar: feature dell'interfaccia

Ogni area visibile vive in `Features/<Feature>`:

- `Chat`: `Models`, `Persistence`, `ViewModels`, `Views`;
- `Server`: `API`, `Networking`, `Concurrency`, `Services`, `Controllers`,
  `Views`;
- `Benchmark`, `Diagnostics`, `Distributed`: controller e viste separati;
- `ModelManagement`: modelli, servizi e viste;
- `Project`, `Settings`, `Tuning`: contenuto specifico della feature.

`App` contiene solo bootstrap, ambiente, impostazioni globali e navigazione
radice. `Shared/Support` è riservato a helper realmente condivisi tra più
feature; non deve diventare una cartella generica.

### DS4Demo: CLI e diagnostica

`Command/main.swift` gestisce argomenti e ciclo di esecuzione. Logging, audit
del modello e benchmark del disco stanno in `Diagnostics`, così la CLI non
duplica logica del motore.

### Test: struttura speculare ai domini

Il target SwiftPM resta unico (`DS4CoreTests`), ma i file sono raggruppati per
modulo e responsabilità:

```text
Tests/DS4CoreTests/
  Core/                 contratti comuni e regressioni dei backend Core
  Metal/                runtime/kernel comuni e regressioni dei backend GPU
  Engine/               selezione backend, inferenza e servizi applicativi
```

Le sottocartelle sono scoperte ricorsivamente sia da SwiftPM sia da XcodeGen.
Un nuovo test va collocato accanto al dominio del codice verificato; il nome del
target non implica che debba riguardare soltanto `DS4Core`.

## Regole per nuovi file

1. Collocare una struttura dati accanto al dominio che la possiede, non accanto
   al primo chiamante. I DTO pubblici vanno in `Inference/API`; i messaggi wire
   in `Distributed/Protocol`; i modelli solo UI nella feature corrispondente.
2. Separare definizione dei dati, serializzazione, I/O ed esecuzione quando
   cambiano per motivi diversi.
3. Preferire un tipo principale o un'estensione coesa per file. Per estendere
   un tipo usare `Tipo+Funzionalita.swift`.
4. Tenere le dipendenze orientate verso il basso: Core non conosce Metal,
   Engine o UI; Metal non conosce Engine o UI; Engine non conosce SwiftUI.
5. Ispezione e descrizione dell'architettura stanno nel livello comune; forma,
   tokenizer, template, tensori, decoder e payload KV stanno nel backend.
6. Un wrapper Metal riutilizzabile va in `Kernels/<Area>`; il sorgente eseguito
   dalla GPU va in `metal/*.metal`. Un grafo specifico va in
   `Backends/<Architettura>` e non nel runtime comune.
7. Una feature GUI nuova riceve una cartella propria, con sottocartelle create
   solo quando esistono responsabilità distinte (modelli, servizi, controller,
   viste o persistenza).
8. Non modificare a mano file generati. Documentare sempre il comando che li
   rigenera.
9. Ogni nuova cartella sorgente, test o operativa deve contenere un
   `README.md` con scopo, dipendenze, file principali e regole di verifica. I
   dettagli trasversali vanno in un documento tematico sotto `docs/`.

I Markdown collocati dentro un target sono esclusi automaticamente dal helper
`markdownFiles` in `Package.swift`; XcodeGen usa l'esclusione ricorsiva
`**/*.md`. Non mantenere liste manuali di README nel manifesto.

## Kernel generati

I file `metal/*.metal` sono la fonte autorevole. Il comando:

```sh
make embed-kernels
```

genera
`Sources/DS4Metal/Runtime/Generated/KernelSources.swift`, incorporato nei
binari SwiftPM e nell'app. Ogni modifica a un kernel deve quindi seguire questo
flusso:

1. modificare il file `.metal`;
2. aggiornare il wrapper in `Sources/DS4Metal/Kernels/<Area>` se cambia la
   firma;
3. eseguire `make embed-kernels`;
4. compilare e lanciare i test.

## Build, test e progetto Xcode

Dalla radice del repository:

```sh
# Build di debug di tutti i target
swift build --disable-sandbox

# Suite di test
swift test --disable-sandbox

# Build release della demo
swift build -c release --product DS4Demo --disable-sandbox

# Rigenera il progetto dopo aggiunte, rimozioni o spostamenti di file
xcodegen generate

# Avvia la CLI o la GUI tramite SwiftPM
swift run DS4Demo
swift run DwarfStar
```

Su macOS, se la toolchain attiva non punta all'installazione completa di
Xcode, anteporre:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build --disable-sandbox
```

Dopo una riorganizzazione verificare sia SwiftPM sia il progetto rigenerato:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild build \
  -project DwarfStar.xcodeproj \
  -scheme DwarfStar \
  -destination 'platform=macOS' \
  -derivedDataPath /tmp/DwarfStarDerivedData \
  CODE_SIGNING_ALLOWED=NO
```

`xcodegen generate` usa `project.yml` come fonte autorevole: non aggiungere i
file manualmente al `.pbxproj`.
