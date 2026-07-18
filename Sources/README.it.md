# Sources

L'albero dei sorgenti è organizzato prima per **target Swift**, poi per feature
o responsabilità tecnica. I confini fra i target sono imposti da SwiftPM; le
cartelle all'interno di un target sono organizzative e non creano moduli Swift
aggiuntivi.

Il grafo esatto dei target SwiftPM è riportato sotto. Una freccia `A -> B`
significa che **B importa e dipende direttamente da A**; la disponibilità
transitiva non è una licenza per saltare il confine dichiarato.

```text
DS4Core  -> DS4Metal
DS4Core  -> DS4Engine       DS4Metal -> DS4Engine
DS4Core  -> DwarfStar       DS4Engine -> DwarfStar
DS4Core  -> DS4Demo         DS4Metal -> DS4Demo

DS4Core + DS4Metal + DS4Engine -> DS4CoreTests
```

Di conseguenza, `DS4Core` non ha dipendenze da target del progetto; `DS4Metal`
può costruire solo verso l'alto a partire da `DS4Core`; e `DS4Engine` è il
primo livello autorizzato a unire i dati portabili con il backend Metal e la
politica applicativa. `DwarfStar` importa direttamente `DS4Engine` e `DS4Core`
ma non `DS4Metal`. `DS4Demo` aggira deliberatamente `DS4Engine` e pilota
`DS4Core` + `DS4Metal` per bring-up e diagnostica delle prestazioni.
`DS4CoreTests` è il target di test condiviso e può importare tutte e tre le
librerie.

| Target | Tipo | Dipendenze dirette dai target | Responsabilità |
|---|---|---|---|
| `DS4Core/` | libreria | nessuna | Formati in puro Swift, ispezione dell'architettura, contratti portabili e frontend tokenizer/conversazione di proprietà dei backend. Niente Metal né SwiftUI. |
| `DS4Metal/` | libreria | `DS4Core` | Runtime e kernel Metal condivisi, più decoder di modello concreti e fisicamente separati e stato GPU. |
| `DS4Engine/` | libreria | `DS4Core`, `DS4Metal` | Selezione del backend e orchestrazione applicativa: servizio di inferenza, DTO, tool, persistenza, download e distribuzione. |
| `DwarfStar/` | eseguibile | `DS4Engine`, `DS4Core` | Applicazione SwiftUI nativa e il suo server HTTP rivolto all'app, raggruppati per feature visibile all'utente. |
| `DS4Demo/` | eseguibile | `DS4Core`, `DS4Metal` | Piccola CLI che pilota direttamente il core e il runtime GPU per audit, diagnostica e generazione. |

## Mappa delle directory

```text
DS4Core/
  Conversation/
    Models/             turni di chat, chiamate ai tool e dati di conversazione condivisi
    Backends/
      DeepSeekV4/DSML/  template DeepSeek, markup e parser delle chiamate ai tool
      GLM52/            ruoli GLM, reasoning e protocollo tool XML nativo
      Qwen/             segnaposto documentato, non implementato
  Diagnostics/          report di caricamento/avanzamento senza dipendenze UI
  Formats/
    GGUF/                tipi GGUF, cursore e modello mappato
    KVCheckpoint/        formato su disco dei checkpoint KV
    Quantization/        conversione numerica e helper di quantizzazione su CPU
  Generation/           sampling
  Model/
    Common/             id di architettura, famiglia, descrittore e capacità
    Backends/
      DeepSeekV4/       validazione di forma e metadati Flash/Pro
      GLM52/            validazione rigorosa di forma e metadati GLM 5.2
      Qwen/             segnaposto documentato, non implementato
  Storage/              pianificazione SSD/cache e simulazione del memory-lock
  Tokenization/
    API/                contratto tokenizer minimale e neutro rispetto all'architettura
    Common/             helper a livello di byte
    Backends/
      DeepSeekV4/       tokenizer concreto, token speciali e thinking mode
      GLM52/            tokenizer GPT-2/glm4 e token speciali GLM
      Qwen/             segnaposto documentato, non implementato

DS4Metal/
  Runtime/
    Core/                device Metal, pipeline e primitive tensoriali su GPU
    Generated/           sorgente Metal embedded generato; mai modificare a mano
  Model/Quantization/   descrittori di quantizzazione indipendenti dall'architettura
  Backends/
    Common/             confine di alto livello; non fa mai dispatch dentro un layer
    DeepSeekV4/
      Architecture/     dimensioni Flash e parametri RoPE
      Weights/          pesi dei layer basati su GGUF e provider
      Experts/ MTP/     bundle degli esperti, MetalIO e sidecar MTP opzionale
      Streaming/        streaming dei pesi densi e cache di riquantizzazione
      Decode/           Execution, Generation, Attention, Cache, KV, Prefill,
                        State, Diagnostics e Reference
    GLM52/              schema dei tensori, riferimenti DSA compatti e kernel a stadi;
                        nessun decoder eseguibile per ora
    Qwen/               segnaposto documentato, nessun decoder né kernel finti
  Graph/
    Core/                contesto del grafo
    Operations/          stadi del grafo mirati (attention, MoE, RoPE, output…)
  Kernels/
    Attention/ Compression/ Dense/ MoE/ Tensor/
                        wrapper Swift di dispatch raggruppati per operazione

DS4Engine/
  Runtime/
    Common/             ispezione del modello, descrittore, capacità e selettore
    Backends/
      DeepSeekV4/       registrazione del backend concreto operativo
      GLM52/            definizione del backend riconosciuto, deliberatamente non disponibile
      Qwen/             backend riconosciuto ma deliberatamente non disponibile
  Inference/
    API/                 strutture dati pubbliche di richiesta/risultato/evento
    Service/             attore di inferenza ed estensioni conversazione/generazione
    Benchmark/           operazioni di benchmark
    Diagnostics/         diagnostica tokenizer/template
    Subagents/           esecuzione isolata di sub-agent
    Tuning/              operazioni di tuning a runtime
  Distributed/
    Protocol/            tipi wire raggruppati per framing, handshake, file,
                         KV, lavoro, esperti, codec e serializzazione
    Coordinator/         orchestrazione di connessioni, file, KV, chat e benchmark
    Worker/              stato del worker più i sottodomini Assignments, Files, KV,
                         Lifecycle, Serving e Concurrency
    Transport/           trasporto di rete
    Execution/           esecuzione del modello per nodo
    Files/               distribuzione del modello
  Tools/
    Core/                registry e contratti comuni dei tool
    Builtins/            un'implementazione per ogni tool integrato, raggruppate per area
    Integrations/        client Git/GitHub/web condivisi
    MCP/                 configurazione, protocollo e trasporti MCP
  Agents/                profili degli agent
  ModelManagement/       download dei modelli e operazioni sui bundle di esperti
  Persistence/           stato applicativo e KV persistiti su disco
  Projects/              indicizzazione/cache dei progetti

DwarfStar/
  App/                   entry point dell'applicazione, environment e vista radice
  Features/
    Chat/                 Models, Persistence, ViewModels e Views
    Server/               adapter API, networking, concorrenza, servizi, UI
    Benchmark/            Controllers e Views
    Diagnostics/          Controllers e Views
    Distributed/          Controllers e Views
    ModelManagement/      Models, Services e Views
    Project/ Settings/ Tuning/
                         viste e stato specifici della feature
  Shared/Support/         helper GUI condivisi da più feature
  Assets.xcassets/        asset dell'app (solo build Xcode)

DS4Demo/
  Command/               entry point dell'eseguibile e gestione degli argomenti
  Diagnostics/           logging, benchmark del disco e helper di audit del modello
```

## Regole di collocazione

- Metti i dati riutilizzabili e indipendenti dall'hardware in `DS4Core`; lì non
  importare Metal, Network o SwiftUI. Mantieni comune l'identità
  dell'architettura, ma tieni tokenizer, formato di chat e forma nel backend
  che li possiede.
- Metti stato e operazioni basati su GPU in `DS4Metal`. La politica applicativa
  visibile al modello appartiene a `DS4Engine`, non a un wrapper di kernel. Non
  aggiungere mai condizionali per famiglia di modello nel hot loop per layer;
  seleziona prima un backend concreto.
- Metti i tipi pubblici di richiesta/risultato/evento dell'inferenza in
  `DS4Engine/Inference/API`. Mantieni il comportamento del servizio in
  un'estensione mirata `InferenceService+Area.swift`.
- Metti i dati wire distribuiti sotto `DS4Engine/Distributed/Protocol`,
  raggruppati per aspetto del protocollo. Il trasporto e il comportamento di
  coordinator/worker restano fuori dalla cartella del protocollo.
- Metti modelli solo UI, adapter di persistenza, controller/view model e viste
  dentro il sottoalbero `DwarfStar/Features/<Feature>` che li possiede. Usa
  `DwarfStar/Shared` solo quando più di una feature possiede la dipendenza.
- Tieni i file generati sotto una directory `Generated/` e documentane il
  generatore. `DS4Metal/Runtime/Generated/KernelSources.swift` è generato da
  `metal/*.metal` tramite `make embed-kernels`; modifica invece i file
  `.metal`.
- Preferisci un tipo primario o un'estensione coesa per file. Chiama le
  estensioni `Type+Responsibility.swift` e tieni i DTO separati da I/O o
  esecuzione.

Vedi [`../docs/STRUTTURA-PROGETTO.md`](../docs/STRUTTURA-PROGETTO.md) per le
regole di dipendenza, il flusso di contribuzione e i comandi di build.
Vedi [`../docs/ARCHITETTURE-SUPPORTATE.md`](../docs/ARCHITETTURE-SUPPORTATE.md)
per la matrice di supporto e le checklist per GLM 5.2 e Qwen.

## Regola della documentazione locale

Ogni directory in `Sources` ha un `README.md`. Leggi prima il README del
target, poi il README locale più vicino prima di modificare un tipo. I file
locali documentano proprietà, dipendenze, file principali e regole di
estensione/test; il comportamento trasversale appartiene a
[`../docs/`](../docs/README.md).

Tutti i file Markdown collocati accanto ai sorgenti dei target vengono
individuati ed esclusi dall'helper in `Package.swift`; `project.yml` applica
l'esclusione ricorsiva equivalente per XcodeGen. Aggiungere una guida accanto
al codice quindi non la copia nell'eseguibile né produce warning di risorse non
gestite.
