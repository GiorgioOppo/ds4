[English](README.md) | **Italiano**

# DwarfStar

L'app macOS SwiftUI per Apple Silicon. È guidata da `DS4Engine`; una sidebar
seleziona i pannelli principali. Il percorso del modello e la lunghezza del
contesto si configurano una sola volta in **Settings** tramite `AppSettings`,
e vengono poi ereditati da ogni controller. Ogni impostazione persistita
(chiave UserDefaults, valore predefinito, variabile d'ambiente dell'engine) è
documentata nel
[Riferimento di Configurazione](../../README.it.md#riferimento-di-configurazione) alla
radice.

L'app è organizzata per **feature**, con una cartella per tab o area sotto
`Features/`:

- **`App/`**: entry point, impostazioni condivise, root view e helper
  d'ambiente.
- **`Features/Chat/`**: modelli, persistenza, view model e viste per la chat
  in streaming con Markdown, reasoning, tool call live, allegati e il view
  model `ChatStore`.
- **`Features/ModelManagement/`**: discovery dei GGUF basata su catalogo,
  selezione manuale validata, download riprendibili e UI di progresso. Flash e
  il Pro Q2 a file singolo sono eseguibili; il pacchetto Pro Q4 a due file e
  le tre varianti GLM 5.2 sono solo download.
- **`Features/Project/`**: libreria dei progetti, con cartelle con bookmark di
  sandbox indicizzate per i tool degli agenti.
- **`Features/Tuning/`**: slot della expert cache, policy del pool mixed-quant,
  hit-rate, concentrazione del routing ed editor degli agenti.
- **`Features/Server/`**: adapter API, networking, concorrenza e UI per il
  server HTTP nativo in-process compatibile con endpoint in stile OpenAI e
  Anthropic, che espone l'engine condiviso caricato dalle Settings.
- **`Features/Distributed/`**: UI per l'inferenza distribuita
  worker/coordinator.
- **`Features/Benchmark/`**: benchmark di prefill/generazione locali o
  distribuiti su dimensioni di contesto crescenti.
- **`Features/Diagnostics/`**: dump di token e di chat template.
- **`Features/Settings/`**: impostazioni globali di modello, contesto,
  modalità di esecuzione e memoria/I/O come expert cache mixed-quant
  consapevole dei layer, expert bundle, dense streaming, cache densa Q4, disk
  KV e ring raw-KV.
- **`Shared/Support/`**: utility trasversali come i log dell'engine e gli
  stream di processo.
- **`Assets.xcassets/`**: asset dell'icona dell'app.

## Flusso di dipendenze e stato

`DwarfStarApp` crea `AppSettings`, `ChatStore` e lo stato MCP. `RootView`
passa questi oggetti condivisi ai controller e alle viste delle feature. Chat
possiede l'unico `InferenceService` locale; Server e Benchmark locale prendono
in prestito quell'istanza e serializzano il lavoro invece di caricare pesi
duplicati.

Il codice applicativo può adattare le API di `DS4Engine` e `DS4Core`. Il
comportamento riutilizzabile di modello, inferenza, protocollo, storage e tool
appartiene a quei moduli, non al target SwiftUI.

I GGUF scaricati vivono in `~/Library/Application Support/DwarfStar/models/`.
L'app renderizza il catalogo da `DS4Engine`; non duplica nomi di file remoti,
valori SHA-256 o decisioni di supporto runtime.

## Mappa della documentazione

- [`App/README.md`](App/README.it.md): avvio e impostazioni condivise.
- [`Features/README.md`](Features/README.it.md): indice delle feature e confini.
- [`Features/Chat/FLOW.md`](Features/Chat/FLOW.it.md): flusso di messaggi, tool,
  persistenza ed engine condiviso.
- [`Features/ModelManagement/README.md`](Features/ModelManagement/README.it.md):
  catalogo GUI, riuso, ripresa, selezione e confine di runtime.
- [`Features/Server/HTTP-API.md`](Features/Server/HTTP-API.it.md): endpoint e
  ciclo di vita delle richieste.
- [`Shared/README.md`](Shared/README.it.md): regole di supporto trasversali alle
  feature.

Ogni directory sorgente ha un README locale. Aggiorna il README più vicino
quando i file si spostano o la proprietà cambia; non documentare i default
d'ambiente in più punti quando il Riferimento di Configurazione alla radice è
autoritativo.
