[English](README.md) | **Italiano**

# Feature di DwarfStar

Ogni directory figlia implementa un'area rivolta all'utente dell'applicazione
macOS. Il codice delle feature può dipendere da `App/`, `Shared/Support`,
`DS4Core` e `DS4Engine`, ma una feature non deve possedere lo stato di
un'altra feature.

## Aree

- `Chat/`: conversazioni, stato di generazione, strumenti, allegati e
  sessioni.
- `ModelManagement/`: scoperta di GGUF basata sul catalogo dell'engine,
  selezione manuale convalidata, download riprendibili e avanzamento dei
  pacchetti.
- `Project/`: bookmark della libreria dei progetti e selezione del progetto.
- `Settings/`: configurazione globale del runtime e di MCP.
- `Tuning/`: telemetria della cache degli esperti ed editing degli agenti.
- `Server/`: la façade HTTP locale compatibile OpenAI/Anthropic.
- `Distributed/`: controlli per coordinator e worker.
- `Benchmark/`: misurazioni di throughput locali e distribuite.
- `Diagnostics/`: ispezione di tokenizer, template e log dell'engine.

## Regole di modifica

- Metti lo stato di presentazione mutabile in un controller o in un view
  model, non in una view.
- Mantieni la logica riutilizzabile di inferenza, persistenza, protocollo e
  strumenti nei moduli dell'engine. Le feature adattano quelle API per
  SwiftUI.
- Leggi le impostazioni condivise di modello e runtime tramite `AppSettings`;
  non creare una seconda istanza del modello per un singolo pannello.
- Aggiungi un README a livello di feature e aggiorna questo indice quando
  introduci un'area.
