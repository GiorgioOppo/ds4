# DwarfStar

L'app SwiftUI (macOS, Apple Silicon). Pilotata da `DS4Engine`; una sidebar seleziona i pannelli. Modello e contesto si impostano una volta in Impostazioni (`AppSettings`) e sono ereditati da ogni controller.

Organizzata per **feature** (una cartella per tab/area):

- **`App/`** — entry point, settings condivisi, root view, ambiente.
- **`Chat/`** — chat in streaming (markdown, reasoning, tool-call live, allegati), `ChatStore` (view-model).
- **`Models/`** — selezione/scansione/download dei GGUF.
- **`Project/`** — libreria progetti (cartelle indicizzate via bookmark sandbox).
- **`Tuning/`** — slot cache esperti, hit-rate, editor agenti.
- **`Server/`** — server HTTP nativo in-process (OpenAI/Anthropic-compatible).
- **`Distributed/`** — worker/coordinatore lato UI.
- **`Bench/`** — benchmark (prefill+gen vs contesto), locale o distribuito.
- **`Diagnostics/`** — dump token / chat template.
- **`Settings/`** — pannello Impostazioni.
- **`Support/`** — utility (log motore, stream di processo).
- **`Assets.xcassets/`** — icona app.
