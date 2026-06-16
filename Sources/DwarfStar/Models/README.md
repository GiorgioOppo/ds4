# DwarfStar/Models

Selezione, scansione e download dei modelli GGUF.

- **`ModelPicker.swift`** — selezione sandbox-friendly via `NSOpenPanel` + bookmark security-scoped (riapre lo stesso file tra i lanci).
- **`ModelCatalog.swift`** — scansione delle cartelle per i `.gguf` presenti.
- **`DownloadView.swift` / `DownloadRunner.swift`** — UI e driver del download nativo (`DS4Engine.ModelDownloader`), con progresso e fasi (verifica integrità).
