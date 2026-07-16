# DwarfStar/Features/ModelManagement

Selezione e download nativo dei GGUF supportati dalla GUI.

- **`Views/ModelPicker.swift`** provides sandbox-friendly selection through
  `NSOpenPanel` plus security-scoped bookmarks, so the same file can be reopened
  across launches.
- **`Models/ModelCatalog.swift`** individua i tre DeepSeek V4 Flash e il Pro Q2
  singolo che il runtime corrente può caricare; **Browse** resta la via esplicita per GGUF
  custom e li valida prima di memorizzarli.
- **`Views/DownloadView.swift` / `Services/DownloadRunner.swift`** provide the UI and driver for
  the native downloader (`DS4Engine.ModelDownloader`). La GUI deriva tutte le
  righe da `DeepSeekV4ModelCatalog`: non mantiene un secondo elenco hardcoded.
  Mostra stato Installato/Parziale/Non scaricato, spazio, fase, avanzamento,
  annullamento e ripresa. Il runner passa il token Keychain configurato in
  Settings → Hugging Face senza mostrarne il valore completo.

Catalog records and scans live in `Models/`, UI adapters in `Services/`, and
selection/progress rendering in `Views/`. Remote protocol, integrity, resume,
and credential logic stays in `DS4Engine`; never log or persist a plaintext
token in this feature.

I nuovi file finiscono in `Application Support/DwarfStar/models`, non nelle
Resources read-only del bundle. Prima del download vengono cercati anche nelle
directory di sviluppo e accanto al modello selezionato: un file catalogato già
presente e non vuoto viene riusato senza rete né copia. Un `.part` nella cartella
gestita viene conservato dopo Cancel e ripreso al tentativo successivo.

## Confine di supporto

Le tre quantizzazioni Flash e Pro IQ2 singolo sono scaricabili e selezionabili.
Il package Pro Q4 a due shard è visibile e scaricabile, ma mantiene il badge
download-only e non diventa automaticamente il modello attivo. MTP e Qwen non
compaiono nel catalogo principale.
