# DwarfStar/Features/ModelManagement

Selezione e download nativo dei GGUF supportati dalla GUI.

- **`Views/ModelPicker.swift`** fornisce una selezione compatibile con la
  sandbox tramite `NSOpenPanel` più bookmark security-scoped, così lo stesso
  file può essere riaperto tra un avvio e l'altro.
- **`Models/ModelCatalog.swift`** identifica i tre DeepSeek V4 Flash e il Pro
  Q2 a file singolo che il runtime corrente può caricare; **Browse** resta il
  percorso esplicito per i GGUF personalizzati e li valida prima di
  memorizzarli.
- **`Views/DownloadView.swift` / `Services/DownloadRunner.swift`** forniscono la UI e il driver del
  downloader nativo (`DS4Engine.ModelDownloader`). La GUI deriva tutte le
  righe da `ModelCatalogRegistry`: non mantiene una seconda lista hardcoded.
  Mostra lo stato Installato/Parziale/Non scaricato, lo spazio su disco, la
  fase, l'avanzamento, l'annullamento e la ripresa. Il runner passa il token
  del Keychain configurato in Settings → Hugging Face senza mai mostrarne il
  valore completo.

I record e le scansioni del catalogo vivono in `Models/`, gli adapter della UI
in `Services/` e il rendering di selezione/avanzamento in `Views/`. La logica
di protocollo remoto, integrità, ripresa e credenziali resta in `DS4Engine`;
non loggare né persistere mai un token in chiaro in questa feature.

I nuovi file finiscono in `Application Support/DwarfStar/models`, non nelle
Resources in sola lettura del bundle. Prima di un download vengono cercati
anche nelle directory di sviluppo e accanto al modello selezionato: un file a
catalogo già presente e non vuoto viene riusato senza rete né copia, purché
anche l'eventuale dimensione esatta corrisponda. Un `.part` nella cartella
gestita viene conservato dopo l'annullamento e ripreso al tentativo
successivo.

## Confine del supporto

Le tre quantizzazioni Flash e il Pro IQ2 a file singolo sono scaricabili e
selezionabili. Il pacchetto Pro Q4 a due shard è visibile e scaricabile, ma
mantiene il badge di solo download e non diventa automaticamente il modello
attivo. Le tre varianti GLM 5.2 sono visibili e scaricabili dal loro
repository HF dedicato, ma restano di solo download. MTP e Qwen non compaiono
nel catalogo principale.
