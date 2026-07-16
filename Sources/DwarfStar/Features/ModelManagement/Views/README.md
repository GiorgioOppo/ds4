# Model Management Views

- `ModelPicker.swift` usa `NSOpenPanel` e bookmark security-scoped per GGUF
  esterni. Prima di accettarli chiama l'ispezione Engine e `BackendSelector`,
  così profili sconosciuti, MTP, shard e Qwen non vengono salvati come runtime locale valido.
- `DownloadView.swift` rende esclusivamente `DeepSeekV4ModelCatalog`: tipo di
  artifact, disponibilità runtime, stato locale, spazio, fase e progresso. Offre
  Scarica/Riprendi, Annulla/Riprova e Seleziona per Flash e Pro Q2 runnable.

La sheet è raggiungibile sia dal preload della Chat sia dalle Settings. Non si
chiude mentre un download è attivo; la selezione di un file gestito neutralizza
un vecchio bookmark esterno e persiste il path in `AppSettings`.
