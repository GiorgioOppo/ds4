# DwarfStar/Features/Diagnostics

- **`Controllers/DiagnosticsController.swift` / `Views/DiagnosticsView.swift`**
  eseguono il dump dei token e del rendering del chat template tramite il
  tokenizer nativo, senza sottoprocessi. È utile per verificare la
  tokenizzazione, il rendering dei prompt, gli schemi dei tool o la
  formattazione DSML delle tool call. La vista incorpora anche
  `EngineConsole`, una vista live dello stderr catturato dall'engine
  (diagnostica Metal/kernel).

Il controller esegue l'ispezione ed espone output osservabile; la vista si
limita a renderizzarlo. La diagnostica deve restare in sola lettura rispetto
all'inferenza attiva e allo stato KV.
