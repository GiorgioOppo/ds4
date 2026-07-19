[English](README.md) | **Italiano**

# Diagnostica di DS4Demo

Helper mirati usati dall'eseguibile a riga di comando:

- `DemoLog.swift` mantiene il logging diagnostico su `stderr`, separato dal
  testo generato scritto su `stdout`. Possiede anche la traccia A/B dei logit,
  opt-in e limitata: i vettori vengono conservati copy-on-write durante
  l'inferenza e serializzati solo dopo la regione cronometrata, per
  `scripts/metal_ab.sh`.
- `DiskBenchmark.swift` misura il comportamento dello storage rilevante per i
  pesi in streaming.
- `ModelDiagnostics.swift` riporta informazioni sui tensori GGUF e sul
  tokenizer.

La diagnostica può osservare e riportare il comportamento a runtime, ma non
deve cambiare silenziosamente le impostazioni di generazione. Mantieni
l'output abbastanza stabile per i confronti di benchmark e la cattura da
shell.

Il resoconto della cache degli esperti separa gli hit/miss sui layer
cacheabili dalle selezioni in bypass e riporta l'hit-rate globale risultante.
I byte di look-ahead sono riportati separatamente perché sono traffico di
storage reale nascosto fuori dal timer del gather sul percorso critico. Le
righe slot/RAM per layer sono istantanee di pool realmente esistiti nella run
misurata, mai un'allocazione post-run ricalcolata da una cronologia di
routing cambiata mentre la run era in esecuzione.
