# DwarfStar/Bench

- **`BenchController.swift`** — benchmark nativo: misura prefill + generazione (token/s) a contesti crescenti. Motore selezionabile **Locale** (engine in-process) o **Distribuito** (riusa il coordinatore già connesso).
- **`BenchView.swift`** — UI: selettore motore, frontiere di contesto, grafico del throughput (Swift Charts), indicatore del motore in esecuzione.
