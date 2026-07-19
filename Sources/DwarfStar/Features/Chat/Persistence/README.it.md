[English](README.md) | **Italiano**

# Persistenza della Chat

Questa directory possiede la rappresentazione su disco delle conversazioni
dell'app.

- `ChatSession.swift` definisce i record `Codable` di sessione, messaggio,
  chiamata di tool e sub-agent e lo `ChatSessionStore` basato su JSON.
- `ChatMessageMapping.swift` converte tra record persistiti, messaggi della UI
  e ruoli dell'engine.
- `DS4Engine/Inference/Autotuning/MachineAutoTuneTransactionStore.swift`
  possiede il commit a due fasi a prova di crash per un vincitore validato
  dell'auto-tune della macchina. Registra atomicamente
  `prepared → installed → committing → committed` e ripristina lo snapshot
  iniziale completo dei knob al successivo avvio se una fase non terminale è
  stata interrotta.

Il flusso è `UIMessage` -> `StoredMessage` -> un file JSON per chat sotto
Application Support, e l'inverso quando una sessione viene aperta. Preserva la
decodifica retrocompatibile quando aggiungi campi; usa default o proprietà
opzionali invece di rendere illeggibili i file di chat esistenti.
