# Diagnostics

Primitive portabili per comunicare lo stato delle operazioni lunghe.

## File principali

- [`LoadProgress.swift`](LoadProgress.swift): singleton thread-safe che pubblica
  frazione e descrizione dello stadio durante apertura GGUF, preparazione cache e
  caricamento dei pesi.

## Flusso e dipendenze

Il produttore chiama `reset`, `set`, `begin` e `advance`; UI o servizio leggono
periodicamente `snapshot`. La sincronizzazione avviene con `NSLock`, senza
dipendenze da SwiftUI o Metal.

## Regole di modifica

Le scritture possono arrivare da worker concorrenti: non esporre direttamente lo
stato mutabile e mantenere `snapshot` economico. Nuove metriche strutturate
devono restare indipendenti dalla UI.
