# DS4Core/Streaming

Supporto allo streaming dei pesi da SSD (il modello non sta tutto in RAM).

- **`SSDCachePlan.swift`** — pianificazione di cosa tenere residente vs streammare.
- **`SimulatedMemoryLock.swift`** — lock di memoria simulato (per ragionare sul working set senza wiring reale).
