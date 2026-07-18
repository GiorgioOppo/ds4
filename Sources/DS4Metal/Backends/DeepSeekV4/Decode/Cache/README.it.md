# DeepSeekV4/Decode/Cache

Cache residente degli expert MoE e storico della frequenza di routing.

## File principali

- [`ExpertSlotCache.swift`](ExpertSlotCache.swift): pool LRU per layer, riempimento
  concorrente, prefetch speculativo e protezione delle letture GPU in-flight.
- [`ExpertUsage.swift`](ExpertUsage.swift): statistiche thread-safe, warm set,
  allocazione adattiva degli slot e persistenza JSON. Nei modelli mixed-quant
  l'allocatore lavora su un budget in byte: il costo reale del record per layer
  impedisce ai pool Q4 di far crescere silenziosamente la RAM. Senza una storia
  completa usa un piano byte-balanced; i layer hash con look-ahead esatto
  possono restare al floor per lasciare budget ai router non prevedibili.

## Flusso

Il router seleziona gli expert; `acquire` traduce gli id in slot, serve gli hit e
riempie i miss dal GGUF o dall'expert bundle. Il look-ahead prepara il layer
successivo senza bloccare la domanda. Le selezioni reali aggiornano le
statistiche, riutilizzate nelle sessioni successive per il warm-up.
Il piano di slot viene congelato una volta per generazione della cache. Le
diagnostiche leggono i pool effettivamente materializzati, non ricalcolano una
nuova allocazione usando le statistiche mutate durante la generazione.

## Regole di modifica

La concorrenza è serializzata per layer e lo stato globale ha un lock separato:
non invertire questo ordine. Uno slot letto da un command buffer non può essere
evinto finché è in-flight. Conservare il floor `k+2` e validare gli id caricati.
