# Graph/Core

Stato condiviso e primitive di dispatch del grafo Metal.

## File principali

- [`GraphContext.swift`](GraphContext.swift): raccoglie `MetalRuntime`, dimensioni,
  scratch e opzioni; offre helper per pipeline/command buffer e legge knob come
  `DS4_Q8_NSG`, `DS4_MOE_NSG`, `DS4_DENSE_Q4_NSG` e fusioni del compressor.

## Flusso e dipendenze

Il decoder costruisce il contesto; le estensioni in
[`Operations`](../Operations/README.md) lo usano per codificare fasi consecutive.
La configurazione delle threadgroup viene validata contro device, forma e
quantizzazione prima del dispatch.

## Regole di modifica

Non leggere variabili d'ambiente dentro ogni kernel se possono essere congelate
all'avvio. Validare range e limiti hardware dei valori configurabili. Gli helper
generici restano qui; la logica di una singola fase va in `Operations`.
