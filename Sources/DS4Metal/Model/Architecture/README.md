# Model/Architecture

Costanti compilate e dimensioni derivate del modello DeepSeek-V4 Flash servito
dai kernel Metal.

## File principali

- [`DSV4Shape.swift`](DSV4Shape.swift): layer, head, expert, rapporti di
  compressione e costanti architetturali attese.
- [`DSV4Dims.swift`](DSV4Dims.swift): dimensioni runtime derivate e flag delle
  fusioni/kernel configurabili.
- [`RopeParams.swift`](RopeParams.swift): parametri per RoPE e scaling del contesto.

## Flusso e dipendenze

Il loader confronta i metadati GGUF con `DSV4Shape`, quindi costruisce `DSV4Dims`.
Decoder, scratch, grafo e wrapper kernel condividono questa singola descrizione
per calcolare shape, stride e capacità KV.

## Regole di modifica

Non correggere una forma incompatibile con fallback silenziosi. Distinguere
costanti del modello da tuning hardware. Ogni nuovo campo deve indicare unità,
origine e consumatori; aggiornare validazione e test di allocazione insieme.
