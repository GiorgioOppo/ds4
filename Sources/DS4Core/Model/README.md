# Model

Descrizione portabile della variante e della forma del modello.

## File principali

- [`ModelShape.swift`](ModelShape.swift): `ModelVariant`, errori di
  configurazione, valori predefiniti, `ModelShape` e `ModelConfig`.

## Flusso e dipendenze

I metadati GGUF vengono confrontati con la forma prevista; i livelli superiori
usano questa configurazione per dimensionare cache e validare opzioni. Le
costanti specifiche dei kernel DeepSeek-V4 restano in
[`DS4Metal/Model/Architecture`](../../DS4Metal/Model/Architecture/README.md).

## Regole di modifica

Tenere separati default generali e dettagli del backend. Nuove varianti devono
avere validazione esplicita e valori derivati controllati; non duplicare qui
costanti già lette dal GGUF.
