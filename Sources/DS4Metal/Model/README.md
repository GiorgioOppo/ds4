# Model

Descrizione Metal dell'architettura e strategie per trasformare un GGUF in pesi
utilizzabili dal decoder, residenti o in streaming.

## Struttura

- [`Architecture/`](Architecture/README.md): dimensioni e parametri DeepSeek-V4 Flash.
- [`Weights/`](Weights/README.md): validazione e assemblaggio dei pesi GGUF.
- [`Quantization/`](Quantization/README.md): metadati dei layout MoE quantizzati.
- [`Streaming/`](Streaming/README.md): ring di staging dei pesi densi da SSD.
- [`Experts/`](Experts/README.md): sidecar contiguo, I/O e supporto cache expert.

## Flusso

I metadati GGUF vengono validati contro `DSV4Shape`; il loader crea pesi piccoli
residenti e sceglie tra viste mmap, cache quantizzate, streaming denso ed expert
gather. `LayerWeights` presenta al decoder un contratto uniforme indipendente
dalla provenienza corrente dei byte.

Le opzioni sono descritte nella
[Configuration Reference](../../../README.md#configuration-reference).

## Regole di modifica

Fallire esplicitamente su forma o quantizzazione non supportata. Tenere separati
layout del modello, policy di residenza e meccanismo I/O. Ogni cache persistente
deve includere abbastanza identità/versione da non riusare byte incompatibili.
