[English](README.md) | **Italiano**

# Test Metal

Validazione per `DS4Metal`, suddivisa per livello di astrazione:

- `Kernels/`: singole primitive GPU confrontate con riferimenti CPU.
- `Graph/`: primitive del grafo indipendenti dal modello.
- `Backends/DeepSeekV4/`: comportamento di grafo, modello e decode specifico
  di DeepSeek-V4.
- `Runtime/`: creazione di device, libreria e pipeline.

I casi dipendenti dalla GPU devono saltare esplicitamente quando Metal non è
disponibile. Un test saltato non è un pass; vedi
[`METAL-TESTS.md`](../../METAL-TESTS.it.md) per le convenzioni.
