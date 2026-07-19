[English](README.md) | **Italiano**

# Kernels/Tensor

Wrapper delle operazioni tensor generiche riutilizzate dal grafo.

## File principali

- `MetalNorm.swift`, `MetalSoftmax.swift`, `MetalGLU.swift`, `MetalUnary.swift` e
  `MetalBin.swift`: normalizzazione, attivazioni e operazioni element-wise.
- `MetalGetRows.swift`, `MetalSetRows.swift`, `MetalSumRows.swift`: accesso e
  riduzione per righe.
- `MetalCopy.swift`, `MetalConcat.swift`, `MetalRepeat.swift`: movimento/forma dati.
- `MetalArgsort.swift`: ordinamento degli indici.

## Flusso e dipendenze

Le operazioni vengono codificate nello stesso command buffer delle fasi del
modello quando possibile, lavorando su `GPUTensor` condivisi. Sono primitive del
backend e non contengono policy di decode o caricamento.

## Regole di modifica

Supportare correttamente tensori vuoti o parziali secondo il contratto della
funzione e rispettare `byteOffset`. Rendere espliciti broadcasting, inplace e
intervalli sovrapposti. Aggiungere test piccoli con risultato CPU noto prima di
usare una primitiva in una fusione complessa.
