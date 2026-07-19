[English](README.md) | **Italiano**

# Runtime

Fondazione Metal sulla quale poggiano loader, grafo e decoder.

## Struttura

- [`Core/`](Core/README.it.md): device, libreria, pipeline e `GPUTensor`.
- [`Generated/`](Generated/README.it.md): sorgenti kernel incorporate e generate.

## Flusso e dipendenze

`MetalRuntime` crea device e command queue, concatena le sorgenti nell'ordine
canonico e compila una `MTLLibrary`. I wrapper richiedono pipeline per nome e
operano su `GPUTensor` in memoria unificata. In produzione non è necessaria una
cartella `metal/` accanto all'app.

## Regole di modifica

Tenere questo livello indipendente dalla semantica DeepSeek quando possibile.
Centralizzare creazione e cache delle pipeline, propagare errori descrittivi e
non introdurre sincronizzazioni implicite nei contenitori tensor.
