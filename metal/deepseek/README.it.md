[English](README.md) | **Italiano**

# Kernel DeepSeek V4

Sorgenti dei kernel Metal per il grafo DeepSeek V4 più le op generiche
condivise (normalizzazione, softmax, argsort, copy/cast, unarie, glue di
riduzione) che il decoder DeepSeek pilota oggi. I nomi dei kernel sono
globalmente unici: Metal compila un'unica libreria da ogni file vendorizzato,
quindi un backend futuro può riutilizzare le op generiche da qui senza
duplicarle.

Il flusso di lavoro di modifica, l'embedding (`make embed-kernels`) e lo stato
dell'audit vivono in [`../README.md`](../README.it.md); la policy
runtime/wrapper è in [`docs/BACKEND-METAL.md`](../../docs/BACKEND-METAL.it.md).
