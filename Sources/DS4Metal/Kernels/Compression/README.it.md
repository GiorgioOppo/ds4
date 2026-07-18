# Kernels/Compression

Wrapper per riduzione/espansione delle HyperConnections e compressione ricorrente
del contesto NSA.

## File principali

- [`MetalHCSplit.swift`](MetalHCSplit.swift): separazione dei flussi HC.
- [`MetalHyperConnections.swift`](MetalHyperConnections.swift): reduce/expand e fusioni HC.
- [`MetalCompressor.swift`](MetalCompressor.swift): proiezioni e stato del compressor.
- [`MetalKVCompress.swift`](MetalKVCompress.swift): scrittura delle righe KV compresse.

## Flusso e dipendenze

Prima dell'attention lo stato residuo viene ridotto; compressor e gate aggregano
token raw in righe NSA persistenti. Dopo attention/FFN, l'uscita viene espansa
nei flussi HyperConnection. Le operazioni sono composte da
[`Graph/Operations`](../../Graph/Operations/README.md).

## Regole di modifica

Contatori, rapporto di compressione e ordine degli aggiornamenti sono stato
ricorrente: una variazione può invalidare KV snapshot e parità prefill/decode.
Una fusione deve conservare un percorso non fuso per verifica numerica.
