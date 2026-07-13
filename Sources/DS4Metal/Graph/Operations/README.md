# Graph/Operations

Operazioni del grafo raggruppate per fase matematica del modello.

## File principali

- `GraphAttention.swift` e `GraphAttentionOutput.swift`: attention e proiezione d'uscita.
- `GraphRopeKV.swift`: RoPE, scrittura e lettura KV.
- `GraphCompressor*.swift` e `GPUTensor+Compressor.swift`: pooling, compressor e indexer NSA.
- `GraphRouter.swift`, `GraphMoE.swift` e `GraphMoEMM.swift`: routing e FFN routed,
  inclusa la variante matrix-matrix del prefill.
- `GraphHyperConnections.swift`: reduce/expand delle HyperConnections.
- `GraphOutput.swift` e `GraphUnary.swift`: output head e trasformazioni semplici.

## Flusso

Le operazioni sono estensioni di `GraphContext`: ricevono tensori già dimensionati,
selezionano pipeline e quantizzazione, codificano i kernel e restituiscono il
tensor/stato per la fase seguente. Il percorso esatto è orchestrato da
[`Decode/Execution`](../../Decode/Execution/README.md).

## Regole di modifica

Una funzione deve rappresentare una responsabilità matematica riconoscibile.
Non nascondere wait CPU o readback in helper apparentemente GPU-only. Quando una
fusione cambia l'ordine floating-point, mantenerla disattivabile e confrontarla
con il percorso non fuso.
