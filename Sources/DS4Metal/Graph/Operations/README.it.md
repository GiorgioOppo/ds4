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
[`Backends/DeepSeekV4/Decode/Execution`](../../Backends/DeepSeekV4/Decode/Execution/README.md).

## Attention decode: raw ring e split-K

`GraphAttention.swift` materializza per FlashAttention prima le righe raw della
finestra SWA e poi le righe compresse. Con `DS4_RAW_RING=1` il raw KV è un
`MTLBuffer` in memoria shared/unificata di Metal, non una KV cache su SSD e non
un alias dei checkpoint Disk-KV. Quando la finestra circolare attraversa la fine
del buffer, `kernel_dsv4_raw_ring_cpy_f32_f16` la riordina cronologicamente e la
converte F32→F16 con una sola dispatch GPU 2D; il caso contiguo mantiene la copia
lineare esistente.

Lo split-K adattivo usa il numero totale di righe visibili, cioè
`totalRows = rawRows + compressedRows`, e sceglie esattamente:

```text
nwg = min(32, max(1, ceil(totalRows / 32)))
```

Non arrotonda più `nwg` alla successiva potenza di due: 128 righe usano 4
workgroup, 129 ne usano 5, 161 ne usano 6. I kernel vec/reduce supportano tutti
i valori 1…32; con `nwg == 1` il reduce separato non serve. Impostare
`DS4_ADAPTIVE_SPLITK=0` ripristina sempre 32 workgroup per un confronto A/B.

## Regole di modifica

Una funzione deve rappresentare una responsabilità matematica riconoscibile.
Non nascondere wait CPU o readback in helper apparentemente GPU-only. Quando una
fusione cambia l'ordine floating-point, mantenerla disattivabile e confrontarla
con il percorso non fuso.
