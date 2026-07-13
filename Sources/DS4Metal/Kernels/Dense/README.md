# Kernels/Dense

Wrapper delle moltiplicazioni dense usate da proiezioni, output head e prefill.

## File principali

- [`MetalDense.swift`](MetalDense.swift): matvec F16/Q8_0 e varianti Q4_K
  specializzate tramite function constants e numero di simdgroup.
- [`MetalMatmulMM.swift`](MetalMatmulMM.swift): matrix-matrix e percorsi batched,
  inclusi gli ingressi organizzati per id.

## Flusso e dipendenze

Il decode usa soprattutto matvec a batch uno; il prefill può aggregare token e
usare matmul per riutilizzare i pesi. `GraphContext` seleziona pipeline e
configurazione in base a quantizzazione e forma del tensore.

## Regole di modifica

Validare divisibilità dei blocchi quantizzati, dimensioni K/N e numero massimo
di thread/simdgroup. Non scegliere una variante solo dal byte count: il tipo
GGUF e il layout logico devono essere espliciti. Confrontare matvec e matmul sugli
stessi input.
