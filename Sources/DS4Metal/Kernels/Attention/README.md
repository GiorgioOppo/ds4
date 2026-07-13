# Kernels/Attention

Wrapper delle operazioni di attention standard e sparsa NSA.

## File principali

- [`MetalFlashAttn.swift`](MetalFlashAttn.swift): attention sul contesto KV.
- [`MetalAttnOutLow.swift`](MetalAttnOutLow.swift): proiezione low-rank dell'output.
- [`MetalRoPE.swift`](MetalRoPE.swift): rotazione/inversa delle componenti RoPE.
- [`MetalIndexerScore.swift`](MetalIndexerScore.swift): score di rilevanza delle righe compresse.
- [`MetalIndexerPool.swift`](MetalIndexerPool.swift): pooling dell'indexer.
- [`MetalSparseSelect.swift`](MetalSparseSelect.swift): selezione top-k sul device.

## Flusso e dipendenze

Il decoder proietta query/KV, applica RoPE e aggiorna le cache. L'indexer assegna
score alle righe compresse; una selezione GPU o il fallback CPU decide il
sottoinsieme letto da flash attention, poi la proiezione d'uscita torna allo
spazio residuo.

## Regole di modifica

Rispettare stride, numero di head, finestra raw e contatori delle righe compresse.
Tie-break del top-k GPU e CPU devono coincidere. Verificare contesti vuoti,
parziali, wrap della ring buffer e offset non allineati del mmap.
