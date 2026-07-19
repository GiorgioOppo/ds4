[English](README.md) | **Italiano**

# DeepSeekV4/Model

Caricamento dei pesi DeepSeek V4 dal GGUF verso GPUTensor: primitive per-tensore
e assemblaggio per layer/output usati sia dal percorso all-resident sia dallo
streaming.

## File principali

- [`GGUFWeights.swift`](GGUFWeights.swift): loader per-tensore (copia, vista
  mmap no-copy), assemblaggio `LayerWeights` (`layer`, `layerMappedExperts`,
  `layerMappedDense`, `layerSmallSkeleton`), output head, rilevamento e
  validazione delle quantizzazioni routed (`detectMoEQuant`,
  `validateRuntimeLayout`), gather/copia degli expert selezionati e primitive
  `pread`/`madvise` condivise dagli streamer.

## Flusso

`validateRuntimeLayout` chiude il contratto al bind: tipi quant ammessi, gate/up
coerenti, forme conformi alla geometria e tabella hash presente dove richiesta.
Solo dopo si assemblano i `LayerWeights`; gli expert routed possono restare
dummy (gather on-demand), viste mmap dell'intero tensore o slot della cache a
seconda del factory chiamante.

## Regole di modifica

Le letture non devono mai superare i limiti dichiarati dal descrittore GGUF.
Ogni builder di `LayerWeights` deve passare da `setExpertQuant` (quant per-layer
mixed-precision). Le primitive `pread` restano thread-safe su fd condiviso
(offset espliciti); gli hint `madvise` sono advisory e non possono cambiare la
numerica.
