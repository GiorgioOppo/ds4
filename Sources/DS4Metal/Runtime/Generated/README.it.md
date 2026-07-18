# Runtime/Generated

Artefatti generati necessari per distribuire l'app senza sorgenti Metal esterne.

## File principali

- [`KernelSources.swift`](KernelSources.swift): dizionario delle sorgenti
  `metal/*.metal`, concatenate da `MetalRuntime` nell'ordine canonico.

## Generazione

La sorgente autorevole è nella cartella `metal/` del repository. Dopo una
modifica eseguire `make embed-kernels`, che invoca `scripts/embed_kernels.sh` e
rigenera questo file; quindi compilare almeno il target `DS4Metal` e i test dei
kernel incorporati.

## Regole di modifica

Non modificare `KernelSources.swift` a mano e non aggiungere logica applicativa
in questa cartella. Le differenze devono essere riproducibili eseguendo il
generatore su un checkout pulito; aggiornare insieme ordine/nome dei kernel in
`MetalRuntime` quando cambia il set sorgente.
