# DS4Metal/Model

Shape del modello e caricamento dei pesi dal GGUF verso la GPU.

- **`DSV4Shape.swift`** — costanti compilate del modello (n. layer, head, headDim, `nSWA`, ratio di compressione, n. esperti…) e i `DSV4Dims`.
- **`GGUFWeights.swift`** — assembla i pesi di un layer dal GGUF: pesi non-routed **mmap no-copy** (residenti via page cache), gather dei 6/256 esperti selezionati (slab contigui), primitive per lo slot-cache.
