# scripts

Script di build/analisi.

- **`embed_kernels.sh`** — rigenera `Sources/DS4Metal/Runtime/KernelSources.swift` da `metal/*.metal` (embed dei sorgenti kernel nel binario). Invocato da `make embed-kernels`.

Eventuali altri tool di analisi GGUF (spettro, export grafo) vanno qui.
