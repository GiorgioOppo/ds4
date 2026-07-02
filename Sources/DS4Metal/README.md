# DS4Metal

Runtime Metal in puro Swift + grafo di decode: port fedele di `ds4_metal.m`. Compila ed esegue i kernel di `metal/*.metal` (embeddati nel binario) e li dispatcha. Dipende da `DS4Core`; linka `Metal.framework`.

- **`Runtime/`** — `MetalRuntime` (device/pipeline), `GPUTensor` (buffer condivisi), `KernelSources.swift` (sorgenti kernel embeddati — **generato**).
- **`Model/`** — shape compilata (`DSV4Shape`) e caricamento pesi dal GGUF (`GGUFWeights`: mmap no-copy + gather esperti).
- **`Decode/`** — `StreamingDecoder` (forward/prefill/slice), grafo di decode, KV cache (raw window + compressore NSA), slot-cache esperti, snapshot KV.
- **`Kernels/`** — un wrapper Swift per ogni kernel Metal (matvec MoE, flash-attention, RoPE, norm, ecc.).

La correttezza (regola #1) è validata dai test in `Tests/DS4CoreTests`.
