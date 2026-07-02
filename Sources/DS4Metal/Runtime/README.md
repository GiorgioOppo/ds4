# DS4Metal/Runtime

Fondamenta del runtime Metal.

- **`MetalRuntime.swift`** — device, command queue, compilazione e cache delle pipeline dai sorgenti kernel embeddati.
- **`GPUTensor.swift`** — wrapper su `MTLBuffer` a storage condiviso (unified memory): `zeros`/`bytes`/`floats`, `count`, `byteOffset`, viste no-copy su mmap.
- **`KernelSources.swift`** — **GENERATO** da `metal/*.metal` via `scripts/embed_kernels.sh` (`make embed-kernels`). Non modificare a mano: edita i `.metal` e rigenera.
