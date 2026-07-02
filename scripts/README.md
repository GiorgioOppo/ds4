# Scripts

Build and analysis scripts.

- **`embed_kernels.sh`** regenerates
  `Sources/DS4Metal/Runtime/KernelSources.swift` from `metal/*.metal`, embedding
  kernel source code into the binary. It is invoked by `make embed-kernels`.

GGUF analysis tools, such as spectrum inspection and graph export, also belong in
this directory.
