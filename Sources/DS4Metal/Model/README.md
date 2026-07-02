# DS4Metal/Model

Model shape definitions and GGUF-to-GPU weight loading.

- **`DSV4Shape.swift`** contains compiled model constants such as layer count,
  attention head count, `headDim`, `nSWA`, compression ratios, expert count, and
  the derived `DSV4Dims` values used by kernels.
- **`GGUFWeights.swift`** assembles layer weights from the GGUF. Non-routed weights
  stay mmap-backed and no-copy through the page cache, while the selected 6/256
  experts are gathered into contiguous slabs. It also provides primitives used by
  the expert slot cache.
