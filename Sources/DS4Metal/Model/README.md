# DS4Metal/Model

Model shape definitions and GGUF-to-GPU weight loading.

- **`DSV4Shape.swift`** contains compiled model constants such as layer count,
  attention head count, `nHeadDim`, per-layer compression ratios and RoPE
  parameters, expert count, and the derived `DSV4Dims` values used by kernels.
- **`GGUFWeights.swift`** assembles layer weights from the GGUF. Non-routed weights
  stay mmap-backed and no-copy through the page cache, while the selected 6/256
  experts are gathered into contiguous slabs. It also provides primitives used by
  the expert slot cache.
- **`DenseStreamer.swift`** streams the per-layer dense weights through a
  pread/F_NOCACHE staging ring instead of keeping them resident
  (`DS4_DENSE_STREAM`, read-ahead depth `DS4_DENSE_AHEAD`). It also hosts the
  resident Q4 requant of the big attention projections (`DS4_DENSE_Q4`, plus
  `DS4_SHARED_Q4` and the `DS4_Q4_CACHE_DIR` requant cache) and the resident
  NSA-compressor projections (`DS4_RESIDENT_COMP`).
- **`ExpertBundle.swift`** builds and serves the opt-in `<gguf>.expbundle`
  sidecar that repacks each routed expert's gate/up/down slabs contiguously
  (`DS4_EXPERT_BUNDLE`, location override `DS4_BUNDLE_DIR`).

These knobs are documented in the root README's
[Configuration Reference](../../../README.md#configuration-reference).
