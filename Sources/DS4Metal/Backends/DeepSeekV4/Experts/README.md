# DeepSeekV4/Experts

Sidecar format and optimized I/O paths for the routed experts, the most
expensive weights to fetch during generation.

## Main files

- [`ExpertBundle.swift`](ExpertBundle.swift): state and header of the `.expbundle` sidecar.
- `ExpertBundle+Builder.swift` and `ExpertBundle+Layout.swift`: path selection,
  layout validation and atomic construction of the bundle.
- `ExpertBundle+Read.swift`: CPU/pread reads of the contiguous slabs.
- `ExpertBundle+MetalIO.swift`: direct loads through MetalIO when available.
- `ExpertBundle+Usage.swift`: usage information and bundle diagnostics.
- [`MetalIOCircuitBreaker.swift`](MetalIOCircuitBreaker.swift): disables MetalIO
  when performance/errors indicate an unreliable path.

## Flow

The builder reorders each expert's gate/up/down into contiguous regions. At
runtime the cache requests a list of experts and the bundle fills disjoint
slots with one batch read; without a valid sidecar or MetalIO, the factory
falls back to the gather from the GGUF without changing the layout delivered
to the kernels.

## Modification rules

Verify GGUF identity, sizes, offsets and completeness before using the
sidecar. Construction must publish only complete files. MetalIO must have a
correct CPU fallback and cannot change the bytes; bump the version/layout if
the slab order changes.
