# Runtime/Generated

Generated artifacts required to ship the app without external Metal sources.

## Main files

- [`KernelSources.swift`](KernelSources.swift): dictionary of the
  `metal/*.metal` sources, concatenated by `MetalRuntime` in canonical order.

## Generation

The authoritative source lives in the repository's `metal/` folder. After a
change, run `make embed-kernels`, which invokes `scripts/embed_kernels.sh` and
regenerates this file; then build at least the `DS4Metal` target and the
embedded-kernel tests.

## Modification rules

Do not edit `KernelSources.swift` by hand and do not add application logic to
this folder. Diffs must be reproducible by running the generator on a clean
checkout; update kernel order/names in `MetalRuntime` together with any change
to the source set.
