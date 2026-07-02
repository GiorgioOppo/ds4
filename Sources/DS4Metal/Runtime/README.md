# DS4Metal/Runtime

Foundational Metal runtime pieces.

- **`MetalRuntime.swift`** owns the device, command queue, and pipeline
  compilation/cache for the embedded kernel sources.
- **`GPUTensor.swift`** wraps shared-storage `MTLBuffer` allocations on unified
  memory. It exposes helpers such as `zeros`, `bytes`, `floats`, `count`,
  `byteOffset`, and no-copy views over mmap-backed data.
- **`KernelSources.swift`** is **generated** from `metal/*.metal` by
  `scripts/embed_kernels.sh` through `make embed-kernels`. Do not edit it by
  hand; edit the `.metal` files and regenerate.
