**English** | [Italiano](README.it.md)

# Runtime/Core

Minimal abstractions over Metal and Apple Silicon unified memory.

## Main files

- [`MetalRuntime.swift`](MetalRuntime.swift): selects the device, creates
  queue and library, concatenates the embedded kernels and caches the compute
  pipelines.
- [`GPUTensor.swift`](GPUTensor.swift): buffer with logical length, byte
  offset, zero-copy views, resident/mmap allocations and best-effort page
  locking.

## Flow

The runtime is created once per decoder. Loaders and scratch build
`GPUTensor`s; the kernel wrappers bind `buffer` and `byteOffset` to the
encoders. The mmap views share the page cache, while subviews and staging
share an already allocated Metal buffer.

## Modification rules

Every binding must honor `byteOffset`, not just the base buffer. Use
uninitialized buffers only when the write covers the whole range before the
read. Do not hold CPU pointers beyond the buffer/mmap lifetime and do not
create a pipeline in the per-token path if it can be cached.
