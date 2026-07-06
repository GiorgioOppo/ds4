# DS4Metal

Pure Swift Metal runtime plus decode graph: a faithful port of `ds4_metal.m`.
This target compiles the embedded `metal/*.metal` kernels, owns the GPU dispatch
path, and links `Metal.framework`. It depends on `DS4Core`.

- **`Runtime/`**: `MetalRuntime` for device/pipeline setup, `GPUTensor` for
  shared-memory buffers, and generated embedded kernel sources.
- **`Model/`**: compiled model shape (`DSV4Shape`), GGUF-backed weight loading
  (`GGUFWeights`) with no-copy mmap access and expert gathering, dense-weight
  streaming (`DenseStreamer`), and the contiguous expert sidecar (`ExpertBundle`).
- **`Decode/`**: `StreamingDecoder`, forward/prefill/slice execution, decode graph,
  KV cache, raw window plus NSA compressor, expert slot cache, and KV snapshots.
- **`Kernels/`**: one Swift wrapper per Metal kernel, covering MoE matvec,
  flash attention, RoPE, normalization, utility kernels, and related dispatch
  glue.

Correctness is rule #1 and is validated by tests in `Tests/DS4CoreTests`.
