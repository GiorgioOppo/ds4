**English** | [Italiano](README.it.md)

# Graph

Composition layer between the decoder and the Metal kernel wrappers. It
translates the model's mathematical phases into dispatches over tensors and
command buffers.

## Structure

- [`Core/`](Core/README.md): shared context, pipelines, and configuration.
- [`Operations/`](Operations/README.md): attention, compressor, MoE, router,
  output, and element-wise transform operations.

## Flow and dependencies

The [`DeepSeek-V4 decode`](../Backends/DeepSeekV4/Decode/README.md) creates or reuses a `GraphContext`; the
extensions in `Operations` choose the [`Kernels`](../Kernels/README.md)
wrapper, set buffers/offsets, and encode the dispatch. Tensors are provided by
[`Runtime`](../Runtime/README.md) and weights by the selected backend.

## Modification rules

The graph orchestrates but must not duplicate the Metal code. Every operation
must make explicit its shape, quantization, command buffer ownership, and
synchronization requirements. Keep the extensions split by mathematical phase.
