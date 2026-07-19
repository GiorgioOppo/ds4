**English** | [Italiano](README.it.md)

# DS4Metal

Multi-backend Metal runtime. The common primitives manage device, tensors,
pipelines and GPU operations; each model family owns its architecture,
weights, recurrent state, streaming and decode orchestration. The target
depends on `DS4Core` and `Metal.framework`.

## Structure

- [`Runtime/`](Runtime/README.md): device, command queue, pipelines and GPU tensors.
- [`Backends/`](Backends/README.md): family-specific model implementations.
- [`Model/`](Model/README.md): model types genuinely shared across backends.
- [`Kernels/`](Kernels/README.md): Swift wrappers for the `.metal` kernels.
- [`Graph/`](Graph/README.md): operations that compose the inference graph.

## Flow

`MetalRuntime` compiles the embedded sources. The selected backend converts
the tensor descriptors of `DS4Core.GGUFModel` into `GPUTensor`, sizes scratch
and KV cache, and orchestrates prefill/forward via `GraphContext`. The
DeepSeek-V4 backend is operational; the Qwen folder documents the prepared
boundary but does not yet provide a decoder.

## Change rules

- Numerical correctness and parity with the reference path come before
  optimizations.
- Declare layout, quantized type, offsets and synchronization explicitly.
- Do not add fields from different families to a single weights or scratch
  container.
- Backend selection happens outside the per-layer loop: the hot path stays
  concrete, with no per-operation dynamic dispatch.
- Do not edit generated code directly; regenerate it from the source.
- Document new `DS4_*` knobs in the main configuration and in the domain.
- Add targeted CPU or Metal tests for every new dispatch path.
