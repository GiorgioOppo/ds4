# Graph/Core

Shared state and dispatch primitives of the Metal graph.

## Main files

- [`GraphContext.swift`](GraphContext.swift): gathers `MetalRuntime`,
  dimensions, scratch and options; provides pipeline/command buffer helpers
  and reads knobs such as `DS4_Q8_NSG`, `DS4_MOE_NSG`, `DS4_DENSE_Q4_NSG` and
  the compressor fusions.

## Flow and dependencies

The decoder builds the context; the extensions in
[`Operations`](../Operations/README.md) use it to encode consecutive phases.
The threadgroup configuration is validated against device, shape and
quantization before dispatch.

## Change rules

Do not read environment variables inside every kernel if they can be frozen
at startup. Validate ranges and hardware limits of configurable values.
Generic helpers stay here; the logic of a single phase belongs in
`Operations`.
