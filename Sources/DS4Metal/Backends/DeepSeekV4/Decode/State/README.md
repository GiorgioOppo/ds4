**English** | [Italiano](README.it.md)

# DeepSeekV4/Decode/State

Temporary buffers reused along the forward pass to avoid per-token
allocations.

## Main files

- [`DecodeScratch.swift`](DecodeScratch.swift): scratch tensors for
  projections, attention, router, MoE, output and indices; includes special
  views for the quantized dense paths.
- [`LayerWeights.swift`](LayerWeights.swift): resident weights of one layer as
  GPUTensor, with the per-layer quantization flags (routed, dense Q4, Q8
  compressors) read by the dispatches of `DecodeLayer` and the streamers.

## Flow and dependencies

The scratch starts from the capacity required by the actually used context and
grows geometrically when a new high-water mark is exceeded. The configured
maximum capacity remains just a limit: it is not committed on first demand.
Replacement happens after the command buffers drain, and the scratch keeps
being shared sequentially by the graph's operations. It does not represent the
persistent conversation state: that belongs to the KV cache.

## Modification rules

Document for every buffer its logical shape, effective bytes, dtype and
lifetime. Aliases/views are allowed only if the lifetimes do not overlap; a
new allocation in the per-token path requires a measured justification.
