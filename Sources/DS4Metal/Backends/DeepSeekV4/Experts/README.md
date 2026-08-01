**English** | [Italiano](README.it.md)

# DeepSeekV4/Experts

Optimized direct-GGUF I/O paths for routed experts.

## Main files

- [`MetalIOCircuitBreaker.swift`](MetalIOCircuitBreaker.swift): disables MetalIO
  when performance/errors indicate an unreliable path.

## Flow

The cache gathers expert slabs directly from the GGUF with split parallel
preads. `DS4_PREAD_SPLIT=3` is the measured default.

## Modification rules

Keep split ranges disjoint and verify byte-identical output when changing I/O.
