**English** | [Italiano](README.it.md)

# DeepSeek-V4 Graph Tests

DeepSeek-V4-specific `GraphContext` compositions: latent Q/KV, MLA attention,
low-rank output, NSA compressor, top-6 router, routed MoE, HyperConnections
and complete decoder/layers.

Tests for the reusable primitives remain in
[`Metal/Graph`](../../../Graph/README.md); edge cases of individual kernels
remain in `Metal/Kernels`.
