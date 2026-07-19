**English** | [Italiano](README.it.md)

# GLM 5.2 Metal backend

This tree owns numerical code for the `glm-dsa` architecture. GLM code stays
outside `Backends/DeepSeekV4` so model-family decisions are made once at the
runtime boundary rather than inside hot token loops.

The first correctness tranche implements the architecture-exact sigmoid
router: 256 experts, top-8 selection, selection-only bias, unbiased normalized
weights and scale 2.5. Both a scalar oracle and the Metal kernel are kept so
later MoE, resident and streaming paths can be checked against the same result.

Compact DSA now also has isolated numerical fixtures for KV-LoRA RMSNorm,
compact F16 KV placement, centered indexer-key LayerNorm/affine/RoPE placement,
and fixed-geometry indexer scoring. These remain validation primitives rather
than an executable graph.

`Streaming/` starts roadmap step 1: `GLM52PayloadReader` executes the validated
weight map and expert stream plans against the real GGUF payload with bounded
`pread`s — descriptors and top-8 plans become bytes, packed as gate|up|down
records in router rank order.

The backend is not registered as runnable yet. Compact DSA attention, tensor
loading, dense/MoE execution and complete prefill/decode must pass their fixtures
before `BackendCapabilities.generation` or catalog selection is enabled.
