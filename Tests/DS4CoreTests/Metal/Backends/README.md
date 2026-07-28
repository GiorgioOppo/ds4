**English** | [Italiano](README.it.md)

# Backend Metal Tests

Tests organized by model family. The common runtime and kernel suites stay in
the top-level folders; here live shapes, GGUF schema, KV state and graph
compositions with backend-specific semantics.

- [`DeepSeekV4/`](DeepSeekV4/README.md): currently implemented backend.
- [`GLM52/`](GLM52/README.md): schema/DSA, oracles and progressive Metal
  primitives; no test declares the decoder available yet.
- [`Laguna/`](Laguna/README.md): tensor-schema contract only; the decoder is
  not ported yet.
