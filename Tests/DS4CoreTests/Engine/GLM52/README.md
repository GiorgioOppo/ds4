**English** | [Italiano](README.it.md)

# Engine tests — GLM 5.2 service

Tests for the DS4Engine-level GLM 5.2 surfaces (`GLM52ChatService` and its
benchmark/accuracy helpers), as opposed to the Metal engine tests under
`Metal/Backends/GLM52`.

- `GLM52AccuracyCandidatesTests.swift` pins the partial-selection Top-K
  scorer used by the GLM Correctness benchmark: descending order,
  lowest-id-wins ties (the greedy argmax rule) and equivalence with a full
  sort on a large vocabulary.
- `GLM52DiskKVStoreTests.swift` covers the prefix-keyed checkpoint store
  over synthetic GKV1 files: longest-prefix lookup, dedup/minTokens,
  store-interval policy, supersede eviction under a token budget, legacy
  `state.glmkv` adoption, index rebuild and foreign-file tolerance.

Rules: tests here must not require a real GGUF or Metal device; anything that
executes the engine belongs with the Metal-side GLM tests.
