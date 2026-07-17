# GLM 5.2 Metal tests

These tests compare architecture-owned GLM Metal kernels with independent
scalar oracles. They use synthetic deterministic inputs and therefore run
without downloading the 200+ GB model.

The router suite covers selection-only bias, unbiased normalized route weights,
top-8 geometry, stable sigmoid extremes and deterministic expert-id tie breaks.

The compact-KV suite checks exact F32-to-F16 bit patterns, non-zero `pos0`,
capacity rejection and preservation of cache rows outside the written range.

The KV-LoRA normalization suite verifies RMSNorm is restricted to the 512-wide
prefix and that every raw F32 bit in the 64-wide K-RoPE payload survives.

The indexer-key store suite distinguishes centered LayerNorm from RMSNorm,
checks that only `0..<64` receives RoPE, verifies non-zero cache positions and
preserves rows outside the write range.

The indexer suite covers the fixed 32x128 geometry, F16 key-cache reads,
per-head ReLU before weighting, positive scale, token-major output and causal
`-infinity` masking. Metal comparisons skip when no device is available; the
independent scalar oracles and validation tests remain device-free.

The streaming suite (`Streaming/`) proves the payload reader byte-faithful on
synthetic pattern files and that its bound checks reject truncated files and
malformed plans before any byte moves.

The compact-attention suite compares the staged kernels (`qk_lowrank`,
`attention_indexed`, `value_project`) stage by stage against scalar dots and
chained end to end against `GLM52AttentionCPUReference` on the same
F16-rounded cache, plus the selection/geometry rejection paths.

The top-k suite proves the multi-block argsort+merge dispatch equal to
`GLM52IndexerReference.causalTopK` on distinct scores — single-pass and merge
path — and that `-INFINITY` causal rows are never selected.
