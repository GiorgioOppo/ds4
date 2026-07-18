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
`GLM52IndexerCPUReference.causalTopK` on distinct scores — single-pass and merge
path — and that `-INFINITY` causal rows are never selected.

The DSA-chain suite composes the GPU primitives end to end (indexer scores →
top-k → staged compact attention) against the same chain run through the CPU
oracles: selections must match exactly, attention output within tolerance.

The Q8_0 suite quantizes the projection weights with the shared test
quantizer, dequantizes the same bytes, and requires the Q8 kernels to match
the F32 baselines on those dequantized values — quantization error belongs to
the fixture, never to the kernel.

The rope-tail suite pins identity at position 0, untouched nope prefixes,
per-pair norm preservation, inverse composition, and GPU-vs-oracle parity on
query heads and the single K row at moderate positions (fp32 trigonometry of
huge angles diverges by argument reduction — the documented rope caveat).

The MoE suite applies the same discipline to the quantized FFN kernels:
Q4_K fixtures from the real quantizer, synthesized Q2_K/Q5_K/Q6_K blocks
decoded by the `Quantize` references, Q8_0 dense/output-head paths on widths
that are multiples of 32 but not 256, stage and chained comparisons against
`GLM52FFNCPUReference`, and contract rejections (types, sizes, widths).
