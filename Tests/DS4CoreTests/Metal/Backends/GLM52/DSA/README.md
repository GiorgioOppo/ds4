# GLM compact DSA tests

Pure CPU tests cover compact-cache bytes, lazy slab growth, the 21-layer
IndexShare schedule, weighted ReLU scoring, causal deterministic top-k and
selection reuse. No Metal device or model GGUF is required.

The attention-reference suite proves the expanded and absorbed evaluation
orders of `GLM52AttentionCPUReference` agree within float tolerance, that
selection order is irrelevant, that degenerate selections reduce to plain
value projections, and that malformed inputs (empty/duplicate/out-of-range
selections, wrong dimensions, non-finite values) are rejected before any
computation.
