**English** | [Italiano](README.it.md)

# DeepSeekV4/Decode/Attention

CPU support logic for NSA sparse attention selection.

## Key files

- [`IndexerSelect.swift`](IndexerSelect.swift): heap-based top-k with
  deterministic ordering by decreasing score and increasing index on ties.

## Flow and dependencies

The kernels produce the indexer scores; when the CPU fallback is used, the
scores are read back and `IndexerSelect` returns the indices of the KV rows to
pass to sparse attention. The equivalent GPU path lives in the
[`Kernels/Attention`](../../../../Kernels/Attention/README.md) wrappers.

## Modification rules

Preserve the same result set and tie-breaking as the full sort, including NaN
and contexts shorter than k. Measure CPU complexity, GPU synchronization, and
readback cost separately.
