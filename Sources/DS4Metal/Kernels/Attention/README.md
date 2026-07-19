**English** | [Italiano](README.it.md)

# Kernels/Attention

Wrappers for standard and NSA sparse attention operations.

## Main files

- [`MetalFlashAttn.swift`](MetalFlashAttn.swift): attention over the KV context.
- [`MetalAttnOutLow.swift`](MetalAttnOutLow.swift): low-rank output projection.
- [`MetalRoPE.swift`](MetalRoPE.swift): rotation/inverse of the RoPE components.
- [`MetalIndexerScore.swift`](MetalIndexerScore.swift): relevance scores for the compressed rows.
- [`MetalIndexerPool.swift`](MetalIndexerPool.swift): indexer pooling.
- [`MetalSparseSelect.swift`](MetalSparseSelect.swift): on-device top-k selection.

## Flow and dependencies

The decoder projects query/KV, applies RoPE and updates the caches. The
indexer scores the compressed rows; a GPU selection or the CPU fallback
decides the subset read by flash attention, then the output projection
returns to the residual space.

## Modification rules

Respect strides, head counts, the raw window and the compressed-row counters.
GPU and CPU top-k tie-breaks must match. Verify empty and partial contexts,
ring buffer wrap and unaligned mmap offsets.
