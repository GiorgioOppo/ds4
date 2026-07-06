# DS4Metal/Decode

The decode graph and KV cache: the core of per-token inference.

- **`StreamingDecoder.swift`** is the orchestrator. It implements `forward`
  for one-token decode, chunked layer-major `prefill`, distributed slices, KV
  allocation, `exportKV`/`importKV`, and hooks for prefetching and the expert
  slot cache.
- **`DecodeLayer.swift`** runs one layer: HC-reduce, attention over the raw SWA
  window plus compressed NSA rows, top-K indexing, router execution, and KV
  writes.
- **`Graph.swift` / `GraphContext.swift` / `GraphCompressor.swift`** encode
  command buffers and track recurrent compressor state.
- **`GraphMoEMM.swift`** batches the prefill routed FFN through the
  `mul_mm_id` matrix-matrix kernels (opt-in via `DS4_PREFILL_MM`) instead of
  one matvec per token.
- **`IndexerSelect.swift`** is the heap-based top-K used by the NSA indexer
  mask: same selected set as a full (score DESC, index ASC) sort at
  O(n log k) — it runs per ratio-4 layer per token, with n growing with the
  context.
- **`KVSnapshot.swift`** captures CPU-side KV state, including the raw window and
  compressor state. It is used by disk KV and sub-agent context switching.
- **`ExpertSlotCache.swift` / `ExpertUsage.swift`** implement the LRU pool of
  resident experts and the usage imatrix that records routing statistics.
- **`DSV4Decoder.swift`** is a reference decoder with dense attention for parity
  tests.

Key semantic: raw KV is a **sliding window of `nSWA` tokens** (`DSV4Dims.nSWA`).
Older context survives only through compressed rows.

Runtime knobs (`DS4_PREFILL_MM`, `DS4_FUSED_MOE`, `DS4_ASYNC_FFN`, …) are
documented in the root README's
[Configuration Reference](../../../README.md#configuration-reference).
