**English** | [Italiano](README.it.md)

# DeepSeekV4/Decode/Cache

Resident cache of MoE experts and routing-frequency history.

## Main files

- [`ExpertSlotCache.swift`](ExpertSlotCache.swift): per-layer LRU pool,
  concurrent filling, speculative prefetch and protection of in-flight GPU
  reads.
- [`ExpertUsage.swift`](ExpertUsage.swift): thread-safe statistics, warm set,
  adaptive slot allocation and JSON persistence. In mixed-quant models the
  allocator works on a byte budget: the real per-layer record cost prevents
  Q4 pools from silently growing RAM. Without a complete history it uses a
  byte-balanced plan; hash layers with exact look-ahead may stay at the floor
  to leave budget for the non-predictable routers.

## Flow

The router selects the experts; `acquire` translates the IDs into slots,
serves the hits and fills the misses from the GGUF or the expert bundle. The
look-ahead prepares the next layer without blocking demand. The actual
selections update the statistics, reused in later sessions for warm-up.
The slot plan is frozen once per cache generation. Diagnostics read the pools
actually materialized; they do not recompute a fresh allocation using the
statistics mutated during generation.

## Change rules

Concurrency is serialized per layer and the global state has a separate lock:
do not invert this order. A slot read by a command buffer cannot be evicted
while it is in-flight. Preserve the `k+2` floor and validate the loaded IDs.
