# DS4Metal/Decode

Il grafo di decode e la KV cache: il cuore dell'inferenza per-token.

- **`StreamingDecoder.swift`** — l'orchestratore: `forward` (decode 1 token), `prefill` (a chunk, layer-major), slice distribuiti; alloca la KV cache; espone `exportKV`/`importKV` e l'hook di prefetch e di slot-cache esperti.
- **`DecodeLayer.swift`** — un layer: HC-reduce, attenzione (finestra SWA raw + righe compresse NSA, indexer top-K), router, scrittura KV.
- **`Graph.swift` / `GraphContext.swift` / `GraphCompressor.swift`** — encoder dei command buffer e stato del compressore ricorrente.
- **`KVSnapshot.swift`** — snapshot CPU della KV (finestra raw + stato compressore) per disk-KV e per il context-switch dei sub-agent.
- **`ExpertSlotCache.swift` / `ExpertUsage.swift`** — pool LRU degli esperti residenti + "usage imatrix" (statistiche di routing).
- **`DSV4Decoder.swift`** — decoder di riferimento (attenzione densa) per parità/test.

Semantica chiave: la raw KV è una **finestra scorrevole di `nSWA`** (vedi `DSV4Shape.nSWA`); il contesto vecchio vive solo nelle righe compresse.
