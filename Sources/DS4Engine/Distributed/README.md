# DS4Engine/Distributed

Inferenza distribuita per parallelismo di pipeline su range di layer contigui (modellata su `ds4_distributed.c`). Ogni **worker** possiede uno slice di layer (pesi + shard KV); il **coordinatore** possiede embedding, sampling e prompt. Lo stato HC (`nHC×nEmbd` float, trasportato a 32/16/8 bit) attraversa i worker per token.

- **`DistEngine.swift`** — engine per-nodo: espone le slice-op di basso livello (embed/forwardSlice/head) + tokenizer/sampling per il coordinatore.
- **`DistCoordinator.swift`** — connette i worker, valida la copertura contigua dei layer, esegue una chat multi-turno sul cluster; include `benchmark()`.
- **`DistWorker.swift`** — il nodo worker (ascolta il coordinatore, esegue il suo slice).
- **`DistProtocol.swift` / `DistTransport.swift`** — frame del protocollo e connessione async su `NWConnection` (TCP, in chiaro: usare su reti fidate).
