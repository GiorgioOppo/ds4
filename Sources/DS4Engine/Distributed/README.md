# DS4Engine/Distributed

Distributed inference with pipeline parallelism across contiguous layer ranges,
modeled after `ds4_distributed.c`.

Each **worker** owns a slice of layers, including weights and its KV shard. The
**coordinator** owns embeddings, sampling, prompt rendering, and cluster control.
For each token, the HC state (`nHC x nEmbd` floats, transported as 32/16/8-bit
depending on configuration) moves through the worker chain.

- **`DistEngine.swift`** is the per-node engine. It exposes low-level slice
  operations (`embed`, `forwardSlice`, `head`) plus tokenizer/sampling utilities
  needed by the coordinator.
- **`DistCoordinator.swift`** connects workers, validates protocol version and
  contiguous layer coverage, runs multi-turn chat on the cluster, and exposes
  `benchmark()`. Every turn gets a fresh `session` id that workers echo in each
  RESULT: a reply left in a TCP buffer by a cancelled turn is discarded on the
  next read instead of being mistaken for the new turn's answer.
- **`DistWorker.swift`** implements the worker node: it listens for a coordinator
  and executes its assigned layer slice. Incoming WORK frames are validated
  (token count vs payload, layer bounds, position vs context) before touching
  the engine, and one TURN at a time is enforced at the session level — a
  competing coordinator gets an explicit ERROR frame instead of silently
  resetting the active turn's KV shard.
- **`DistProtocol.swift` / `DistTransport.swift`** define protocol frames and the
  async `NWConnection` transport. Decoding is STRICT: truncated activation
  payloads and hostile lengths (route cap) reject the frame rather than
  producing short arrays. The activation codec moves data as bulk buffer
  copies — it is the wire hot path. TCP traffic is plaintext and listeners are
  unauthenticated, so use it only on trusted networks.
