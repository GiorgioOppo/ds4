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
- **`DistCoordinator.swift`** connects workers, validates contiguous layer
  coverage, runs multi-turn chat on the cluster, and exposes `benchmark()`.
- **`DistWorker.swift`** implements the worker node: it listens for a coordinator
  and executes its assigned layer slice.
- **`DistProtocol.swift` / `DistTransport.swift`** define protocol frames and the
  async `NWConnection` transport. TCP traffic is plaintext, so use it only on
  trusted networks.
