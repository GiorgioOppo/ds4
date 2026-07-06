# DS4Engine/Service

Inference service and persistence.

- **`InferenceService.swift`** is the central actor. It owns the
  `StreamingDecoder`; exposes `send`, `provideToolResults`, and `complete` as
  event streams; manages append-only multi-turn KV reuse through `committedIds`;
  runs benchmarks; switches agent profiles and usage imatrices; and runs
  **sub-agents** through `runSubAgent`, snapshotting/restoring the main KV around
  an isolated context.
- **`DiskKVStore.swift`** implements disk-backed KV cache in the `ds4_kvstore`
  style: prefix checkpoints, cold restore, and budget-aware eviction. It is also
  used for content-keyed sub-agent KV caches. Both directions are RAM-bounded:
  restore streams the checkpoint one layer at a time into the decoder (each
  batch freed right after import — peak = one layer, not the whole file), and
  store writes from a uniquely-owned `SnapshotBox` whose layers are dropped as
  they reach the disk. Both fds use F_NOCACHE so checkpoint bytes never
  displace the hot page cache.
- **`ExpertBundleTool.swift`** builds/verifies the expert-bundle sidecar for a
  GGUF on demand (the Settings button in the GUI) without loading the engine:
  it opens the model (mmap + metadata only) and calls `ExpertBundle.openOrBuild`
  with the same paths as the load (sibling read → `DS4_BUNDLE_DIR`).
- **`Diagnostics.swift`** dumps tokens and chat-template rendering through the
  native tokenizer, without subprocesses.

`InferenceService` is intentionally central today. The sub-agent implementation
is a good future candidate for a `SubAgent.swift` extension once the current
`private` boundaries are loosened.
