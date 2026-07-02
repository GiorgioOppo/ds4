# DS4Engine/Service

Inference service and persistence.

- **`InferenceService.swift`** is the central actor. It owns the
  `StreamingDecoder`; exposes `send`, `provideToolResults`, and `complete` as
  event streams; manages append-only multi-turn KV reuse through `committedIds`;
  runs benchmarks; switches agent profiles and usage imatrices; and runs
  **sub-agents** through `runSubAgent`, snapshotting/restoring the main KV around
  an isolated context.
- **`DiskKVStore.swift`** implements disk-backed KV checkpoints in the
  `ds4_kvstore` style: prefix checkpoints, cold restore, and budget-aware
  eviction. It does not replace the live KV/compressor buffers used during
  prefill/decode. It is also used for content-keyed sub-agent KV caches.
- **`Diagnostics.swift`** dumps tokens and chat-template rendering through the
  native tokenizer, without subprocesses.

`InferenceService` is intentionally central today. The sub-agent implementation
is a good future candidate for a `SubAgent.swift` extension once the current
`private` boundaries are loosened.
