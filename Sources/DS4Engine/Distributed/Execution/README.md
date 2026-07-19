**English** | [Italiano](README.it.md)

# Distributed/Execution

## Architectural boundary

`DistEngine` and `ExpertShard` physically stay in this folder to avoid
breaking the public APIs and the distributed protocol v11, but the
implementation is DeepSeek-V4-specific. The geometry is not fixed: Flash uses
43 layers and 256 experts, Pro 61 layers and 384 experts. Both now go through
`RuntimeBackendFactory` before building tokenizer or decoder: a recognized
Qwen family is rejected as a backend not yet implemented.

Do not simply move these types under a Qwen backend: first you need a new
distributed capability and a handshake that includes architecture, model
fingerprint and activation geometry. Every wire-incompatible change
requires a new protocol; it is not negotiated between different builds.

The single-file Pro Q2 GGUF uses these paths. The two-file Pro Q4 package
remains download-only: the names `Layers00-30` and `Layers31-output` are no
substitute for a loader capable of assembling and validating multiple shards.

Adapts `DS4Metal` to distributed work units without including networking.

## Components

- `DistEngine.swift`: embedding, forward of single or batched slices, output
  head, tokenizer, sampling, KV shard and vertical path.
- `ExpertShard.swift`: `ExpertShardEngine`, which loads an expert mask and
  returns the partial sum requested by `DistExpertWork`.

## Dependencies and flow

Depends on `DS4Core`, `DS4Metal` and the types in [`Protocol`](../Protocol/README.md).
Coordinator and workers turn frames into calls to these engines; no type in
this folder opens sockets.

`DistEngine.chatPromptIds` applies the same control-token neutralization used
by local inference to turns, history and tool schemas, then leaves the
renderer as the sole party adding the trusted structural delimiters.

## Extension

Keep the numeric semantics of distributed execution here. Validate shapes,
layers, quantization and masks before GPU dispatch. New transport strategies
belong in [`Transport`](../Transport/README.md).
