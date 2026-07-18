# Inference/Service

Implements the central actor that loads the model and serializes every
operation on the decoder.

## Main files

- `InferenceService.swift`: dependencies, state, executor and initialization.
- `InferenceService+Conversation.swift`: rendering, turns and KV continuity.
- `InferenceService+Generation.swift`: prefill, sampling, streaming and
  checkpoints.
- `InferenceService+Agents.swift`: agent profile selection.

## Flow

The caller configures conversation and tools, then starts `send`,
`sendWithHistory` or `complete`. The service renders the tokens, reuses or
rebuilds the KV, runs prefill and decode, and returns an
`AsyncThrowingStream<GenEvent>`. The detailed flow is in
[`../FLUSSO-INFERENZA.md`](../FLUSSO-INFERENZA.md).

Before tokenizer, DeepSeek configuration and Metal, the initializer goes
through [`RuntimeBackendFactory`](../../Runtime/README.md). Qwen is recognized
but explicitly rejected as a backend not yet implemented; DeepSeek decode
keeps using the concrete `StreamingDecoder` type.

Before rendering, every field supplied by users, history, tools or API is
neutralized with respect to the GGUF's structural tokens. Only the framing
produced by `ChatRenderer` can therefore become BOS, a role or an atomic DSML
delimiter.

## Dependencies

Depends on `DS4Core`, `DS4Metal`, [`Persistence/KV`](../../Persistence/KV/README.md),
[`Agents`](../../Agents/README.md) and [`Tools`](../../Tools/README.md).

## Extension

Keep only the inference-loop responsibilities here. Benchmarks, tuning and
sub-agents have their own folders. Every new path must handle cancellation,
the context limit and `kvDirty` transitions.
