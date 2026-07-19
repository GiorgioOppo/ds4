**English** | [Italiano](README.it.md)

# Inference

This area exposes the application API and coordinates the complete inference
cycle. The `InferenceService` actor is the sole owner of the mutable state of
the decoder, the conversation and KV continuity.

## Structure

- [`API`](API/README.md): public DTOs, sampling parameters and events.
- [`Service`](Service/README.md): loading, conversation, prefill and decode.
- [`Benchmark`](Benchmark/README.md): measurement and warm-up.
- [`Diagnostics`](Diagnostics/README.md): tokenizer and template inspection.
- [`Subagents`](Subagents/README.md): isolated contexts for delegated work.
- [`Tuning`](Tuning/README.md): expert usage profile and cache metrics.

The step-by-step description is in [FLUSSO-INFERENZA.md](FLUSSO-INFERENZA.md).

## Dependencies

`DS4Core` provides tokenizer, rendering, tool calls and sampling; `DS4Metal`
provides the runtime and `StreamingDecoder`. KV persistence is encapsulated in
[`Persistence/KV`](../Persistence/KV/README.md).

## Extension rules

- Add a public DTO in `API` only if it must cross the service
  boundary.
- Add a cohesive responsibility as an `InferenceService` extension
  in the relevant folder.
- Do not access the decoder outside the actor's isolation.
- An interruption must leave `kvDirty` consistent, so the next turn can
  rebuild the state without producing numerically corrupted output.
