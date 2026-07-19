**English** | [Italiano](README.it.md)

# DS4Core

Portable foundation of the project, written in Swift and free of Metal
dependencies. It exposes formats, conversation models, tokenization, sampling
and shared types used by `DS4Metal` and `DS4Engine`.

## Structure

- [`Conversation/`](Conversation/README.md): common types and per-backend chat formats.
- [`Diagnostics/`](Diagnostics/README.md): thread-safe load progress.
- [`Formats/`](Formats/README.md): GGUF, KV checkpoints and quantization primitives.
- [`Generation/`](Generation/README.md): next-token selection.
- [`Model/`](Model/README.md): architecture detection and per-backend configurations.
- [`Storage/`](Storage/README.md): SSD cache planning and RAM simulation.
- [`Tokenization/`](Tokenization/README.md): common API and per-backend tokenizers.

## Dependencies and flow

`DS4Core` depends only on the standard library and Foundation. The typical
flow is: open the GGUF -> build the tokenizer -> render the conversation ->
tokenize -> sample the logits produced by the backend. GPU operations and
decoder execution belong to `DS4Metal`.

## Modification rules

- Do not introduce Metal imports or types into this target.
- Keep parsers, rendering and sampler deterministic, with parity tests.
- Preserve binary compatibility of persisted formats.
- Place every new responsibility in its domain subfolder and update the
  corresponding README.
