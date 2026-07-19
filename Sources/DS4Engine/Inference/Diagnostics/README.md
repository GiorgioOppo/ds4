**English** | [Italiano](README.it.md)

# Inference/Diagnostics

Provides read-only diagnostics on the GGUF without starting the Metal decoder.

## Component

`Diagnostics.swift` opens metadata and tokenizer to:

- show token IDs and text;
- inspect `tokenizer.chat_template`;
- verify special tokens and DSML markup;
- compare full and compact tool declarations.

## Dependencies and flow

Depends on `DS4Core` and Foundation. Outputs are strings intended for GUI,
logs or tests; no service state is modified.

## Extension

Prefer deterministic, side-effect-free checks. A diagnostic that measures
the GPU belongs in [`Benchmark`](../Benchmark/README.md), not here.
