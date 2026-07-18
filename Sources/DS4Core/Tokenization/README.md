# Tokenization

Shared contract and backend-specific tokenization implementations.

## Structure

- [`API/`](API/README.md): minimal `TokenizerProtocol` surface.
- [`Common/`](Common/README.md): reusable byte-level primitives.
- [`Backends/DeepSeekV4/`](Backends/DeepSeekV4/README.md): JoyAI/DeepSeek BPE,
  special tokens and reasoning; keeps the historical `Tokenizer` alias.
- [`Backends/GLM52/`](Backends/GLM52/README.md): GPT-2 byte-level BPE,
  `glm4` pretokenizer, GLM controls, and stop policy.
- [`Backends/Qwen/`](Backends/Qwen/README.md): extension point, with no
  placeholder implementation.

## Flow

The concrete tokenizers read their tables through
[`GGUFModel`](../Formats/GGUF/README.md).
Plain text goes through pre-tokenization and BPE merges; an already-rendered
chat first recognizes special tokens indexed by initial byte. The generated
ids feed prefill and decode; outgoing ids are reassembled into bytes/text.

`neutralizeSpecialTokenLiterals(in:)` contains the prompt boundaries: it
breaks up literal sequences that the model classifies as control tokens when
they come from untrusted data. It must be applied to individual fields before
rendering (system, user, history, tool results, and schemas), never to the
already-rendered chat: BOS, roles, and delimiters added by the renderer must
remain atomic tokens. The `inJSON:` variant decodes and recursively visits
keys and values before reserializing: even a delimiter hidden behind `\uXXXX`
escapes is neutralized before the schema or argument renderer expands it.

## Modification rules

Byte-for-byte correspondence with the selected tokenizer is mandatory.
Preserve longest-match for special tokens and the single-byte fallback; test
Unicode, CJK, DSML delimiters, and malformed input. Avoid allocations in the
hot loop without measuring the impact on prefill.
