# Tokenization Tests

[`Backends/DeepSeekV4/`](Backends/DeepSeekV4/README.md) covers vocabulary
loading, byte-level fallback, special tokens, encode/decode round-trips and
compatibility of the historical tokenizer API.

[`Backends/GLM52/`](Backends/GLM52/README.md) covers the `glm4` pretokenizer,
BPE, real special-token IDs, stop policy, byte-level round-trips and the
per-architecture factory.

Add fixtures for Unicode, invalid byte sequences, adjacent special tokens, and
empty input when tokenizer behavior changes. Avoid assertions tied to a local
model path.
