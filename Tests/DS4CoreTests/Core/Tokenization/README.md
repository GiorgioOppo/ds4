# Tokenization Tests

`TokenizerTests.swift` covers vocabulary loading, byte-level fallback, special
tokens, encode/decode round-trips, and think-mode behavior.

Add fixtures for Unicode, invalid byte sequences, adjacent special tokens, and
empty input when tokenizer behavior changes. Avoid assertions tied to a local
model path.

