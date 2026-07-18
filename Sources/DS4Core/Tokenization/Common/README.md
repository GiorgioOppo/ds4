# Tokenization/Common

Reusable primitives that do not select a template or an architecture.

- `ByteLevel.swift` implements the GPT-2 byte/codepoint map and the UTF-8
  utilities used by the DeepSeek BPE; a future tokenizer can reuse them when
  compatible.
