# Tokenization/Backends/DeepSeekV4

Byte-level BPE tokenizer with the JoyAI/DeepSeek pre-tokenizer, DeepSeek V4
special tokens and reasoning mode.

- `DeepSeekV4Tokenizer.swift` contains the concrete implementation and keeps
  the historical public alias `Tokenizer`.
- `ThinkMode.swift` contains the DeepSeek reasoning mode and prefix.

The byte-for-byte behavior and the existing APIs remain unchanged; the new
location prevents accidentally reusing this tokenizer for a Qwen GGUF.
