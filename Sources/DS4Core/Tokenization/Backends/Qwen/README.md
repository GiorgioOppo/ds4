# Tokenization/Backends/Qwen

Extension point reserved for the future Qwen tokenizer.

Do not automatically reuse the JoyAI/DeepSeek pre-tokenizer: vocabulary,
merges, regex/pre-tokenizer, special tokens and detokenization must be read
and verified against the chosen Qwen GGUF. No implementation is exposed until
those invariants are covered by parity tests.
