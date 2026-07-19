**English** | [Italiano](README.it.md)

# Tokenization API tests

The factory tests verify architecture-only policy and construction from tiny
synthetic GGUF tokenizer tables. Explicit Qwen and unknown architectures must
fail instead of falling back to DeepSeek; the legacy DeepSeek fallback applies
only when `general.architecture` is missing.

Conversation protocol selection is tested at the same frontend boundary and
does not change runtime availability.
