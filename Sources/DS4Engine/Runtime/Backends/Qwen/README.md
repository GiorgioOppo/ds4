**English** | [Italiano](README.it.md)

# Qwen Backend

Status: GGUF family recognized, backend not yet implemented.

Inspection returns the model's name, family and capabilities, but the runtime
capabilities remain empty and `BackendSelector` terminates with
`backend <architettura> non ancora implementato`. This happens before the
code attempts to read DeepSeek metadata, special tokens or tensors.

Enabling Qwen will require a Metal decoder, weight schema, KV, chat
template/tool codec, numeric tests and dedicated GGUF smoke tests; removing
the check from the factory is not enough.
