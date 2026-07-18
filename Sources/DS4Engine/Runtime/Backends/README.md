# Runtime/Backends

Each folder registers the capabilities and selection policy of one family.
Decoders and tensors stay in their respective DS4Metal backends; only the
bridge used by DS4Engine is kept here.

- [`DeepSeekV4/`](DeepSeekV4/README.md): operational backend.
- [`GLM52/`](GLM52/README.md): model definition/capabilities with the runtime
  still deliberately empty.
- [`Qwen/`](Qwen/README.md): family recognized but not available.

A new backend must add, together: detection, configuration, tokenizer and
chat format, decoder, diagnostics, numeric tests and UI capabilities.
Recognizing an architecture's name is not the same as declaring it runnable.
