**English** | [Italiano](README.it.md)

# Model/Backends

Portable configurations specific to the supported architectures. Each backend
owns its own shape, metadata keys and validations; the shared contracts and
GGUF detection stay in [`../Common`](../Common/README.md).

- [`DeepSeekV4/`](DeepSeekV4/README.md): operational Flash/Pro profiles.
- [`GLM52/`](GLM52/README.md): validated GLM 5.2 contract, runtime under
  development.
- [`Qwen/`](Qwen/README.md): groundwork without a decoder.
