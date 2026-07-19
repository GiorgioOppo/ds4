**English** | [Italiano](README.it.md)

# Model/Backends/DeepSeekV4

Shape and metadata validation for DeepSeek V4 Flash/Pro.

- `DeepSeekV4Configuration.swift` contains `DeepSeekV4Shape`,
  `DeepSeekV4Configuration`, defaults and family-specific errors.
- The historical public aliases (`ModelShape`, `ModelConfig`, `ModelDefaults`,
  `ModelVariant`, `ModelConfigError`) remain available without changing the
  behavior of existing consumers.

The configuration checks `general.architecture` first; a Qwen GGUF is
recognized as a family but rejected because its backend is not yet
implemented. Old DeepSeek GGUFs without the general key continue to be
recognized through the `deepseek4.*` metadata.
