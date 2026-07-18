# Model/Backends

Configurazioni portabili specifiche delle architetture supportate. Ogni backend
possiede forma, chiavi metadata e validazioni proprie; i contratti condivisi e
il rilevamento GGUF restano in [`../Common`](../Common/README.md).

- [`DeepSeekV4/`](DeepSeekV4/README.md): profili Flash/Pro operativi.
- [`GLM52/`](GLM52/README.md): contratto GLM 5.2 validato, runtime in sviluppo.
- [`Qwen/`](Qwen/README.md): predisposizione senza decoder.
