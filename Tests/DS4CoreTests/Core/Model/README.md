# Core Model Tests

- [`Common/`](Common/README.md) verifica identificatori, detection e disponibilità
  dei backend senza GGUF o GPU.
- [`Backends/DeepSeekV4/`](Backends/DeepSeekV4/README.md) verifica forma,
  configurazione e alias compatibili DeepSeek.

When adding a model-family invariant, cover both a valid configuration and the
specific invalid boundary it rejects.
