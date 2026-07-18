# Core Model Tests

- [`Common/`](Common/README.md) verifies backend identifiers, detection and
  availability without GGUF or GPU.
- [`Backends/DeepSeekV4/`](Backends/DeepSeekV4/README.md) verifies
  DeepSeek-compatible shapes, configuration and aliases.

When adding a model-family invariant, cover both a valid configuration and the
specific invalid boundary it rejects.
