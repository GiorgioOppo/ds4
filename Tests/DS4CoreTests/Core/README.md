# Core Tests

Deterministic tests for the platform-independent `DS4Core` module. The child
areas cover conversation markup, file formats, generation, model metadata,
storage planning, and tokenization.

These tests should not require a Metal device, network access, or a production
GGUF. Prefer small in-memory fixtures and exact assertions. New `DS4Core`
behavior belongs in the matching child directory.

