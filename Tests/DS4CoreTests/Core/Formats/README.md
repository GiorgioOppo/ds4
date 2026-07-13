# Format Tests

Tests for persistent and model-file formats owned by `DS4Core`:

- `GGUF/`: container metadata and tensor descriptors.
- `KVCheckpoint/`: KV checkpoint encoding and validation.
- `Quantization/`: numeric conversion and quantization helpers.

Format tests must include malformed/truncated inputs and preserve compatibility
with already-written data whenever the format is stable.

