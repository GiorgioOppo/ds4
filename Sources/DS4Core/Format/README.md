# DS4Core/Format

Binary on-disk formats and numeric helpers.

- **`GGUF.swift`** parses and loads GGUF files. It mmaps the file once, accesses
  tensors by absolute offset without copying, and reads metadata. Public helpers
  include `mapBase`, `findTensor`, and `prefetch` (`madvise(WILLNEED)`).
- **`KVCFile.swift`** stores disk-backed KV cache checkpoints, ported from
  `ds4_kvstore.c`. It defines the checkpoint header, eviction score, and SHA-1
  naming convention.
- **`Half.swift`** provides portable f32/f16 conversions, including a software
  path for architecture-safe behavior.
