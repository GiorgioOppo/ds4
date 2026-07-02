# DS4Core/Format

Formati binari su disco e helper numerici.

- **`GGUF.swift`** — parser/loader GGUF: `mmap` del file una volta, accesso ai tensori per offset assoluto (no-copy), lettura metadati. Espone `mapBase`, `findTensor`, `prefetch` (madvise WILLNEED).
- **`KVCFile.swift`** — formato file della KV cache su disco (port di `ds4_kvstore.c`): header, eviction score, naming SHA-1.
- **`Half.swift`** — conversioni f32↔f16 portabili (anche software, arch-safe).
