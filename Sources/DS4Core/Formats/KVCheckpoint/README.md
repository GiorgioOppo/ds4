# Formats/KVCheckpoint

On-disk format for saving and reusing the decoder's KV state.

## Main files

- [`KVCFile.swift`](KVCFile.swift): header, extension flags, SHA-1 naming,
  eviction score and payload encoding/decoding.
- `DSV4PayloadHeader`: describes the DeepSeek-V4-specific snapshot shape.

## Flow and dependencies

The persistence layer converts a Metal `KVSnapshot` into payload and header,
writes the checkpoint and later validates it before restore. Only the binary
contract is defined here: cache policy and orchestrated I/O belong to
`DS4Engine`.

## Modification rules

Keep magic, version, flags and field order compatible. Verify shape and size
before allocating or decoding. A payload extension must use a flag/version
recognizable by older readers.
