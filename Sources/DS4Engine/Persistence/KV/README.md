# Persistence/KV

`DiskKVStore` keeps decoder checkpoints indexed by the exact token prefix,
allowing chat, stateless API, sub-agents and workers to avoid a full prefill.

## Files

- `DiskKVStore.swift`: configuration, budget and state.
- `+Index`: entry scanning, hits and eviction strategy.
- `+Lookup`: longest-prefix search and restore.
- `+Store`: snapshot, atomic write and budget maintenance.
- `+Streaming`: import/export one layer at a time.
- `+Serialization`: binary body primitives.

The layout is documented in [`FORMATO-CHECKPOINT.md`](FORMATO-CHECKPOINT.md).

## Flow and dependencies

The lookup compares model and tokens before the restore. Import and store
release each layer after use, limiting peak RAM; `F_NOCACHE` prevents the
checkpoints from evicting the hot weights out of the page cache. Depends on
`DS4Core` for the headers and on `DS4Metal` for `KVSnapshot`.

## Extension

Preserve temporary write plus rename, full validation before import and a
budget in both bytes and tokens. An incompatible format must be versioned and
old files must fail safely.
