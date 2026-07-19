**English** | [Italiano](README.it.md)

# Engine Persistence Tests

`DiskKVStoreTests.swift` validates KV lookup, serialization, indexing,
streaming, invalidation, and budget behavior.

Use a fresh temporary directory per test and clean it up. Include corrupt,
partial, incompatible, and eviction cases when the on-disk representation or
index policy changes.

