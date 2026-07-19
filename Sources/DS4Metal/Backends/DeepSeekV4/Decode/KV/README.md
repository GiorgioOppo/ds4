**English** | [Italiano](README.it.md)

# DeepSeekV4/Decode/KV

Export and restore of the decoder's recurrent state.

## Main files

- [`KVSnapshot.swift`](KVSnapshot.swift): `CompSnapshot`, `KVLayerSnapshot` and
  `KVSnapshot`, plus the `StreamingDecoder` extensions for capture/restore.
- `KVSnapshotError`: reports shape incompatibilities during restore.

## Flow

The decoder copies the raw KV window, compressed rows and indexer state into a
CPU snapshot. `DS4Engine` can keep it in memory or encode it with
[`KVCFile`](../../../../../DS4Core/Formats/KVCheckpoint/README.md), then restore it
into a decoder with the same architecture.

## Modification rules

A snapshot must be self-consistent and independent of temporary buffers.
Validate layers, dimensions, counters and capacity before writing to the GPU.
Update snapshot and KVC format together when the persisted state changes.
