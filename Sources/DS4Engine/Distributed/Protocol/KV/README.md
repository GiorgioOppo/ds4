# Protocol/KV

`DistKV.swift` serializes the KV continuity messages for each slice.

## Operations

- encode/decode of the token lists for query and restore;
- list of the available prefix lengths;
- save request with cold/continued indication;
- ack with outcome and diagnostic message.

## Flow and dependencies

The coordinator intersects the lengths returned by all workers and picks a
common prefix; each worker forwards the operation to its own `DistEngine` and
[`DiskKVStore`](../../../Persistence/KV/README.md).

## Extension

A restore is valid only if all shards confirm the same frontier. Enforce
limits on the token count and do not treat a partial ack as global success.
