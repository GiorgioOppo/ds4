# Worker/KV

Handles the query, restore and save frames for the shard's KV cache.

## Component and flow

`DistWorker+KV.swift` decodes the request via
[`Protocol/KV`](../../Protocol/KV/README.md), delegates to `DistEngine` and
replies with lengths or acks. Checkpoints are separated by model and layer
range, so a shard cannot restore data from another assignment.

## Extension

Run import/export under the compute gate, keep the persistent I/O streaming,
and return an explicit failure when the engine is not ready or the prefix
does not match exactly.
