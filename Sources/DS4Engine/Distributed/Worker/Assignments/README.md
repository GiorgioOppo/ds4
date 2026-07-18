# Worker/Assignments

Loads and publishes the responsibilities assigned to the worker.

## Files

- `DistWorker+Assignment.swift`: layer slice, output head, KV cache and knobs.
- `DistWorker+ExpertAssignment.swift`: vertical expert mask and its
  `ExpertShardEngine`.

## Flow and dependencies

The handler validates the payload, resolves the transferred files, applies
only whitelisted knobs and builds the engine outside the lock. The previous
state stays usable until the atomic commit of the new engine; `READY` is sent
at the end. Depends on [`Execution`](../../Execution/README.md) and
[`Protocol/Handshake`](../../Protocol/Handshake/README.md).

## Extension

Keep the claim, load and commit phases separate. Do not publish partial
assignments and do not reuse an engine if model, slice, context or numeric
options do not match.
