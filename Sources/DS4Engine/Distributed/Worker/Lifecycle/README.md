**English** | [Italiano](README.it.md)

# Worker/Lifecycle

Owns the listener's lifecycle and the worker's session state.

## Component

`DistWorker+Lifecycle.swift` creates the shard directory, persists usage,
starts/stops `NWListener`, accepts connections, builds `HELLO` and resolves
the local or transferred model.

## Flow and dependencies

`start` opens the listener; each connection is handed to
[`Serving`](../Serving/README.md). `admit` prevents concurrent turns from
resetting the active KV. `stop` closes listener, tasks and persistent
resources.

## Extension

Start and stop must be idempotent. Do not keep a session after terminal
errors, and do not use a received model path without falling back to the
sanitized file in the managed archive.
