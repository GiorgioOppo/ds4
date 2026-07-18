# Distributed/Worker

`DistWorker` is an initially idle node that listens on a port, receives files
and loads the work assigned by the coordinator.

## Structure

- `DistWorker.swift`: configuration and shared state protected by a lock.
- [`Lifecycle`](Lifecycle/README.md): listener, start/stop and `HELLO`.
- [`Assignments`](Assignments/README.md): loading slices or expert shards.
- [`Files`](Files/README.md): resumable reception and verification.
- [`KV`](KV/README.md): checkpoint commands.
- [`Serving`](Serving/README.md): frame dispatch and pipeline.
- [`Concurrency`](Concurrency/README.md): serialization of Metal work.

## Flow

Each connection is served in a task, but `DistGate` allows only one compute
job at a time. The assignment can be reused if it matches; otherwise
the worker loads the new engine outside the lock and publishes the state only
once loading is complete.

## Extension

Validate frame and session before touching the engine. Do not hold locks
during I/O or long loads. Every persistent resource must be identified by
model and node responsibility to avoid reuse across incompatible shards.
