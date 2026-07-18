# Worker/Concurrency

`DistGate.swift` defines the actor that serializes computation on the
worker's decoder, even when multiple TCP connections are served in parallel.

## Flow and dependencies

[`Serving`](../Serving/README.md) runs the Metal closures through the gate.
Session ownership is checked separately by the lifecycle; the gate protects
execution, not frame semantics.

## Extension

Do not put network waits inside `run`. A future concurrent scheduler must
prove that decoder, KV cache and scratch buffers are independent before
allowing multiple simultaneous jobs.
