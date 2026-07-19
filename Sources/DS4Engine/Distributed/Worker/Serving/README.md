**English** | [Italiano](README.it.md)

# Worker/Serving

The main dispatcher for frames received from a coordinator connection.

## Flow

`DistWorker+Serving.swift` sends `HELLO`, then handles file offer/receipt,
assignments, KV commands, expert requests and `WORK`. For the horizontal
pipeline it runs the slice under `DistGate`, forwards the state to the next
worker or returns the result to the return listener.

## Dependencies

Composes all the worker areas and uses [`Transport`](../../Transport/README.md)
and [`Protocol`](../../Protocol/README.md). It does not implement its own wire
codecs.

## Extension

Every new `MsgType` must have a lifecycle state in which it is allowed and a
deterministic error response. Validate session, assignment, slice,
position, context and shape before touching the engine.
