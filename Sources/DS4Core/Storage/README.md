# Storage

Portable working-set planning for when the model is larger than the
available RAM.

## Main files

- [`SSDCachePlan.swift`](SSDCachePlan.swift): computes the budget and number
  of storable experts and interprets the SSD-streaming-related arguments.
- [`SimulatedMemoryLock.swift`](SimulatedMemoryLock.swift): reserves and
  locks anonymous memory to test reduced-RAM scenarios.

## Flow and dependencies

The plan is built before the decoder and drives the size of the concrete
`DS4Metal` caches. The simulation is a diagnostic tool: it contains no
decoder policy and reads no weights. The runtime options are collected in
the [Configuration Reference](../../../README.md#configuration-reference).

## Modification rules

Use checked arithmetic for bytes and GiB, clearly distinguish estimates from
real allocations, and always release locked resources. Keep this layer free
of Metal dependencies.
