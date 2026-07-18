# Diagnostics

Portable primitives for reporting the state of long-running operations.

## Main files

- [`LoadProgress.swift`](LoadProgress.swift): thread-safe singleton that
  publishes stage fraction and description during GGUF opening, cache
  preparation and weight loading.

## Flow and dependencies

The producer calls `reset`, `set`, `begin` and `advance`; the UI or service
periodically reads `snapshot`. Synchronization uses `NSLock`, with no SwiftUI
or Metal dependencies.

## Modification rules

Writes can come from concurrent workers: do not expose mutable state directly
and keep `snapshot` cheap. New structured metrics must remain UI-independent.
