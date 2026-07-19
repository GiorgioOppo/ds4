**English** | [Italiano](README.it.md)

# Persistence

Contains application persistence independent of the GUI. It currently hosts
the KV checkpoint cache in [`KV`](KV/README.md).

## Flow and dependencies

`InferenceService` and the distributed workers request lookup, restore and
store; the `Persistence` folder defines the on-disk lifecycle, while snapshots
and layer imports are provided by `DS4Metal`.

## Extension

Every new store must declare key, format/version, budget, eviction strategy,
atomicity and behavior on corrupted files. Avoid dependencies on SwiftUI and
keep RAM bounded during large-data I/O.
