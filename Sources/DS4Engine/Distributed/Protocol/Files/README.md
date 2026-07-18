# Protocol/Files

Defines the resumable transfer of GGUFs and sidecars.

## Types

- `DistFileEntry`: kind, name, size, SHA-256 and checkpoint chain.
- `DistFileOffer`: manifest proposed by the coordinator.
- `DistFileNeed`: missing indices and validated resume offsets.
- `DistFileChunk` / `DistFileDone`: sequential content and end of file.

`FILE_ACK` uses the ack format shared with the control flow.

## Flow and dependencies

Metadata is built in [`../../Files`](../../Files/README.md), sent by the
[`Coordinator`](../../Coordinator/README.md) and verified in
[`Worker/Files`](../../Worker/Files/README.md).

## Extension

Bound the number of entries, the chunk size and the name/hash lengths. A new
`Kind` must specify whether it is mandatory, how its path is computed and how
it is activated in the assignment.
