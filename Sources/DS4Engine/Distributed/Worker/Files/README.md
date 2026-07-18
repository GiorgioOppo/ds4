# Worker/Files

Receives large files incrementally and resumably.

## Component and flow

`DistWorker+Files.swift` manages the `.part` file, rebuilds the hash up to the
agreed offset, verifies the chain's checkpoints, appends sequential chunks and
promotes the file into the `DistFileStore` only after the final SHA-256. A
disconnection suspends the file without deleting the valid prefix.

## Dependencies

Uses CryptoKit, [`Distributed/Files`](../../Files/README.md) and the messages
in [`Protocol/Files`](../../Protocol/Files/README.md).

## Extension

Reject non-monotonic offsets, unsanitized names, out-of-index chunks and
sizes exceeding the manifest. Do not load the entire artifact into RAM and do
not record an unverified file in the manifest.
