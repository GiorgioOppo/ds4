**English** | [Italiano](README.it.md)

# KV Checkpoint Tests

`KVCFileTests.swift` covers checkpoint headers, metadata, tensor payloads,
round-trips, and validation for `KVCFile`.

Use temporary files owned by the test and remove them in teardown. Changes to
the serialized layout require both a new round-trip test and a compatibility or
clear rejection test for the previous representation.

