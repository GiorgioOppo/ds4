# Protocol/Framing

`DistFrameHeader.swift` defines the envelope of every message: magic, type
and payload byte count.

## Flow and dependencies

[`Transport`](../../Transport/README.md) reads the header first, validates
magic and type and then requests exactly the declared length. Serialization
uses the primitives in [`Serialization`](../Serialization/README.md).

## Extension

The header must stay small and deterministic. A layout change is always
incompatible and requires a version bump plus tests for empty, truncated,
oversized and wrong-magic frames.
