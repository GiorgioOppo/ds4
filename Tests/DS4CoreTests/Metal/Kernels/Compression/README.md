# Compression Kernel Tests

Tests for compressor projections and Hyper-Connection split/reduce primitives:
`MetalCompressorTests`, `MetalHCSplitTests`, and
`MetalHyperConnectionsTests`.

Validate both shapes and numerical output, including fused versus reference
paths where available. Keep test tensors small enough for fast local runs.

