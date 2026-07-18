# Kernels/Compression

Wrappers for HyperConnections reduce/expand and the recurrent compression of
the NSA context.

## Main files

- [`MetalHCSplit.swift`](MetalHCSplit.swift): separation of the HC streams.
- [`MetalHyperConnections.swift`](MetalHyperConnections.swift): HC reduce/expand and fusions.
- [`MetalCompressor.swift`](MetalCompressor.swift): compressor projections and state.
- [`MetalKVCompress.swift`](MetalKVCompress.swift): writing of the compressed KV rows.

## Flow and dependencies

Before attention the residual state is reduced; compressor and gate aggregate
raw tokens into persistent NSA rows. After attention/FFN, the output is
expanded into the HyperConnection streams. The operations are composed by
[`Graph/Operations`](../../Graph/Operations/README.md).

## Modification rules

Counters, compression ratio and update order are recurrent state: a change
can invalidate KV snapshots and prefill/decode parity. A fusion must keep a
non-fused path for numerical verification.
