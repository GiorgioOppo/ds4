**English** | [Italiano](README.it.md)

# Protocol/Codec

`ActivationCodec.swift` converts `Float` vectors into 32-, 16- or 8-bit
payloads and reconstructs them on reception. It is the hot path of inference
traffic.

## Flow and dependencies

The [`Work`](../Work/README.md) and [`Experts`](../Experts/README.md) messages
use the codec to reduce bandwidth and copies. The 8-bit format includes the
data needed for dequantization; the decoder always receives the expected
count.

## Extension

Optimize with bulk copies and contiguous buffers, preserving round-trip and
length checking. A new precision changes the wire format and requires
numerical tests with a declared tolerance in addition to the protocol bump.
