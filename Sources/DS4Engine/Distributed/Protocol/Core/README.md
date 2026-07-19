**English** | [Italiano](README.it.md)

# Protocol/Core

`Dist.swift` is the authoritative source of the global contract: magic, wire
version, maximum sizes, frame types, work flags and the knob whitelist.

## Flow and dependencies

All other codecs depend on `Dist`; coordinator, worker and transport use its
values without redefining them. It depends on Foundation and the `DS4Core`
utilities.

## Extension

Assign stable numeric values to new `MsgType`s, document the semantics in
[`../../PROTOCOLLO.md`](../../PROTOCOLLO.md) and bump `protocolVersion` if an
older node cannot correctly interpret the new flow. Limits on collections
coming from the network are mandatory.
