**English** | [Italiano](README.it.md)

# Distributed/Protocol

Collects all the types serialized on the wire. These structures open no
sockets and know nothing about coordinator or worker state.

## Areas

- [`Core`](Core/README.md): version, message types, flags and limits.
- [`Framing`](Framing/README.md): common frame header.
- [`Serialization`](Serialization/README.md): little-endian primitives.
- [`Handshake`](Handshake/README.md): identity and assignment.
- [`Files`](Files/README.md): artifact offer and transfer.
- [`KV`](KV/README.md): shard checkpoint control.
- [`Work`](Work/README.md): hidden states, routes and results.
- [`Experts`](Experts/README.md): vertical MoE parallelism.
- [`Codec`](Codec/README.md): activation compression.

The complete sequence is in [`../PROTOCOLLO.md`](../PROTOCOLLO.md).

## Rules

Every `encoded()` must have a symmetric `decode` that checks all limits
before constructing the value. Do not use `MemoryLayout` as the wire format:
fields, order and endianness must be explicit. An incompatible change
requires bumping the version in `Core/Dist.swift`.
