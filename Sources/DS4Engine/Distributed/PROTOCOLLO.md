**English** | [Italiano](PROTOCOLLO.it.md)

# Distributed protocol v11

## Frame

Every frame contains a `DistFrameHeader` followed by a payload: magic `DS4D`,
message type and length, all little-endian. The decoder rejects a wrong magic,
types unknown to the active path, truncated payloads and counts beyond the
limits.

## Connection and assignment

```text
coordinator                 worker
     | ------- TCP ----------> |
     | <------ HELLO ---------- |
     | ------ FILE_OFFER -----> |
     | <----- FILE_NEED ------- |
     | -- FILE_CHUNK/DONE ----> |
     | <------ FILE_ACK -------- |
     | ------- ASSIGN --------> |
     | <------- READY ---------- |
```

`HELLO` identifies the version and current assignment. `FILE_NEED` includes
resume offsets validated through a per-block SHA-256 chain; the complete file
is promoted only after the final hash is verified. `ASSIGN` defines model,
context, slice, cache, sidecars and the allowed performance knobs.

## Horizontal pipeline

The coordinator creates one session per turn, produces the hidden state and
sends a `DistWork`. The message carries the absolute position, the chunk's
tokens, slice, route and activation bits. Each worker applies its own layers
and forwards the state; the terminal node returns the hidden state or logits
to the return listener. A result with a stale session is discarded.

## Vertical parallelism

`EXPERT_ASSIGN` distributes disjoint expert masks. The payload encodes
first the `UInt32` length and then the mask bytes: 32 bytes for the 256
Flash experts, 48 bytes for the 384 Pro experts. Length, padding bits and
coverage are validated against the GGUF geometry. For each MoE layer
the coordinator sends `EXPERT_WORK` with IDs, weights and activation; each
shard replies with `EXPERT_SUM`. The validated partial sums are aggregated
into the local backbone.

Horizontal assignments are validated against the layer count actually
inspected: 43 for Flash or 61 for Pro. An idle worker announces zero layers
until `ASSIGN` completes; `READY` must then report the loaded geometry.

## KV continuity

`KV_QUERY`/`KV_LENGTHS` look for lengths common to all shards;
`KV_RESTORE` restores exactly the agreed prefix;
`KV_SAVE`/`KV_ACK` save the clean turn. If even a single worker cannot
restore the chosen length, the coordinator falls back to a cold prefill.

## Compatibility

The protocol does not negotiate features across different versions: a
mismatch aborts the setup. Every new field must have explicit limits, a
deterministic encoding, decode that fails atomically and round-trip tests
plus truncated cases.

See [`Protocol`](Protocol/README.md), [`Coordinator`](Coordinator/README.md)
and [`Worker`](Worker/README.md).
