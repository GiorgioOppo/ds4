**English** | [Italiano](README.it.md)

# Distributed/Coordinator

`DistCoordinator` owns the cluster configuration, the local engine and the
worker connections. Responsibilities are split into extensions.

## Files

- `DistCoordinator.swift`: state, `Peer`, `Config` and active route.
- `+Connections`: partitioning, handshake, transfer and assignment.
- `+Files`: offer construction and file streaming.
- `+KV`: negotiation and saving of shard checkpoints.
- `+HorizontalChat`: pipeline over layer slices.
- `+VerticalChat` and `+ExpertParallelism`: local backbone and expert shards.
- `+Benchmark`: measurements for both topologies.

## Dependencies and flow

Uses [`Protocol`](../Protocol/README.md), [`Transport`](../Transport/README.md),
[`Execution`](../Execution/README.md) and [`Files`](../Files/README.md).
`connect` prepares the horizontal route; `connectVertical` prepares the expert
shards. Only after all `READY`s can a chat or a benchmark start.

## Extension

Keep the configuration immutable during a turn, associate every result with
the current session, and close connections/return listeners on every error
path. Scheduling and retries stay here, not in the protocol types.
