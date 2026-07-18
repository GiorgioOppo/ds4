# Distributed

Implements inference across multiple Macs with two strategies:

- **horizontal pipeline**: each worker owns a contiguous range of layers;
- **vertical expert parallelism**: the coordinator runs the backbone and
  aggregates MoE contributions produced by remote shards.

The current wire protocol is `Dist.protocolVersion = 11` and requires strict
equality between nodes.

The geometry is read from the GGUF: the pipeline covers 43 layers/256 experts
for Flash and 61 layers/384 experts for Pro. The distributed runtime accepts
the complete Pro Q2 GGUF; the two files of the Pro Q4 package are not runnable
slices and remain download-only until a multi-shard loader exists.

## Structure

- [`Protocol`](Protocol/README.md): framing and serializable messages.
- [`Transport`](Transport/README.md): TCP connections based on Network.framework.
- [`Coordinator`](Coordinator/README.md): topology, chat, KV and file distribution.
- [`Worker`](Worker/README.md): listener, assignments and request execution.
- [`Execution`](Execution/README.md): decoder adapters for slices and shards.
- [`Files`](Files/README.md): hashes, manifests and the local model archive.

The wire format and sequences are described in
[`PROTOCOLLO.md`](PROTOCOLLO.md).

## Flow at a glance

1. The coordinator connects and verifies `HELLO` and the version.
2. It offers GGUF and sidecars; the worker requests only missing or
   incomplete files.
3. The coordinator sends `ASSIGN`; the worker loads its responsibility and
   replies `READY`.
4. `WORK` messages traverse the route and produce `RESULT`, or the expert
   messages produce partial sums.
5. At the end of a turn the nodes can save KV checkpoints of their
   respective parts.

## Security and constraints

Traffic is plain TCP and the listener does not authenticate peers: use only
trusted networks. Lengths, routes, slices, sessions and payloads must be
validated before allocating memory or invoking Metal. Only the `DS4_*` knobs
present in the whitelist may traverse `ASSIGN`.

## Extension

An incompatible wire change requires bumping `protocolVersion`, symmetric
codecs and tests with truncated/hostile payloads. The transport must not
contain scheduling logic; messages must not depend on the coordinator
or the worker.
