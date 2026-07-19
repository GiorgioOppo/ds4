**English** | [Italiano](INFERENZA-DISTRIBUITA.it.md)

# Distributed inference

DwarfStar supports two distinct topologies over the same native TCP protocol:
a horizontal pipeline over layer ranges and vertical expert parallelism. This
document describes the behavior implemented by protocol v11.

## When to use it

Distribution can reduce per-node SSD work or parallelize the expert gather,
but it adds latency and network traffic. Before configuring it:

- use identical builds on all Macs;
- prefer direct Ethernet or a Thunderbolt bridge;
- keep the network trusted: transport and server are cleartext;
- verify model and settings locally on each hardware class.

## Horizontal topology: layer pipeline

```text
coordinator -> worker 1 [layers 0...a]
            -> worker 2 [layers a+1...b]
            -> worker N [layers ...last + head]
            -> coordinator -> sampling
```

The coordinator owns rendering, tokenizer, embedding, sampling, tools and
conversation state. Each worker owns a contiguous range of layers, the
corresponding weights and the KV shard. The HC state traverses the workers for
every token or prefill chunk.

This mode reduces the experts read from each SSD to roughly a fraction of the
total layers. Per-token time, however, stays close to the sum of the workers'
times, plus transport. Worker-to-worker forwarding avoids some round trips
back to the coordinator, but requires a reachable return address.

## Vertical topology: expert parallelism

```text
                +-> worker 1: expert subset of all layers --+
local backbone -+-> worker 2: expert subset of all layers --+-> sum
                +-> worker N: expert subset of all layers --+
```

The coordinator runs the entire dense backbone: embedding, route/attention,
KV, compressors, shared FFN and output head. Each layer's routed experts — 256
for Flash or 384 for Pro — are partitioned across the workers with explicit
masks. For each routed layer the coordinator sends activation, ids and weights
only to the owners involved; the workers return a partial sum.

The expert gather can proceed in parallel across the SSDs, but the path
requires roughly one round-trip per routed layer. It is only suitable for
links with an RTT below about 1 ms. On Wi-Fi, network latency dominates by
construction.

Current status:

- `expertAssign`, `expertWork`, `expertSum` protocol active;
- workers with `ExpertShard` and bundle/cache filtered by the mask;
- local backbone wired up through the `remoteExperts` callback;
- vertical chat and dedicated benchmark available in the GUI;
- current partition is round-robin; balancing from the usage imatrix remains
  a possible improvement.

Performance details are in
[EXPERT_PARALLELISM.md](EXPERT_PARALLELISM.md).

## Connection cycle

1. The worker starts a listener without loading any model.
2. The coordinator opens the connection and validates magic and version.
3. It offers GGUF and sidecars with name, size and SHA-256.
4. The worker requests only missing files or suffixes.
5. The coordinator sends `ASSIGN` for a horizontal slice or
   `EXPERT_ASSIGN` for a vertical shard.
6. The worker applies the knob whitelist, inspects the GGUF geometry,
   validates the slice or the mask, loads the assigned engine and sends
   progress and `READY` with 43 layers for Flash or 61 for Pro.
7. The route becomes available only when all required peers are ready and
   coverage is valid.

Peer setup proceeds in parallel. A disconnection during a file transfer
preserves the `.part`; on reconnection the hash chain identifies the last
valid checkpoint and resumes from there.

## Protocol v11

Framing uses magic `DS4D`, little-endian headers and payloads with explicit
limits. The version must match exactly: there is no negotiation between
incompatible semantics.

`EXPERT_ASSIGN` carries a fixed-length mask: 32 bytes for Flash and 48 for
Pro. The decoder rejects wrong lengths, non-zero padding bits and truncated
payloads. An unassigned worker announces zero layers; after `ASSIGN`,
`READY` must match the geometry actually loaded.

Message families are separated by responsibility:

- handshake and assignment;
- pipeline work/result;
- KV checkpoints;
- file offer, request, chunk and confirmation;
- expert shard assignment and work;
- errors and progress.

The wire structures live in `Distributed/Protocol` and do not depend on
sockets, coordinator or worker. `DistTransport` handles connection and frames;
coordinator and worker apply the semantics.

## KV continuity

In the horizontal pipeline each worker saves only the layers it owns. The
coordinator can reuse an in-memory prefix or negotiate a restore from disk.
The restore is accepted only if all shards hold the same prefix; otherwise the
whole route restarts from a cold prefill.

Session id and `turnStart` prevent a result left in the socket after a stop
from being interpreted as the next turn's response.

The vertical topology keeps KV and recurrent state on the local backbone; the
expert workers are stateless with respect to the sequence and serve FFN
requests.

## File transfer

Workers use a managed store under Application Support. The protocol can
transfer:

- GGUF;
- expert bundles;
- dense Q4 caches;
- other sidecars declared in the manifest.

For Pro, the complete single-file Q2 GGUF is runnable. The Pro Q4 package
`Layers00-30`/`Layers31-output` can be transferred and downloaded, but it is
not a valid route: multi-shard assembly is not implemented.

Chunks are 4 MiB. Each file has a final SHA-256 and chained checkpoints every
256 MiB. A derived file is reused only if the manifest and sizes match.

## Configuration

| Setting | Pipeline | Vertical |
|---|---|---|
| `host:port` list | slice order | shard list |
| activation bits | 32/16/8 for HC state | 32/16 for activations and sums |
| prefill chunk | tokens per frame | local backbone prefill |
| forwarding | optional | not applicable |
| worker expert cache | slice cache | cache of owned experts |
| expert bundle | recommended | strongly recommended |

The coordinator propagates only `Dist.perfKnobKeys`. Arbitrary variables
cannot be set over the network. Options that change the numbers, such as dense
Q4, travel in typed fields together with the corresponding sidecar.

## Concurrency and failures

- A worker serves one route at a time through `DistGate`.
- Stop and cancellation close the turn at the next safe boundary.
- Version, payload or coverage errors are fatal for setup.
- Transport errors during setup can be retried.
- A missing vertical peer invalidates the expert partition.
- Benchmark and chat cannot use the same route/KV at the same time.

## Security

The protocol offers neither TLS nor authentication. Prompts, tokens and
activations travel in cleartext. Use only a trusted LAN or a protected tunnel
and do not expose the worker port to the Internet. See
[CRITTOGRAFIA.md](CRITTOGRAFIA.md).

## Code map

- `Sources/DS4Engine/Distributed/Protocol` — wire data and codecs.
- `Sources/DS4Engine/Distributed/Transport` — connections and framing.
- `Sources/DS4Engine/Distributed/Coordinator` — setup, files, KV and chat.
- `Sources/DS4Engine/Distributed/Worker` — lifecycle and serving.
- `Sources/DS4Engine/Distributed/Execution` — slice decoder and expert shard.
- `Sources/DwarfStar/Features/Distributed` — controllers and views.

## Recommended verification

1. round-trip tests of payloads and limits;
2. loopback connection with files already present;
3. interruption and resume of a partial file;
4. local/pipeline parity at 32 bits;
5. 16- vs 8-bit A/B for quality and network;
6. vertical benchmark only after measuring RTT and the local baseline.
7. for Pro, numerical parity and multi-Mac benchmarks on the real Q2 GGUF
   before considering the performance validation complete.

Do not compare topologies with different models, caches or warm-up.
