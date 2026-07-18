# Expert parallelism — vertical split of the model

Status as of July 16, 2026: **phases A-C implemented in protocol v11**.
Payloads, expert shard workers, coordinator backbone, vertical chat and GUI
benchmark are active. Multi-Mac validation and phase D optimizations remain to
be expanded. Flash uses 256 experts; the full Pro Q2 GGUF uses 384 experts. The
two-file Pro Q4 package is not yet runnable.

Operational prerequisite: RTT below roughly 1 ms between nodes, via a
Thunderbolt bridge or direct Ethernet. With about 41 routed layers, a 7 ms RTT
adds nearly 300 ms per token before any useful work; on Wi-Fi this topology is
normally a losing proposition.

## Why a vertical split

The horizontal pipeline assigns layer ranges to workers. The SSDs work
sequentially along the token's path: the dominant time tends toward the sum
of the slice gathers.

In the vertical split, each layer's experts — 256 for Flash or 384 for
Pro — are distributed across the workers. When the router picks six experts,
each SSD reads its
own share in parallel and returns a partial sum. On the dominant cost the time
can approach the maximum across workers rather than the sum, but you pay one
network round-trip per routed layer.

## Implemented architecture

- **Coordinator/local backbone**: embedding, route/attention for all
  layers, KV, NSA compressors, shared FFN, HC reductions and output head.
- **Expert shard worker**: a subset of the experts valid for all layers;
  no KV or conversation state; gather, gate/up/down and weighted sum.
- **Protocol v11**: `expertAssign`, `expertWork`, `expertSum`, with geometry
  and mask length validated against the GGUF.
- **GUI**: Vertical split toggle, connection, chat and dedicated benchmark.

`StreamingDecoder` exposes a `remoteExperts` callback used in place of the
local routed FFN branch. The coordinator loads the backbone with the local
expert cache disabled; the expert shards use bundles, slot-cache and the
assigned knobs.

## Per-token flow and routed layers

1. The backbone computes route/attention and produces the ids and weights of
   the six experts.
2. The coordinator groups the ids by owning mask.
3. It sends `expertWork` in parallel with layer, sequence, ids, weights and
   activation.
4. Each worker validates that the ids are owned, runs the FFN and sends
   `expertSum`.
5. The coordinator sums the partials and continues with the layer's shared
   FFN/reduction.

Vertical activations and sums are transported at 32 or 16 bits. The 8-bit
option available for the horizontal pipeline is not used by the current
vertical path.

## Expert assignment

`DistExpertAssign.expertMask` is preceded on the wire by its own length and
contains one bit per expert: 32 bytes for Flash, 48 for Pro. Bit `e` indicates
ownership of expert `e`. The current partition is **round-robin**
(`e % workerCount`), with exact coverage and no overlaps. Length, padding
bits, coverage and uniqueness are rejected if they do not match the loaded
geometry.

Greedy balancing based on the usage imatrix is not active yet. It is a
possible phase D: it should distribute the observed load, not just the
number of experts, while keeping exact coverage and a reproducible
configuration.

## Communication costs

Each involved layer sends one activation and receives one sum of the model's
width. Total traffic per involved worker is roughly:

- Flash (4096 elements): 32 KiB at F32 or 16 KiB at F16;
- Pro (7168 elements): 56 KiB at F32 or 28 KiB at F16;
- in both cases headers, ids and weights are added on top.

Latency, more than bandwidth, is the constraint: the round-trip repeats along
the sequence of routed layers and therefore grows with the profile's geometry.

These are order-of-magnitude estimates. The benchmark must measure real
traffic and latency with the same GGUF and the same cache as the local
baseline.

## File transfer

Setup reuses the resumable transfer from protocol v8: GGUF and sidecars are
offered with SHA-256 and concatenated checkpoints. At the moment the worker
can receive the full bundle and open only the records needed by its own
mask.

A future transfer of a physically sharded bundle would reduce setup space and
time, but requires a new manifest type and a dedicated validation
strategy.

## Phase status

- **A — complete**: design, v11 frames, bound-checked encode/decode and
  round-trip tests.
- **B — complete**: `ExpertShard`, worker assignment, masks, bundle/cache and
  `expertWork` -> `expertSum` serving.
- **C — complete**: local backbone, remote scatter/gather, vertical chat,
  toggle and benchmark.
- **D — open**: balancing from the usage imatrix, additional overlap,
  possible shard files and a multi-Mac A/B campaign.

## Validation criteria

1. Measure RTT before connecting the vertical route.
2. Verify mask coverage and uniqueness.
3. Compare vertical F32 with the local six-expert path.
4. Measure F16 separately for quality and network.
5. Record prefill, steady-state decode, bytes/token, bandwidth and cache
   hit-rate.
6. Compare vertical, pipeline and local with identical model, prompt and
   warm-up.
7. Before certifying Pro, run numeric parity and multi-Mac benchmarks on the
   full Q2 GGUF; protocol tests alone do not measure logits or quality.

The design goal remains at least 1.5x over local at equal quality. If network
latency or imbalance cancels out the parallel gather, the mode must remain
optional.

## Limits and security

- the full backbone must fit on the coordinator according to the local profile;
- each worker receives model activations in the clear;
- losing a peer invalidates the experts it owns;
- chat and benchmark cannot share the route simultaneously;
- the two partial Pro Q4 GGUFs cannot be used as worker slices: the
  multi-shard loader is still missing;
- the protocol offers no authentication or TLS.

Use only a trusted network. For the full picture see
[INFERENZA-DISTRIBUITA.md](INFERENZA-DISTRIBUITA.md) and
[CRITTOGRAFIA.md](CRITTOGRAFIA.md).

## Code map

- `Sources/DS4Engine/Distributed/Protocol/Experts`
- `Sources/DS4Engine/Distributed/Coordinator/DistCoordinator+ExpertParallelism.swift`
- `Sources/DS4Engine/Distributed/Coordinator/DistCoordinator+VerticalChat.swift`
- `Sources/DS4Engine/Distributed/Execution/ExpertShard.swift`
- `Sources/DS4Engine/Distributed/Worker/Assignments/DistWorker+ExpertAssignment.swift`
- `Sources/DS4Metal/Backends/DeepSeekV4/Decode/Execution/StreamingDecoder.swift`
- `Sources/DS4Metal/Backends/DeepSeekV4/Decode/Execution/DecodeLayer.swift`
