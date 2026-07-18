# Protocol/Experts

Defines the frames of the vertical MoE parallelism, extended by protocol v11
for the Pro geometry.

## Types

- `DistExpertAssign`: model, fixed-length expert mask, cache, usage and
  knobs. The mask is 32 bytes for Flash and 48 for Pro.
- `DistExpertWork`: sequence, layer, selected experts, weights and activation.
- `DistExpertSum`: sequence, layer and the shard's partial sum.

## Flow and dependencies

The coordinator partitions the experts, assigns the masks and sends one
request per routed layer. [`ExpertShardEngine`](../../Execution/README.md)
computes the local contribution; the coordinator validates sequence/layer and
aggregates.

## Extension

Masks must be disjoint or have an explicit aggregation policy. Exact length,
padding bits, ID/weight counts, and activation precision and size must be
validated before running the kernel.
