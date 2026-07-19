**English** | [Italiano](README.it.md)

# Kernels/MoE

Wrappers for the router and the routed/shared feed-forwards of the
Mixture-of-Experts.

## Main files

- [`MetalRouter.swift`](MetalRouter.swift): logits, probabilities, top-k and
  router weights.
- [`MetalMoE.swift`](MetalMoE.swift): matvec for selected experts and reduction.
- [`MetalMoEFused.swift`](MetalMoEFused.swift): fused gate/up SwiGLU and
  down-sum.

## Flow and dependencies

The router produces ids and weights; the cache translates model ids into
resident slots or the loader gathers contiguous slabs. The MoE kernels apply
gate/up, activation and down, then sum the weighted contributions into the
residual state.

## Modification rules

Always distinguish expert id, slot id and index into the prefill union. Check
the active count, zero-weight padding and the gate/up/down layout. Fusions must
remain comparable with the three separate passes and must not alter the
selection.
