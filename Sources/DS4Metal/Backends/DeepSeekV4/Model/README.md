**English** | [Italiano](README.it.md)

# DeepSeekV4/Model

Loading of DeepSeek V4 weights from GGUF into GPUTensor: per-tensor primitives
and per-layer/output assembly used by both the all-resident path and
streaming.

## Main files

- [`GGUFWeights.swift`](GGUFWeights.swift): per-tensor loader (copy, no-copy
  mmap view), `LayerWeights` assembly (`layer`, `layerMappedExperts`,
  `layerMappedDense`, `layerSmallSkeleton`), output head, detection and
  validation of routed quantizations (`detectMoEQuant`,
  `validateRuntimeLayout`), gather/copy of the selected experts, and
  `pread`/`madvise` primitives shared by the streamers.

## Flow

`validateRuntimeLayout` closes the contract at bind time: allowed quant types,
consistent gate/up, shapes conforming to the geometry, and a hash table
present where required. Only then are the `LayerWeights` assembled; routed
experts may remain dummies (on-demand gather), mmap views of the whole tensor,
or cache slots depending on the calling factory.

## Change rules

Reads must never exceed the limits declared by the GGUF descriptor. Every
`LayerWeights` builder must go through `setExpertQuant` (per-layer
mixed-precision quant). The `pread` primitives remain thread-safe on a shared
fd (explicit offsets); `madvise` hints are advisory and cannot change the
numerics.
