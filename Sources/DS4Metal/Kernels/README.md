# DS4Metal/Kernels

One Swift wrapper per Metal kernel. Each wrapper prepares arguments, selects the
pipeline, and dispatches work. The **kernel sources** live in `metal/*.metal`
and are the source of truth; they are embedded into the binary by
`make embed-kernels`, which regenerates `Runtime/KernelSources.swift`.

Main groups:

- **MoE:** `MetalMoE`, `MetalMoEFused` for fused SwiGLU-pair expert matvec plus
  down-sum6, `MetalRouter`, `MetalSparseSelect`, and `MetalArgsort`.
- **Attention:** `MetalFlashAttn`, `MetalAttnOutLow`, `MetalRoPE`,
  `MetalKVCompress`, `MetalCompressor`, `MetalIndexerScore`, and
  `MetalIndexerPool`.
- **Algebra/utility:** `MetalDense`, `MetalMatmulMM`, `MetalNorm`,
  `MetalSoftmax`, `MetalSumRows`, `MetalGLU`, `MetalUnary`,
  `MetalGetRows`/`MetalSetRows`, `MetalCopy`, `MetalConcat`, `MetalRepeat`,
  `MetalBin`, `MetalHCSplit`, and `MetalHyperConnections`.

When adding or changing a kernel, edit `metal/*.metal`, run
`make embed-kernels`, then update or add the matching Swift wrapper here.
