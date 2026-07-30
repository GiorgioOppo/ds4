**English** | [Italiano](README.it.md)

# Laguna kernel wrappers

`extension MetalRuntime` wrappers over the decode kernels in
`metal/laguna/`, in the GLM style: `[Float]` in, `[Float]` out,
one command buffer per call, every pipeline resolved by its exact Metal
function name. They exist so the kernels can be judged against the CPU
oracles in [`../Reference/`](../Reference/README.md) on hardware before any
graph work: per-head RMSNorm + NeoX RoPE for Q and K in one grid, the F16
ring KV store, and the gated GQA decode attention (short, wrapped and
split-reduction windows). The future resident graph encodes the same
pipelines without the per-call sync.
