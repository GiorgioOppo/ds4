**English** | [Italiano](README.it.md)

# Laguna CPU references

Scalar, Metal-free oracles for the Laguna S 2.1 decode path, ported from
`laguna_graph_forward_token` in `ds4.c` and `metal/laguna.metal` on the
reference `laguna-s2.1` branch. They pin the family's semantics — per-head
RMSNorm + NeoX rotary pairs with YaRN only on full-attention blocks, the
per-head softplus attention gate, the F16 ring KV cache, clampless SwiGLU and
the GLM-shared sigmoid router with 10 active experts — so the Metal graph and
kernels have a stable correctness boundary that runs in plain unit tests,
exactly like the GLM `Reference/` oracles.
