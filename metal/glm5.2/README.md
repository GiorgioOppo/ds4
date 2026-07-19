**English** | [Italiano](README.it.md)

# GLM 5.2 kernels

Metal kernel sources for the GLM 5.2 (`glm-dsa`) backend: sigmoid top-8 router,
compact KV-LoRA store/normalization, indexer key store, indexer scoring and the
staged compact-attention core (`qk_lowrank`, `attention_indexed` over the
selected cache rows, `value_project`). Each kernel has an isolated Swift
wrapper and a CPU oracle under `Sources/DS4Metal/Backends/GLM52/`; none is
wired to a decoder yet.

The routed expert stages now have validation kernels (fused gate/up SwiGLU
and down over Q2_K/Q4_K/Q5_K/Q6_K rows, one thread per output row with the
reference element pairing), judged against `GLM52FFNCPUReference` on
dequantized weights; the tuned per-quant families come later beside them.

Editing workflow and embedding (`make embed-kernels`) are documented in
[`../README.md`](../README.md).
