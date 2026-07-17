# GLM 5.2 kernels

Metal kernel sources for the GLM 5.2 (`glm-dsa`) backend: sigmoid top-8 router,
compact KV-LoRA store/normalization, indexer key store, indexer scoring and the
staged compact-attention core (`qk_lowrank`, `attention_indexed` over the
selected cache rows, `value_project`). Each kernel has an isolated Swift
wrapper and a CPU oracle under `Sources/DS4Metal/Backends/GLM52/`; none is
wired to a decoder yet.

MoE expert kernels are added here only after their CPU/reference fixtures
exist — the attention fixtures live in
`Sources/DS4Metal/Backends/GLM52/DSA/GLM52AttentionReference.swift`.

Editing workflow and embedding (`make embed-kernels`) are documented in
[`../README.md`](../README.md).
