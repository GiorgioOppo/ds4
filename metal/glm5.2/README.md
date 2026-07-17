# GLM 5.2 kernels

Metal kernel sources for the GLM 5.2 (`glm-dsa`) backend: sigmoid top-8 router,
compact KV-LoRA store/normalization, indexer key store and indexer scoring.
Each kernel has an isolated Swift wrapper and a CPU oracle under
`Sources/DS4Metal/Backends/GLM52/`; none is wired to a decoder yet.

Compact DSA attention and MoE kernels are added here only after their
CPU/reference fixtures exist (see
`Sources/DS4Metal/Backends/GLM52/DSA/GLM52AttentionReference.swift`).

Editing workflow and embedding (`make embed-kernels`) are documented in
[`../README.md`](../README.md).
