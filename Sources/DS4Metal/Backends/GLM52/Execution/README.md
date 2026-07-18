# GLM 5.2 layer execution

First GPU composition of a GLM layer: the first-token forward with every
quantized matvec dispatched through the validated kernels (Q8_0 attention
projections and dense/shared blocks, K-quant routed experts, GPU router),
judged against `GLM52LayerCPUReference` on the dequantized weights.

Deliberate validation-level split: the cheap glue — RMSNorm over one
embedding row, residual adds, the tiny F32 router matvec (the GGUF stores
`ffn_gate_inp` in F32) — stays on the CPU oracle implementations, so every
divergence from the oracle is attributable to a GPU kernel. Experts arrive
through a provider closure (slot cache, payload reader or fixture bytes).

`glm52FirstTokenForward` chains layers and finishes with the output head.
This is the correctness baseline for the persistent-GPUTensor graph (resident
activations, compact cache, prefill/decode), which comes next — it is not a
decode loop and enables no catalog capability.

# GLM 5.2 decode execution

`glm52DecodeAttention`/`glm52DecodeLayer` compose one DECODE step on GPU over
the validated primitives, judged against `GLM52DecodeCPUReference`: Q8_0
matvecs for every projection, GPU KV-LoRA norm feeding the F16 compact-cache
store (normalized prefix, RAW tail — written BEFORE selection and attention),
the GPU indexer key store (LayerNorm + prefix RoPE), tail/prefix RoPE
kernels, GPU indexer scoring plus multi-block top-k when the visible range
exceeds top-k (causal fill range otherwise, current token always included),
and the indexed attention with per-row tail rotation
(`rotateTailByRowPosition`). IndexShare layers reuse the preceding
full-indexer selection verbatim and never store indexer keys.

Same split as the first-token executor: RMSNorm glue, residual adds and the
tiny F32 router/indexer-proj matvecs stay on the CPU oracle implementations
so every divergence is attributable to a GPU kernel. The attention and
indexer kernels fix the v5_2 head geometry; embedding width, Q-LoRA rank,
FFN widths and top-k stay free for affordable fixtures. The FFN half is the
shared `glm52LayerFFNStage`. Host arrays move through shared buffers per
dispatch — the persistent-GPUTensor decode graph that keeps activations and
caches resident comes next, with this composition as its correctness
baseline. Still not a runnable decode loop; no catalog capability changes.
