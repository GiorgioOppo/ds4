**English** | [Italiano](README.it.md)

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

# GLM 5.2 resident decode graph

`GLM52ResidentDecodeWeights` uploads a layer's quantized weights ONCE into
MTLBuffers; `GLM52ResidentDecodeCaches` keeps the compact cache (interleaved
`[capacity][576]` F16 rows) and the indexer key plane resident on the GPU,
appended in place. `glm52ResidentDecodeAttention` encodes the whole attention
step on chained buffers: in the fill-range path a SINGLE command buffer
covers the norms (new `kernel_glm52_rms_norm_f32`), the LoRA projections,
the compact-row store (new `kernel_glm52_store_compact_row_f16`), both RoPE
kernels, the indexer key store, qk_lowrank, the rotated indexed attention
and the output projection; the top-k path splits only around the score
readback that feeds the host-orchestrated multi-block top-k. The CPU keeps
the residual adds, the router and the 32-row F32 `indexer.proj` matvec.

Correctness anchor: parity with `glm52DecodeAttention`, itself judged by
`GLM52DecodeCPUReference`; the one intentional arithmetic difference is the
float-reduction GPU RMSNorm replacing the Double-accumulation CPU glue.

The stack level completes the resident story: `GLM52ResidentFFN` uploads
each layer's FFN norm plus dense/shared weights once (routed experts remain
a per-token byte stream through the provider — inherent to streaming, not a
residency gap), `glm52ResidentDecodeLayer` runs the residual FFN half on
resident buffers with GPU accumulation (`kernel_glm52_add_f32`), tapping
`ffnIn` back to the host once per sparse layer for the F32 router, and
`glm52ResidentDecodeForward` chains a stack of `GLM52ResidentStackLayer`
values under the REAL IndexShare policy (absolute layer indices; full
layers publish the selection, intermediate layers must match their policy
source and reuse it verbatim), finishing in the resident output head
(`GLM52ResidentOutputHead`: final RMSNorm + vocabulary matvec). Judged
against the per-dispatch composition. Prefill on real prompts and the
real-GGUF logits parity are the remaining gates.
