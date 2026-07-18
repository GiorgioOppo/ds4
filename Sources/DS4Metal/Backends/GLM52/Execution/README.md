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
