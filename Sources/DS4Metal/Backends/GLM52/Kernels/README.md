# GLM 5.2 kernels

Swift wrappers and scalar correctness oracles for kernels in
`metal/glm5.2/glm52.metal` live here. Pipelines are requested lazily from
`MetalRuntime`; merely inspecting or downloading a GLM model does not compile a
separate pipeline set or allocate GPU buffers.

Every new kernel requires a deterministic CPU comparison before it may be used
by the GLM graph. Tests must include tie behavior and edge values where those
affect greedy output.

The currently validated atomic boundaries are:

- `GLM52Router`: sigmoid-plus-bias top-8 routing with unbiased normalized
  weights;
- `GLM52CompactKV`: placement and binary16 conversion of cache-ready 576-wide
  rows into separate 512-wide KV-LoRA and 64-wide K-RoPE planes;
- `GLM52KVLoRANorm`: RMSNorm of the 512-wide KV-LoRA prefix with epsilon
  `1e-5`, preserving the raw 64-wide K-RoPE payload in F32;
- `GLM52IndexerKeyStore`: upstream-compatible centered LayerNorm plus affine
  weight/bias, RoPE over `0..<64`, and capacity-checked F16 cache placement;
- `GLM52IndexerScores`: causal, token-major scoring of 32x128 queries against a
  shared binary16 indexer-key cache using per-head ReLU and weights. Its default
  scale is the architecture value `1/sqrt(32*128) = 1/64`;
- `GLM52CompactAttention`: the staged decode attention core over the compact
  cache — `qk_lowrank` (absorb the 192-wide nope query into `attn_k_b`),
  `attention_indexed` (softmax over the selected rows, accumulated in the
  KV-LoRA domain, selection capped at the architecture's top-2048) and
  `value_project` (`attn_v_b`), each dispatched in isolation plus a chained
  validation path compared against `GLM52AttentionCPUReference`. The two
  projections also exist as Q8_0 variants reading the GGUF weight bytes
  directly (34-byte blocks, scale outside the int8 product like upstream's
  `dot_q8_0_row_f32_ref`); their baseline is the F32 oracle on the
  dequantized weights;
- `GLM52IndexerTopK`: multi-block descending top-k over token-major score rows
  (the `ds4_gpu_indexer_topk_tensor` dispatch: per-block bitonic argsort, then
  iterative binary-search merges), reusing the vendored DeepSeek argsort
  kernels. Causal future rows arrive as `-INFINITY` scores and sink to the
  end; ties follow the bitonic network, not the oracle's lowest-index rule;
- `GLM52MoE`: validation kernels for the routed expert stages — fused
  gate/up SwiGLU (route weight on the mid, before down) and the down
  projection — reading Q2_K/Q4_K/Q5_K/Q6_K rows exactly as stored in the
  GGUF with one thread per output row and the reference element pairing.
  Baseline: `GLM52FFNCPUReference` on the dequantized weights. The tuned
  per-quant families (slots/addr/masked batches) come later beside these.

The compact-store input is intentionally *cache-ready*: its first 512 values
have already passed through `GLM52KVLoRANorm` and its final 64 values are the
untouched K-RoPE payload. The GGUF name `indexer.k_norm` is potentially
misleading: the reference kernel subtracts the mean, so its contract is
LayerNorm rather than RMSNorm. Likewise, the GLM indexer rotates the first 64
columns; this deliberately follows upstream even though one generic query-side
helper calls its operation `rope_tail`.

These wrappers allocate shared buffers for deterministic tests; they do not
make the GLM backend runnable.
