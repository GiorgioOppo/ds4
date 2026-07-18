# GLM 5.2 layer reference

CPU oracle of one full GLM layer and of the first-token forward chain, the
port of upstream's F32 reference path (`layer_glm_first_token_one_f32_ref`,
`forward_glm_first_token_cpu_f32_ref`): single token at position 0, no cache,
where attention over one visible row collapses to the token's own value
projection (softmax of one score is 1 — the Q path is never evaluated).

`GLM52LayerCPUReference` composes the pinned FFN/router oracles into the
pre-norm residual structure (`afterAttn = x + attn(x)`,
`out = afterAttn + ffn(rmsNorm(afterAttn))`), dense for the leading blocks and
sparse with the integrated router elsewhere. Sparse layers fetch ONLY the
router-selected experts through a provider closure — mirroring streaming,
where unselected experts are never read. `firstTokenForward` chains layers;
the output head stays `GLM52FFNCPUReference.outputHead`.

This is the independent baseline for the roadmap's tensor-by-tensor
comparison (step 4): the future GPU graph must match it layer by layer,
embedding to logits. It is deliberately not a decode path.

# GLM 5.2 decode reference

`GLM52DecodeCPUReference` is the CPU oracle of one DECODE step — the port of
upstream's `glm_graph_forward_token` wiring on the indexed-attention path,
composed from the pinned primitive oracles. The binding order it encodes:

- `attn_norm = rmsNorm(cur)` feeds BOTH LoRA down-projections (`attn_q_a`,
  `attn_kv_a_mqa`); the caches are written BEFORE selection and attention
  (upstream's fused store). The compact row stores the normalized 512-wide
  KV-LoRA prefix plus the RAW 64-wide K-RoPE tail — no norm, no rotation;
- the indexer key (full-indexer layers only) projects the RAW residual, takes
  centered LayerNorm (eps `1e-6`, weight+bias), PREFIX RoPE with the token
  position, and lands in the key cache — also before selection;
- the query tail RoPE (last 64 of each 256-wide head) precedes `qk_lowrank`;
  the indexer query from `q_a_norm` rotates its PREFIX instead. The indexer
  head weights come raw (no softmax) from `indexer.proj` of the RAW residual;
- `visible = position + 1`: the current token always participates. When
  `visible <= topK` the selection is the causal fill range; otherwise scores
  over all visible rows feed the lowest-index-tie top-k. IndexShare layers
  reuse the latest full-indexer selection verbatim (absolute rows, no
  offset) and never own indexer keys;
- attention consumes the raw cached tails and rotates each selected row with
  the ROW's own absolute position at attention time
  (`GLM52AttentionCPUReference` with `rotateTailByRowPosition`).

Cache stores round through IEEE binary16 because the real caches are F16 —
upstream numerics, not a GPU artifact. `decodeLayer` shares the residual FFN
stage with the first-token oracle (`ffnStage`). Geometry is parameterized so
tests can shrink every width; `GLM52DecodeGeometry.v5_2` pins the real
architecture (Q-LoRA 2048, nope 192, indexer 32x128 rot 64, top-2048).
