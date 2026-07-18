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
