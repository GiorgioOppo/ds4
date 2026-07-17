# GLM 5.2 tensor-schema tests

Synthetic tensor-directory records exercise all layer boundaries and supported
routed quantizations without creating or mapping a full-size GGUF payload.

The stream-planner tests cover exact IQ2_XXS, Q2_K and Q4_K gate/up block sizes,
Q6_K down projections, rank-ordered top-8 selection, duplicate and out-of-range
IDs, invalid MoE layers, mismatched geometry, truncated directory sizes and
offset overflow. Weight-map tests also pin canonical typed GGUF names and the
payload-free descriptor contract.

`GLM52RealHeaderIntegrationTests` is an optional contract check against an
exact-size sparse copy of a real file. Set `DS4_GLM52_SPARSE_GGUF` to validate
metadata, tokenizer, all typed weight lookups and an expert read plan without
reading weight payloads.
