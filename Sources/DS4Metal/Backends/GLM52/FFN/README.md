**English** | [Italiano](README.it.md)

# GLM 5.2 FFN reference

CPU correctness oracles for the GLM feed-forward stages, ports of upstream's
F32 reference path (`layer_glm_dense_ffn_one_f32_ref`,
`layer_glm_routed_moe_one_f32_ref`, `layer_glm_shared_ffn_one_f32_ref`,
`output_logits_glm_one_f32_ref`): F32 activations against weights the caller
dequantized with the `Quantize` references — never the requantized-activation
fast path.

`GLM52FFNCPUReference` provides RMSNorm (Double square accumulation like
upstream), the stable silu/SwiGLU **without clamp** (GLM ships
`swiglu_clamp_exp = 0`), plain matvec, the dense block (hidden 12288), the
shared expert (hidden 2048), the routed experts — where the router weight
(already ×2.5-normalized by `GLM52RouterReference`) multiplies each expert's
SwiGLU mid BEFORE the down projection, upstream's exact association — the
routed+shared sum, and the output head (RMSNorm + vocab matvec, no softcap).

`GLM52FFNGeometry.v5_2` carries the architecture dimensions; tests use small
parametric geometries. These are validation fixtures for the future MoE/dense
Metal kernels, not a decode path.
