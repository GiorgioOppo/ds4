[English](README.md) | **Italiano**

# Riferimento FFN GLM 5.2

Oracoli di correttezza su CPU per gli stadi feed-forward di GLM, port del
percorso di riferimento F32 di upstream (`layer_glm_dense_ffn_one_f32_ref`,
`layer_glm_routed_moe_one_f32_ref`, `layer_glm_shared_ffn_one_f32_ref`,
`output_logits_glm_one_f32_ref`): attivazioni F32 contro pesi dequantizzati
dal chiamante con i riferimenti `Quantize` — mai il fast path con attivazioni
riquantizzate.

`GLM52FFNCPUReference` fornisce la RMSNorm (accumulo dei quadrati in Double
come upstream), la silu/SwiGLU stabile **senza clamp** (GLM fornisce
`swiglu_clamp_exp = 0`), il matvec semplice, il blocco denso (hidden 12288),
l'esperto condiviso (hidden 2048), gli esperti instradati — dove il peso del
router (già normalizzato ×2.5 da `GLM52RouterReference`) moltiplica il mid
SwiGLU di ogni esperto PRIMA della proiezione down, l'esatta associazione di
upstream — la somma routed+shared e l'output head (RMSNorm + matvec del
vocabolario, senza softcap).

`GLM52FFNGeometry.v5_2` trasporta le dimensioni dell'architettura; i test
usano piccole geometrie parametriche. Si tratta di fixture di validazione per
i futuri kernel Metal MoE/densi, non di un percorso di decode.
