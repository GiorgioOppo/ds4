**English** | [Italiano](README.it.md)

# Laguna S 2.1 model contract

This directory owns the portable Laguna S 2.1 model geometry and GGUF metadata
validation.  It has no Metal dependency and does not register or select a
runtime backend.

`LagunaConfiguration` accepts only `general.architecture = "laguna"` and the
exact 48-block Laguna S 2.1 shape implemented by the reference `laguna-s2.1`
branch of `antirez/ds4` (`DS4_SHAPE_LAGUNA_S21`): GQA with 8 KV heads and a
per-layer query-head alternation (48 full-attention heads on every fourth
block, 72 sliding-window heads elsewhere), YaRN RoPE with an independent
sliding-window frequency base, one leading dense block, and 256 routed experts
with 10 active plus one shared expert.  Unknown Laguna variants fail at load
instead of being run with incompatible dimensions.
