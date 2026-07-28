**English** | [Italiano](README.it.md)

# Laguna S 2.1 backend (staged)

Reserved space for the Laguna S 2.1 Metal backend. Today it contains only the
pieces that can be validated without Apple hardware:

- [`TensorSchema/`](TensorSchema/README.md): the exact GGUF tensor contract of
  the two published quantization recipes (Q8_0 signal path, legacy Q4_K/F16)
  and the mixed Q2_K/Q3_K routed-expert file, ported from
  `weights_validate_laguna_layout` in the reference `laguna-s2.1` branch,
  plus the payload-free `LagunaWeightMap` directory.
- [`Reference/`](Reference/README.md): scalar CPU oracles of the decode path
  (norm/RoPE, gated GQA attention, router, FFN blocks).
- [`Kernels/`](Kernels/README.md): validation wrappers over the ported
  `metal/laguna/` kernels, judged against the oracles on hardware.
- [`Engine/`](Engine/README.md): the single runtime enablement switch,
  currently off.

The decoder itself — `metal/laguna.metal`, the GQA/SWA attention graph, the
MoE gather and the optional DFlash speculative companion — is not ported yet;
see `docs/PORTING-GAPS.md` for the turnkey plan. Until the logits-parity gate
passes on hardware, a Laguna GGUF is recognized, validated and refused with an
explicit not-implemented error.
