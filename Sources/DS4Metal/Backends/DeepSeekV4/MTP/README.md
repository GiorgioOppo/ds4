**English** | [Italiano](README.it.md)

# DeepSeekV4/MTP and DSpark

Components for speculative decode with the DeepSeek MTP head. The sidecar
has the internal accessory id `mtp`, but is excluded from the GUI catalog of
main models. Plan and status in `docs/SELF-SPECULATIVE.md`, section "Phase M".

- `MTPSidecar.swift` — Phase M1: opening the sidecar GGUF, classifying the
  tensors into interface roles (eh_proj, embed_tokens, enorm, hnorm,
  shared_head.*) and a validation report against the main model's
  dimensions. Metadata only: no GPU buffers, no effect on decode.
  Exposed in the demo via `DS4_MTP_GGUF` (explicit path, or `=1` to
  look for `*MTP*.gguf` next to the model).

Phase M2 (resident loading + draft forward) must be wired up ONLY after
reading the M1 report on the real sidecar: names, shapes and quants of the
MTP transformer block's tensors determine the wiring, and guessing them
produces silent garbage.

`DSparkSupportModel`, kept in the same already-compiled source file, handles
the newer multi-stage DSpark artifact. It validates all 81 tensor roles,
metadata, quant classes and shapes against the active Flash geometry. The demo
opens it with `DS4_DSPARK_GGUF=<path>` or `=1`; automatic lookup never mixes
the 0730 and 0731 checkpoints. `DSparkStage0Runtime` then captures the
mean-reduced HC state at every support-declared target layer and runs the
resident `main_proj` + `main_norm` stage on Metal after decode and prefill.
All three transformers are already bound with their official types: large
matrices and expert arrays remain mmap no-copy views, while norms/scales/biases
are copied resident. Prefill retains batched target hiddens and each stage owns
a private KV ring. On first use, captured target rows are tiled into those
rings; the Metal forward then evaluates `[target + 5 draft]` through HC,
batched projections, non-causal attention and routed IQ2_XXS/Q2_K MoE.
`DSparkGreedyVerifier` plus `dsparkVerifyAndCommit` implement target-side
proposal verification. A full accept retains the already-computed verifier
frontier; a partial accept rolls back and exactly replays its accepted prefix.
The last-stage HC collapse, shared Q8 output head, sequential Markov correction,
stable confidence gate and device-side argmax now produce up to five draft
tokens. The engine and demo feed that proposal to the verifier and commit only
the exact greedy prefix. DSpark is intentionally active only for temperature 0
with no repetition penalty; stochastic sampling keeps the ordinary path.
`DS4_DSPARK_EXACT_REPLAY=1` forces replay after full accepts for strict parity
diagnostics, at the cost of removing the speculative speedup.
