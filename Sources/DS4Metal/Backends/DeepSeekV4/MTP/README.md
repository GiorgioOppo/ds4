# DeepSeekV4/MTP (Multi-Token Prediction)

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
