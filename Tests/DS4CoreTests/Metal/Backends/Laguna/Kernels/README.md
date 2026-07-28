**English** | [Italiano](README.it.md)

# Laguna kernel tests

GPU/CPU parity for the Laguna decode kernels, skipped where Metal is
unavailable: per-head norm/RoPE on both block kinds (YaRN and plain) at
several positions, F16 ring-store bit equality, and gated GQA decode
attention on short, wrapped and split-reduction (>256 keys) windows — all
judged against the `Reference/` oracles with the production 128-dimension
heads.
