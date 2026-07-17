# GLM 5.2 FFN reference tests

Device-free suites for the FFN oracles. The matvec is reimplemented naively in
the tests; silu and RMSNorm are pinned by closed-form checks (extremes,
uniform input, identity rows — exact) and then reused inside composed
expectations, so the composed suites (dense, routed, sparse, output head)
prove compositional properties — operation order, weight placement, sums —
with small tolerances for summation-order drift.

The routed suite proves the router weight multiplies the SwiGLU mid before the
down projection and that `GLM52RouterReference` weights (already ×2.5) enter
unchanged; the sparse suite proves routed+shared is a plain sum. Rejection
tests cover wrong dimensions and non-finite inputs.
