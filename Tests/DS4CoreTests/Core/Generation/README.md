**English** | [Italiano](README.it.md)

# Generation Tests

`SamplerTests.swift` validates greedy and stochastic sampling controls,
temperature, top-k/top-p/min-p filtering, and repetition penalty behavior.
It also cross-checks the DS4_FAST_SAMPLER threshold-collected full-vocabulary
path against the historical full build (same token, same RNG stream) over a
grid of parameters and logit shapes.

Seed randomized tests so failures are reproducible. Test filter composition and
degenerate distributions explicitly; do not depend on statistical luck.

