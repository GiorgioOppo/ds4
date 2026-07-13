# Generation Tests

`SamplerTests.swift` validates greedy and stochastic sampling controls,
temperature, top-k/top-p/min-p filtering, and repetition penalty behavior.

Seed randomized tests so failures are reproducible. Test filter composition and
degenerate distributions explicitly; do not depend on statistical luck.

