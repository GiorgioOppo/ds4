# Metal Model Tests

- `GGUFLoaderTests.swift`: model metadata and loader validation.
- `GGUFWeightMapTests.swift`: tensor-name-to-weight mapping and shapes.
- `MetalIOCircuitBreakerTests.swift`: fallback decisions for slow or failed
  Metal I/O windows.

Use synthetic GGUF fixtures where possible. Tests must not require the user's
downloaded model, and circuit-breaker tests should inject timings rather than
depend on current SSD speed.

