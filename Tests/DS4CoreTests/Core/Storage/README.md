**English** | [Italiano](README.it.md)

# Core Storage Tests

`SSDCachePlanTests.swift` checks deterministic storage/cache planning, budgets,
alignment, and boundary behavior without performing production-size I/O.

Use synthetic sizes and make byte-unit assumptions explicit. Performance
benchmarks belong in diagnostics, not in these unit tests.

