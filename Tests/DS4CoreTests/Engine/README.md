**English** | [Italiano](README.it.md)

# Engine Tests

Tests for orchestration and integration services in `DS4Engine`: agent
profiles, benchmark aggregation, diagnostics, distributed protocol, model
management, persistence, projects, and tools.

Prefer injected transports, temporary directories, and local fixtures. Unit
tests must not require credentials, internet access, the user's projects, or a
loaded production model.

The [`Benchmark`](Benchmark/README.md) tests exercise deterministic aggregation
of next-token accuracy observations. They intentionally do not load a GGUF or
run Metal: end-to-end model evaluation belongs to a reproducible benchmark run,
not to the unit-test suite.
