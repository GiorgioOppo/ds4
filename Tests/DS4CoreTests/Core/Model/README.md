# Core Model Tests

`ModelShapeTests.swift` validates model dimensions, derived shape properties,
and architecture constraints that do not require weights or a GPU.

When adding a model-family invariant, cover both a valid configuration and the
specific invalid boundary it rejects.

