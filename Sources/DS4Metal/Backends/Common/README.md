# Backends/Common

Intended boundary for high-level contracts that are genuinely
model-independent, such as family identification, declared capabilities and
token/chunk operations. It currently contains no Swift sources: the existing
public APIs remain unchanged during the purely preparatory move.

Runtime and shared operations continue to live in
[`Runtime`](../../Runtime/README.md), [`Graph`](../../Graph/README.md),
[`Kernels`](../../Kernels/README.md) and [`Model`](../../Model/README.md).

A possible common protocol may be invoked at the session boundary, but it
must not introduce type erasure, string lookups or dynamic dispatch into the
hot path of the layers.
