# Metal Kernel Tests

Focused parity tests for the kernels embedded by `DS4Metal`. Each test should
construct the smallest useful buffer, execute one primitive, and compare its
output with a CPU-faithful reference.

Child directories group attention, compression, dense, MoE, and generic tensor
operations. State tolerances and expected accumulation precision explicitly.
Use the shared embedded runtime rather than a developer-specific kernel path.

