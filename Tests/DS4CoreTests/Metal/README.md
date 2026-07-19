**English** | [Italiano](README.it.md)

# Metal Tests

Validation for `DS4Metal`, divided by abstraction level:

- `Kernels/`: individual GPU primitives compared with CPU references.
- `Graph/`: model-independent graph primitives.
- `Backends/DeepSeekV4/`: graph, model and decode behavior specific to DeepSeek-V4.
- `Runtime/`: device, library, and pipeline creation.

GPU-dependent cases must skip explicitly when Metal is unavailable. A skipped
test is not a pass; see [`METAL-TESTS.md`](../../METAL-TESTS.md) for conventions.
