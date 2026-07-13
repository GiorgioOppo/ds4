# Metal Tests

Validation for `DS4Metal`, divided by abstraction level:

- `Kernels/`: individual GPU primitives compared with CPU references.
- `Graph/`: composed graph operations and layer paths.
- `Decode/`: streaming decoder and expert-cache behavior.
- `Model/`: GGUF-to-Metal weight mapping and Metal I/O policy.
- `Runtime/`: device, library, and pipeline creation.

GPU-dependent cases must skip explicitly when Metal is unavailable. A skipped
test is not a pass; see [`METAL-TESTS.md`](../../METAL-TESTS.md) for conventions.

