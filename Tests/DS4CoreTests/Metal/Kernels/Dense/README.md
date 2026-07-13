# Dense Kernel Tests

`MetalDenseTests.swift` and `MetalMatmulMMTests.swift` validate dense matvec and
matrix-matrix paths across supported shapes and storage formats.

Exercise tail rows/columns and scheduling variants. Performance tuning must not
weaken parity assertions; benchmark throughput separately from correctness.

