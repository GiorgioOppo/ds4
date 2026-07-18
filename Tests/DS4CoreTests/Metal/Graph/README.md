# Metal Graph Tests

This folder keeps the genuinely shared graph operations: `GraphContext`
management and the dense FFN block. The MLA, HyperConnections, NSA, router and
decode layer compositions are tested in the
[`DeepSeek-V4 backend`](../Backends/DeepSeekV4/Graph/README.md).

Graph tests sit above the individual kernels. They must catch wiring, shape,
buffer lifetime and command ordering errors, using CPU/reference components to
compute the expected results. Edge cases of individual kernels stay in
`Kernels/`.
