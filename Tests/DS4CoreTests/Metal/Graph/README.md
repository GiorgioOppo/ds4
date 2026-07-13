# Metal Graph Tests

`Graph*Tests.swift` validates composed `DS4Metal` operations: context setup,
QKV, router, MoE, FFN, compressor, attention variants, HC reduction, complete
decode layers, and decoder execution.

Graph tests sit above individual kernels. They should detect wiring, shape,
buffer-lifetime, and command-order errors while using CPU/reference components
for expected output. Keep kernel-only edge cases in `Kernels/`.

