**English** | [Italiano](README.it.md)

# Tensor Kernel Tests

Elemental GPU operations: copy, unary/binary transforms, normalization,
softmax, GLU, row gathering/scattering, concatenation, sorting, and reductions.

Each kernel test must include shape/tail edge cases and a CPU expected value.
Use exact equality only when the implementation contract is bit-identical;
otherwise document the tolerance next to the assertion.

