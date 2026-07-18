# Kernels/Tensor

Wrappers for the generic tensor operations reused by the graph.

## Main files

- `MetalNorm.swift`, `MetalSoftmax.swift`, `MetalGLU.swift`, `MetalUnary.swift` and
  `MetalBin.swift`: normalization, activations and element-wise operations.
- `MetalGetRows.swift`, `MetalSetRows.swift`, `MetalSumRows.swift`: row access
  and row reduction.
- `MetalCopy.swift`, `MetalConcat.swift`, `MetalRepeat.swift`: data movement/reshaping.
- `MetalArgsort.swift`: index sorting.

## Flow and dependencies

The operations are encoded into the same command buffer as the model phases
when possible, working on shared `GPUTensor`s. They are backend primitives and
contain no decode or loading policy.

## Modification rules

Correctly support empty or partial tensors according to the function's
contract and honor `byteOffset`. Make broadcasting, in-place operation and
overlapping ranges explicit. Add small tests with a known CPU result before
using a primitive in a complex fusion.
