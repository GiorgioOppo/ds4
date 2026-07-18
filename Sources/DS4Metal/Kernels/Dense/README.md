# Kernels/Dense

Wrappers for the dense multiplications used by projections, output head and
prefill.

## Main files

- [`MetalDense.swift`](MetalDense.swift): F16/Q8_0 matvec and specialized Q4_K
  variants via function constants and simdgroup count.
- [`MetalMatmulMM.swift`](MetalMatmulMM.swift): matrix-matrix and batched paths,
  including inputs organized by id.

## Flow and dependencies

Decode mostly uses batch-one matvec; prefill can aggregate tokens and
use matmul to reuse the weights. `GraphContext` selects pipeline and
configuration based on quantization and tensor shape.

## Modification rules

Validate quantized block divisibility, K/N dimensions and the maximum number
of threads/simdgroups. Do not pick a variant from the byte count alone: the
GGUF type and the logical layout must be explicit. Compare matvec and matmul on
the same inputs.
