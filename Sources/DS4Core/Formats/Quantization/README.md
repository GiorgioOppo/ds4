# Formats/Quantization

Portable CPU primitives for weight conversion and requantization.

## Key files

- [`Half.swift`](Half.swift): f32/f16 conversions, including the
  architecture-independent software path.
- [`Quantize.swift`](Quantize.swift): Q8_0 dequantization and f32 -> Q4_K
  quantization consistent with the ggml reference quantizers; reference
  Q2_K/Q5_K/Q6_K dequant (GLM 5.2 routed experts) with no local quantizer —
  the GGUF bytes are the fixture.

## Flow

The loader reads GGUF Q8_0 blocks, converts them to float, and produces
resident Q4_K caches for the configured paths. The resulting layouts are then
consumed by the `DS4Metal` kernels; this folder performs no GPU dispatch.

## Modification rules

Layout, rounding, scales, and block sizes are part of the contract with the
kernels. Every optimization must keep numerical-parity tests and edge-value
cases; avoid APIs that depend on Metal or Accelerate.
