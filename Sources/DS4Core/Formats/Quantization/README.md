**English** | [Italiano](README.it.md)

# Formats/Quantization

Portable CPU primitives for weight conversion and requantization.

## Key files

- [`Half.swift`](Half.swift): f32/f16 conversions, including the
  architecture-independent software path.
- [`Quantize.swift`](Quantize.swift): Q8_0 dequantization and f32 -> Q4_K
  quantization consistent with the ggml reference quantizers; reference
  Q2_K/Q5_K/Q6_K dequant (GLM 5.2 routed experts) with no local quantizer —
  the GGUF bytes are the fixture.
- [`QuantEncode.swift`](QuantEncode.swift) and
  [`QuantEncodeIQ2XXS.swift`](QuantEncodeIQ2XXS.swift): the GGUF-writer
  ENCODERS (q8_0, q2_K, q4_K, q8_K, iq2_xxs; reference and imatrix-weighted
  variants), a Swift port of ds4 `gguf-tools/quants.c` pinned byte-for-byte
  against the compiled C reference in `QuantEncodeTests`.
- [`GGUFRequantizer.swift`](GGUFRequantizer.swift): offline GGUF -> GGUF
  requantization (the in-process counterpart to ds4
  `gguf-tools/deepseek4-quantize.c` selective `--tensor-type`). Dequantizes
  source tensors (f32/f16/q8_0/q2_K/q4_K/q5_K/q6_K) to f32 and re-encodes to a
  target type via `QuantEncode`, writing through `GGUFWriter`. Pure Swift, no
  GPU; tensors it cannot handle pass through unchanged.

## Flow

The loader reads GGUF Q8_0 blocks, converts them to float, and produces
resident Q4_K caches for the configured paths. The resulting layouts are then
consumed by the `DS4Metal` kernels; this folder performs no GPU dispatch.

## Modification rules

Layout, rounding, scales, and block sizes are part of the contract with the
kernels. Every optimization must keep numerical-parity tests and edge-value
cases; avoid APIs that depend on Metal or Accelerate.
