**English** | [Italiano](README.it.md)

# Core tests — Quantization encoders

Byte-exact pinning of `Formats/Quantization`'s GGUF-writer encoders.

- `QuantEncodeTests.swift` compares every implemented output format (q8_0
  with row-offset math, q8_K, q4_K and q2_K in both reference and
  imatrix-weighted variants, iq2_xxs) against fixture bytes.
- `QuantEncodeFixtures.swift` is GENERATED: the inputs are a crafted
  edge-case block (zeros, constants, spikes, alternating signs) followed by
  an xorshift32(0x12345678) stream, and the expected bytes come from the ds4
  `gguf-tools/quants.c` reference compiled with `clang -O2
  -ffp-contract=off` (plain float32 ops on both sides). Regenerate with
  `scripts/quant-fixtures/fixture_gen.c` rather than editing by hand.

A byte diff here means the Swift port diverged from the C reference —
rounding, search order or packing — not a tolerance issue.
