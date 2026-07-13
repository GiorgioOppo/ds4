# Quantization Tests

- `HalfTests.swift` verifies half-precision conversion and edge cases.
- `QuantizeTests.swift` validates quantization/dequantization helpers and error
  bounds.

Numerical tests must state whether they require bit equality or tolerance-based
comparison. Include zero, sign, range-limit, non-finite, and incomplete-block
cases when applicable.

