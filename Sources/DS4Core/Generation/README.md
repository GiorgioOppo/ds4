# Generation

CPU-side selection of the next token from the logits.

## Main files

- [`Sampler.swift`](Sampler.swift): argmax, xorshift64* RNG, temperature,
  top-k, top-p, min-p and repetition penalties.

## Flow and dependencies

The backend produces the logits; the service passes parameters, recent tokens
and RNG state to `Sampler.sample`; the chosen id returns to the tokenizer and
the generation loop. The implementation uses only Swift/libm and is
GPU-independent.

## Modification rules

Reproducibility with the same seed and parity with the C reference are
functional requirements. Preserve candidate order and the fallback behavior
for non-finite logits; accompany new strategies with separate statistical and
deterministic tests.
