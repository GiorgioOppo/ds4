**English** | [Italiano](README.it.md)

# Laguna engine tests

Device-free coverage of the engine's CPU helpers (Q8_0 embedding-row
dequantization against the shared encoder, bounds rejection) plus the opt-in
real-weights smoke test: with `DS4_LAGUNA_GGUF` pointing at the official
Q4_K_M file it loads a truncated stack, runs two decode steps and checks
finite logits and position tracking. Numerical parity against the reference
C engine is the separate gate documented in `docs/PORTING-GAPS.md`.
