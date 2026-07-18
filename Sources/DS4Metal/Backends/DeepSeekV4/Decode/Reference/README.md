# DeepSeekV4/Decode/Reference

Readable, conservative decoder used as a correctness oracle.

## Main files

- [`DSV4Decoder.swift`](DSV4Decoder.swift): reference decoder with dense
  attention and explicit `OutputHeadWeights`.

## Flow and dependencies

It loads the same shapes and the same weights as the optimized backend, but
favors direct, verifiable passes. Tests compare intermediate outputs or logits
to tell mathematical errors apart from streaming/scheduling problems.

## Modification rules

Do not introduce optimizations here that would make the reference depend on the
path under verification. Mathematical changes must derive from the model
specification and come with tests whose tolerances are justified.
