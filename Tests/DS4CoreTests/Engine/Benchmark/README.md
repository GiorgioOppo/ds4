**English** | [Italiano](README.it.md)

# Accuracy Benchmark Tests

These tests validate the pure aggregation contract used by the next-token
correctness benchmark. They cover empty input, fully correct and fully wrong
runs, known mixed ratios, cumulative versus local bucket accuracy, and a final
bucket shorter than the configured bucket size. Top-k cases additionally verify
that top-1, top-2 and top-3 remain nested at observation, bucket and result
level; candidate order, de-duplication and truncation to three; short and empty
candidate lists; and exact compatibility of the legacy top-1 aliases.

Sampling-plan tests cover seed determinism, variation between seeds, distinct
target starts, uniform context bounds, non-negative segment starts, per-piece
evaluation bounds, preference for full-length pieces and fallback to a short
corpus tail only when necessary. They also protect corpus/KV clamps, excessive
piece requests and normalization of an inverted context interval.

Piece plan/result indices are zero-based and must cover `0..<pieces.count`;
adding one is presentation-only.

The maximal-run boundary is `N` corpus tokens producing `N - 1` observations
when the unscored prefix contains one token. With a prefix of `C` tokens there
are `N - C` eligible targets before truncation. Keep an explicit off-by-one test
whenever the scoring loop or result model changes.

No test in this directory loads a GGUF, accesses the network, or requires a
Metal device. End-to-end quality measurements depend on a real model and fixed
corpus and are run from the Benchmark panel; these unit tests protect only the
deterministic result and chart aggregation logic.

The legacy single-piece overload remains part of the compatibility contract;
the pure sampling planner can be tested without loading a GGUF or Metal device.

The three ranked values are next-token candidates from the vocabulary, not MoE
experts. A reference token found at rank one is necessarily also correct at
top-2 and top-3; a rank-two hit contributes only to top-2/top-3, and a
rank-three hit only to top-3. Every final partial bucket must divide each rank's
count by its own actual number of evaluated observations.
