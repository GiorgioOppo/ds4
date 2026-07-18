# Inference/Benchmark

Measures the already-loaded backend without depending on the GUI.

## Component

`InferenceService+Benchmark.swift` exposes two distinct measurements:

- `benchmark` uses a synthetic prompt and measures prefill and decode
  throughput;
- `accuracyBenchmark` tokenizes a plain-text corpus and measures the
  **top-1/top-2/top-3 next-token accuracy** with teacher forcing.

The first records both average throughput and per-token speed; `warmup`
initializes kernels and the expert cache once. The second returns progressive
observations and buckets ready for charting, with local and cumulative top-1,
top-2 and top-3 accuracy expressed as `0...1` fractions.

## Next-token correctness

The corpus must contain at least two tokens. The benchmark builds multiple
corpus segments using a reproducible seed. Each segment has a distinct
starting point for its first target; their intervals may still overlap. The
context length is drawn uniformly between the effective minimum and maximum,
without exceeding either the chosen target or the KV capacity. The planner
prefers points that allow evaluating all requested tokens per segment and uses
the corpus's short tail only when more points are needed. Selection uses a
partial, sparse Fisher-Yates: time and memory depend on the number of segments
actually requested, not on the number of possible starting points in the
corpus. For the same seed, increasing the number of segments preserves the
already-selected prefix and its context lengths.

The segment's context is not evaluated; for each subsequent position the model
extracts the three tokens with the highest logits and checks whether the true
token is the first candidate, appears in the top two, or appears in the top
three. These sets are nested, so by construction `top-1 <= top-2 <= top-3`.
The benchmark then always reinserts the true token: a miss neither shifts nor
contaminates the following observations. The BOS → first token step is not
counted.

The three elements are **vocabulary token candidates**, not MoE experts.
Expert selection happens inside the layers and is not measured by this
benchmark.

`StreamingDecoder.prefillTopK` reuses the layer-major prefill: the chunk's
final hiddens already exist, the output head is applied only to the positions
being evaluated, and the three maxima are read from the shared Metal buffer.
No `token × vocab` logits are retained and there is no degradation to the
costly token-by-token `forward`.

Minimum/maximum context, maximum tokens per segment and the number of segments
are clamped to the corpus and the KV window; each segment always evaluates at
least one token. A zero or negative segment count is normalized to one and the
plan reports the clamp via `truncated`. The plan exposes effective values and
truncation, so the caller does not have to infer them from the result. The
global metrics sum the correct and evaluated counts across all segments before
dividing: they are therefore weighted by the tokens actually evaluated, not a
simple average of the segment percentages. The result also keeps the
per-segment values, used by the comparison chart.

The historical `accuracyBenchmark` overload with a single `contextTokens` and
a single `maxTokens` remains available and builds a single-segment plan with a
fixed context. The percentages shown by the UI are each
`topKAccuracy × 100`: the engine always keeps the `0...1` representation.

## Flow and dependencies

The extension operates on the [`Service`](../Service/README.md) actor and uses
samplers and decoders from `DS4Core`/`DS4Metal`. Both measurements alter the
KV and mark it dirty. The correctness benchmark does not modify
`committedIds`, the system prompt or the conversation's logical state: the
next real turn rebuilds the KV. It also saves and restores the usage-imatrix,
so the test corpus does not change the agent's expert profile. Cancellation
and errors follow the same cleanup.

## Extension

A new metric must clearly separate preparation time, prefill, CPU sampling and
GPU decode. Do not change the user's quality parameters during a benchmark,
and keep long loops cancellable. The top-k accuracies measure exact agreement
with that corpus, not semantic correctness: for comparisons across builds,
GGUFs or knobs, always use the same text, prefix and evaluated interval.
