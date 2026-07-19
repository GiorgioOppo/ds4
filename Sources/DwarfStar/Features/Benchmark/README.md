**English** | [Italiano](README.it.md)

# DwarfStar/Features/Benchmark

Native benchmarks for the shared engine. The panel offers two distinct
measurements:

- **Speed** measures prefill and generation throughput at increasing context
  lengths and is available both locally and through the distributed
  coordinator;
- **Correctness** measures how often the next token of a user-provided text
  appears among the model's top one, two or three candidates. It uses teacher
  forcing and is currently available only on the local engine.

## Components

- [`Controllers/BenchController.swift`](Controllers/BenchController.swift)
  builds the data points for both measurements, coordinates execution and
  cancellation, and prevents local runs while the chat is using the same
  engine.
- [`Views/BenchView.swift`](Views/BenchView.swift) presents engine selection,
  corpus and limits, progress state, KPIs and Swift Charts graphs.

The local benchmark reuses the single `InferenceService` already loaded: it
does not create a second copy of the model. The distributed one reuses the
coordinator's active connection. Both modify KV state and must not overlap
with a chat generation.

## Correctness: meaning of the metric

The text is tokenized by the GGUF's tokenizer. The engine selects multiple
segments reproducibly via seed: the first target of each segment is distinct,
even though segment contexts and targets may overlap. For each segment it
draws the context length uniformly between the chosen limits and evaluates up
to the configured maximum number of tokens. It prefers complete segments and
uses the short tail of the corpus only when the requested count makes it
necessary.

The context itself is not evaluated; for each next token the engine always
receives the real previous token and compares the real following token against
the three candidates with the highest logits. This teacher forcing prevents a
single error from derailing the rest of the test. The candidates are
vocabulary tokens, not the three MoE experts selected internally by the
router.

Top-1 counts the cases where the first candidate is exact, top-2 those where
the expected token appears in the first two, and top-3 those where it appears
in the first three. By construction, `top-1 <= top-2 <= top-3` always holds.
It does not measure factual correctness, reasoning ability or the overall
quality of a response. Comparisons between models or configurations are
meaningful only when using the same text, tokenizer, seed, number of segments,
context range and per-segment limit. The UI shows:

- correct/total and top-1, top-2 and top-3 accuracy, duration and tokens
  evaluated per second;
- a cumulative chart of the progressive aggregation of evaluated tokens;
- a comparative top-1, top-2 and top-3 chart for each segment;
- the global aggregate, computed on the overall counts and therefore weighted
  by the number of tokens actually evaluated in each segment.

To keep the interface responsive even with millions of requested tokens, the
controller keeps at most about 4,096 points for the live chart and
automatically increases the minimum block size of the final chart. The
Top-1/2/3 counters and the final result still include every evaluated token:
only the visual density of the points is reduced.

To reproduce a run, the corpus, seed, number of segments, context range and
maximum tokens per segment must be kept identical. Shorter segments must not
carry the same weight as complete ones in the global metric.

Distributed mode is explicitly rejected for Correctness: the current protocol
returns only the logits of the last token of a chunk, and there is no silent
fallback to the local engine.

## Interpretation

- Layer-major prefill tends to amortize fixed costs as tokens increase, but
  chunking, union, caches and memory pressure can change the trend.
- Decode is a token-by-token sequence; on the streaming Flash profile it can
  be dominated by the SSD gather, while at long contexts the weight of
  attention and KV grows.
- The first point and the first tokens may include warm-up, wiring and cold
  caches. Always compare the same steady-state metric.
- Local and distributed results are not comparable if the GGUF, context,
  activation bits, route, caches or engine knobs change.

The old point-in-time numbers were removed from this README because they
lacked date, build, GGUF hash and the complete knob line. Historical
measurements for which an explicit context is available remain in
[`docs/VALUTAZIONE-DEMO-PERF.md`](../../../../docs/VALUTAZIONE-DEMO-PERF.md).

## Required provenance for new measurements

When updating the documentation with real results, record together:

1. date, commit/build and Local or Distributed mode;
2. Mac model, RAM, macOS version and memory/swap pressure;
3. name, size and SHA-256 of the GGUF;
4. context, prompt or corpus, generated tokens and warm-up;
5. the complete `DS4 engine:` line and any settings not covered by that line;
6. warm/cold state of `.usage.json`, `.q4dense`, `.expbundle` and disk KV;
7. for distribution: peers, topology, activation bits and network
   characteristics;
8. mean/percentile used, number of repetitions and observed variability.

Keep the raw report or a stable reference to it. A single figure without this
data must be described as a non-reproducible observation, not as a default or
a regression.

## Modification rules

The controller owns execution and results; the view only controls and
rendering. Maintain mutual exclusion with Chat, do not introduce a second
engine, and update the tests when the definition of the metrics changes.
Updates produced by detached work must go through Sendable streams: never
mutate MainActor observable state directly from the engine.

See also the [controllers README](Controllers/README.md), the
[views README](Views/README.md) and the
[testing guide](../../../../docs/TESTING-E-VALIDAZIONE.md).
