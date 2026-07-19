**English** | [Italiano](README.it.md)

# Benchmark Controllers

`BenchController.swift` owns benchmark kind, inputs, lifecycle, results, and
error state on the main actor. **Speed** dispatches either to the shared local
`InferenceService` or to the connected distributed coordinator. **Correctness**
calls `InferenceService.accuracyBenchmark` only on the shared local engine; a
Distributed + Correctness selection is visible but cannot be started.

The controller is the only place where a UI benchmark run may mutate engine KV
state. Keep the idle-chat gate intact, publish view-ready `BenchRow` values, and
never load a private model copy from this layer.

Accuracy observations and the final result cross the detached-task boundary via
typed `AsyncStream` continuations. Their consumer tasks inherit `MainActor` and
are the only code that mutates `accuracyObservations` and `accuracyResult`. Keep
this boundary when adding progress, export, or comparison features.

Each observation carries the three ordered next-token candidates. Aggregation
publishes nested top-1, top-2 and top-3 counts and accuracies both globally and
per sampled piece. The global value is derived from total correct/evaluated
counts, not from an unweighted mean of piece percentages. The legacy
`correctTokens`, `accuracy`, `correct`, `localAccuracy` and
`cumulativeAccuracy` names remain top-1 aliases so callers written before the
top-k extension preserve their meaning.

Correctness inputs include a reproducible seed, piece count, context-length
range and maximum evaluated tokens per piece. The plan clamps them to corpus
and KV capacity. Its distinct target starts prevent duplicate samples, but
pieces may overlap; the controller must present the effective plan/result rather
than assuming every requested piece has the maximum length.

Large runs keep exact live counters but retain only a decimated set of
observations for rendering. The controller also raises the effective chart
bucket size when necessary so both live and final charts stay near 4,096 points;
this affects visualization density only, never the correctness totals.

The default correctness corpus is Italian. The backend may clamp or truncate
piece count, context range and evaluated-token limit to the loaded corpus and
context; the final result is authoritative and drives all KPI and charts.
