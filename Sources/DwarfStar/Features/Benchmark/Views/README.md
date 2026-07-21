**English** | [Italiano](README.it.md)

# Benchmark Views

`BenchView.swift` renders the Speed/Correctness picker, configuration, progress,
and Swift Charts results. It observes `BenchController` and contains no
inference implementation.

Speed preserves the prefill/generation throughput chart. Correctness provides a
pasted-text editor, seed, piece-count, context-range and per-piece token
controls, KPI cards and a per-piece chart comparing top-1, top-2 and top-3.
Accuracy values from the engine are fractions in `0...1` and are converted to
percentages only for presentation. Global KPI use token-weighted aggregate
counts. A cumulative chart shows progressive aggregate accuracy, while the
per-piece chart keeps samples separate so corpus-position variability remains
visible.

`ChartExport.swift` saves a chart as a self-describing PNG (title, model,
parameters, date) via `ImageRenderer` at fixed size and light appearance, or
its underlying data as CSV, through the sandbox-safe `NSSavePanel`. Each chart
section exposes PNG/CSV buttons; export failures go to the benchmark log.
CSV values keep the engine's raw units (accuracy fractions in `0...1`).

Piece indices in engine results and observations are zero-based. Labels shown
to people use `index + 1`; do not feed that display value back into result
lookup or chart identity.

Use a distinct, stable color and line style for each rank. The legend must name
all three ranks; the chart must not imply that these candidates are MoE experts.

The explanatory copy must continue to say that exact next-token accuracy is not
factual correctness. When Distributed is selected for Correctness, show the
limitation and disable Start rather than silently changing engine.

Keep formatting and chart presentation here. New benchmark modes or metrics
must first be represented by controller state so command execution remains
testable outside the view body.
