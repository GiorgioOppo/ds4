# Benchmark Views

`BenchView.swift` renders benchmark configuration, progress, and Swift Charts
results. It observes `BenchController` and contains no inference implementation.

Keep formatting and chart presentation here. New benchmark modes or metrics
must first be represented by controller state so command execution remains
testable outside the view body.

