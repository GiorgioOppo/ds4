# DwarfStar/Bench

- **`BenchController.swift`** runs the native benchmark. It measures prefill and
  generation throughput in tokens/second at increasing context sizes. The engine
  can be **Local** (the loaded shared in-process engine, gated so chat is idle)
  or **Distributed** (reusing the connected coordinator).
- **`BenchView.swift`** renders the engine selector, context boundaries,
  throughput chart with Swift Charts, and running-engine indicator.
