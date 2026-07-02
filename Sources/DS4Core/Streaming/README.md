# DS4Core/Streaming

Support code for streaming model weights from SSD when the full model cannot
remain resident in RAM.

- **`SSDCachePlan.swift`** plans what should stay resident and what can be
  streamed on demand.
- **`SimulatedMemoryLock.swift`** provides a simulated memory lock, useful for
  reasoning about the working set before adding real platform-specific wiring.
