# DS4Core/Streaming

Support code for streaming model weights from SSD when the full model cannot
remain resident in RAM.

- **`SSDCachePlan.swift`** plans the streaming cache budget for routed experts
  (as bytes or an explicit expert count) and parses the related arguments; a
  faithful port of `ds4_ssd.c`.
- **`SimulatedMemoryLock.swift`** reserves and `mlock`s a block of anonymous
  memory so streaming behavior can be measured under reduced available RAM
  (port of the C engine's `--simulate-used-memory`).

The runtime streaming knobs (expert cache, pread, dense streaming, ...) are
documented in the root
[Configuration Reference](../../../README.md#configuration-reference).
