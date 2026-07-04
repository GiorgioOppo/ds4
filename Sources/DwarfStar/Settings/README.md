# DwarfStar/Settings

- **`SettingsView.swift`** renders global Settings. Model path and context length
  are configured once through `AppSettings` and inherited everywhere: Chat,
  Server, Benchmark, Diagnostics, and Worker.

The panel also selects execution mode:

- **Local** loads the single in-process engine used by Chat, Server, and local
  Benchmark.
- **Distributed** configures the coordinator route and transport options used by
  the Worker cluster.

The Memory section controls the runtime profile used on the next model load:
expert cache slots, direct expert `pread`, expert-bundle sidecar, dense-weight
streaming, best-effort `mlock`, Q4 attention-projection cache, disk KV plus
budget, and raw-KV ring. Most defaults are RAM-aware; the current low-RAM path
prefers streaming and pinned hot buffers over keeping every dense weight
resident.
