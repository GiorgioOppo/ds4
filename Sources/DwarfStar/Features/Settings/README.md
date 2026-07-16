# DwarfStar/Features/Settings

- **`Views/SettingsView.swift`** renders global Settings. Model path and context length
  are configured once through `AppSettings` and inherited everywhere: Chat,
  Server, Benchmark, Diagnostics, and Worker. The **Hugging Face** section
  stores the download token in the macOS Keychain via
  `DS4Engine.HFTokenStore` (write-only field, redacted status line; the
  downloader receives it explicitly, winning over `HF_TOKEN` env and
  `~/.cache/huggingface/token`).
  The Model section also opens **Scarica…**, whose sheet is owned by
  `Features/ModelManagement`: the three catalog Flash variants and the
  single-file Pro Q2 may be selected after download, while Pro Q4 split remains
  visibly download-only.
- **`Views/MCPServersView.swift`** renders the MCP panel: `MCPStore` persists the
  configured MCP servers (UserDefaults, `mcpServers`-JSON import/export) and
  pushes them to `MCPManager.shared` (in `DS4Engine/Tools/MCP/`), which owns
  the live connections; the view shows per-server status, exposed tools, and an
  add-server sheet (stdio command or Streamable-HTTP URL).

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
resident. Every setting's UserDefaults key and default value is documented in
the root [Configuration Reference](../../../../README.md#configuration-reference).

Model inspection runs without loading Metal. Benchmark and Memory controls are
shown only when `ModelInfo`/`RuntimeModelDescriptor` advertises
`deepSeekPerformanceTuning`; a recognized Qwen model therefore never receives
DeepSeek-only expert, NSA, bundle or requantization settings.

**Browse** is the advanced path for an external GGUF. It validates the runtime
descriptor before updating `DS4ModelPath` and persists a security-scoped
bookmark only for external files. A catalog model under Application Support is
persisted as an app-managed path and clears a stale external bookmark.

`Views/` owns presentation and the app-facing MCP store. Persistent keys and
defaults must stay backward compatible; model-layout settings apply on the next
load. Credentials always go through Keychain-backed engine helpers and must not
appear in logs, exported settings, or UserDefaults.
