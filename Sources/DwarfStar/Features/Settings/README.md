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
  visibly download-only. The three GLM 5.2 quantizations are downloadable from
  their pinned repository revision but remain download-only until their runtime
  is implemented.
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
expert cache slots, byte-budgeted mixed-quant expert pools, direct expert
`pread`, expert-bundle sidecar, dense-weight streaming, best-effort `mlock`, Q4
attention-projection cache, disk KV plus budget, and raw-KV ring. Most defaults
are RAM-aware; the current low-RAM path
prefers streaming and pinned hot buffers over keeping every dense weight
resident. The fast preset enables `DS4MultiQuantCache` by default after the
exact 2026-07-16 A/B; its toggle can restore the legacy off-class bypass.
Every setting's UserDefaults key and default value is documented in
the root [Configuration Reference](../../../../README.md#configuration-reference).

Model inspection runs without loading Metal. Benchmark and Memory controls are
shown only when `ModelInfo`/`RuntimeModelDescriptor` advertises
`deepSeekPerformanceTuning`; a recognized Qwen model therefore never receives
DeepSeek-only expert, NSA, bundle or requantization settings.

The **Auto-tune record-holder** action uses the pure decision layer in
`DS4Engine/Inference/Autotuning`. It measures the loaded warm root once and each
unique candidate at most once; a repeated configuration is a cache hit without
reload. The whole valid observation with the highest decode median remains the
record. Expert-cache slots are walked upward one manifest step at a time; after
a promotion the first loss ends the knob, while a lower-neighbour fallback is
used only when the initial upward step loses. Promotion requires a strictly
higher decode result, immutable bit-exact full-logit quality, prefill within
−8%, tail/head stability ≥0.75, the run RAM floor and at most 128 MiB of steady swapout. A loaded root below 10% enters constrained mode with
an immutable root-relative floor, a 512 MiB reserve and no positive resident
deltas. Skipping repetitions trades away ABBA noise estimation, so a lucky
record can cause conservative false negatives but cannot bypass any guard.
The usage profile is frozen and Raw-KV ring stays on.
Candidate values remain process-local; Settings persists only the finalist after
successful active-agent warmup and a final steady-state swap probe. A durable
adoption transaction restores the
complete initial preference snapshot after a crash or interrupted commit. The
panel owns progress and Stop, and exposes the generated Markdown/JSON report in
Finder.

**Browse** is the advanced path for an external GGUF. It validates the runtime
descriptor before updating `DS4ModelPath` and persists a security-scoped
bookmark only for external files. A catalog model under Application Support is
persisted as an app-managed path and clears a stale external bookmark.

`Views/` owns presentation and the app-facing MCP store. Persistent keys and
defaults must stay backward compatible; model-layout settings apply on the next
load. Credentials always go through Keychain-backed engine helpers and must not
appear in logs, exported settings, or UserDefaults.
