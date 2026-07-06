# DS4Engine

The service layer that powers the GUI. It owns the inference actor,
function-calling, agents, KV persistence, model downloads, and distributed
runtime. It depends on `DS4Core` and `DS4Metal` and has no external links.

- **`Service/`**: `InferenceService`, which turns prompts into streamed events,
  reuses KV across turns, runs the tool loop, manages isolated sub-agents, exposes
  benchmarks, and integrates disk KV. Also contains `DiskKVStore`,
  `ExpertBundleTool`, and diagnostics helpers.
- **`Tools/`**: `ToolRegistry`, one file per built-in tool under `Builtins/`,
  the MCP client under `MCP/`, `ProjectCache`, `GitTool`, the SSRF-guarded
  `WebClient`, and agent profiles/registry.
- **`Download/`**: `ModelDownloader`, a native resumable GGUF downloader for
  Hugging Face with SHA-256 integrity verification.
- **`Distributed/`**: distributed inference coordinator/worker protocol,
  transport, and per-node execution logic.
