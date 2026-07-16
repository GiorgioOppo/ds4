# DwarfStar Features

Each child directory implements one user-facing area of the macOS application.
Feature code may depend on `App/`, `Shared/Support`, `DS4Core`, and `DS4Engine`,
but one feature should not own another feature's state.

## Areas

- `Chat/`: conversations, generation state, tools, attachments, and sessions.
- `ModelManagement/`: Engine-catalog-backed GGUF discovery, validated manual
  selection, resumable downloads and package progress.
- `Project/`: project-library bookmarks and project selection.
- `Settings/`: global runtime and MCP configuration.
- `Tuning/`: expert-cache telemetry and agent editing.
- `Server/`: the local OpenAI/Anthropic-compatible HTTP façade.
- `Distributed/`: coordinator and worker controls.
- `Benchmark/`: local and distributed throughput measurements.
- `Diagnostics/`: tokenizer, template, and engine-log inspection.

## Change rules

- Put mutable presentation state in a controller or view model, not in a view.
- Keep reusable inference, persistence, protocol, and tool logic in the engine
  modules. Features adapt those APIs for SwiftUI.
- Read shared model and runtime settings through `AppSettings`; do not create a
  second model instance for an individual panel.
- Add a feature-level README and update this index when introducing an area.
