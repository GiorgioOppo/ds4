# Settings Views

`SettingsView.swift` edits shared model, context, execution, memory, I/O, and
Hugging Face configuration through `AppSettings`. `MCPServersView.swift` and
its `MCPStore` manage persisted MCP server definitions and live connection
status.

The Model section presents both **Browse** and **Scarica…**. Browse validates a
manual GGUF through the Engine selector; Scarica opens the catalog sheet from
`Features/ModelManagement`. The view must not infer runtime support from a
filename: Flash and single-file Pro Q2 selectability, plus Pro Q4's
download-only state and the three GLM 5.2 download-only entries, come from the
Engine catalog.

Settings that change engine memory layout apply on the next model load. Keep
UserDefaults keys centralized and stable, store credentials through engine
Keychain helpers, and never make a feature-specific copy of global settings.
Reload is disabled while a chat generation is active, and local Settings entry
does not repeat the distributed GGUF geometry inspection. Benchmarks and
auto-tuning are likewise unavailable until the active generation stops.
