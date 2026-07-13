# Settings Views

`SettingsView.swift` edits shared model, context, execution, memory, I/O, and
Hugging Face configuration through `AppSettings`. `MCPServersView.swift` and
its `MCPStore` manage persisted MCP server definitions and live connection
status.

Settings that change engine memory layout apply on the next model load. Keep
UserDefaults keys centralized and stable, store credentials through engine
Keychain helpers, and never make a feature-specific copy of global settings.

