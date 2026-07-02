# DwarfStar/Settings

- **`SettingsView.swift`** renders global Settings. Model path and context length
  are configured once and inherited everywhere. The Memory section controls
  expert cache, disk KV plus budget, and raw-KV ring behavior. Values apply on
  the **next model load**.
