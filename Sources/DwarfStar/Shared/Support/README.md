**English** | [Italiano](README.it.md)

# DwarfStar/Shared/Support

Cross-cutting GUI utilities.

- **`EngineLog.swift`** stores a tail buffer of engine logs, shown after chat
  errors to give the user immediate context.
- **`ProcessStream.swift`** provides helpers for absolute paths and subprocess
  output streams.

These helpers are app-layer infrastructure and may not depend on a specific
feature. Keep log capture bounded, preserve stdout/stderr separation, and move
domain behavior into `DS4Core` or `DS4Engine`.
