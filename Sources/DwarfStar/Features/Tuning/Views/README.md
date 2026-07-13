# Tuning Views

- `TuningView.swift` presents expert-cache sizing, hit rate, and routing
  concentration supplied by the active engine.
- `AgentsView.swift` edits agent prompts, icons, tool grants, and JSON
  import/export.

Views edit engine-backed configuration but do not implement routing or tool
authorization. Validate imported agent data before persisting it, and keep
performance telemetry read-only from the presentation layer.

