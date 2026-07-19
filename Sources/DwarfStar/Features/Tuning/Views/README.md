**English** | [Italiano](README.it.md)

# Tuning Views

- `TuningView.swift` presents expert-cache sizing, mixed-quant pool policy, hit
  rate, and routing concentration supplied by the active engine.
- `AgentsView.swift` edits agent prompts, icons, tool grants, and JSON
  import/export.

Views edit engine-backed configuration but do not implement routing or tool
authorization. Direct tuning bindings are disabled for the full benchmark/
machine-auto-tune lease so they cannot persist a hybrid candidate while another
tab owns the run. Validate imported agent data before persisting it, and keep
performance telemetry read-only from the presentation layer.
