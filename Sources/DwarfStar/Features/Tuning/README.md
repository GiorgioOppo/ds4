# DwarfStar/Features/Tuning

- **`Views/TuningView.swift`** shows expert-cache slots (`DS4ExpertCacheSlots`,
  current measured app preset: 22; see the root
  [Configuration Reference](../../../../README.md#configuration-reference)),
  hit-rate, and routing concentration by layer through the usage imatrix.
- **`Views/AgentsView.swift`** edits agent roles: prompt, icon, per-agent tools, and
  JSON import/export. This is agent management rather than tuning, so it is a
  future candidate for a dedicated `Agents/` folder.

This feature presents and edits engine configuration; it does not implement
expert routing, cache policy, tool authorization, or agent execution. Preserve
that boundary when adding controls.

`TuningView` is capability-gated: without `expertRouting` it displays an
architecture-neutral placeholder instead of DeepSeek cache steppers and usage
statistics.
