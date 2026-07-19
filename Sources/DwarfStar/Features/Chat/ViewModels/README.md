**English** | [Italiano](README.it.md)

# Chat View Models

`ChatStore` is the main-actor state owner for the Chat feature. The primary
file declares shared observable state and initialization; focused extensions
separate responsibilities:

- `+ModelLifecycle`: catalog-filtered local discovery, managed/manual path
  restoration, model loading, unloading, and shared-engine exposure.
- `+Generation`: prompt submission, streaming events, cancellation, and errors.
- `+ToolLoop` and `+Agents`: tool execution and sub-agent coordination.
- `+Sessions`: saved-chat lifecycle and history restoration.
- `+Attachments`: file selection and context preparation.
- `+PerformanceSettings`, `+Tuning`, and `+Benchmark`: runtime knobs, cache
  metrics, and the short prefill-setting benchmarks. The measured fast preset
  owns the persisted `DS4MultiQuantCache` choice here and exports it as
  `DS4_MULTI_QUANT_CACHE` before the next model load.
- `+MachineAutoTune`: the GUI adapter for the pure `DS4Engine` auto-tune core.
  It freezes usage data, measures the loaded warm root once, then performs at
  most one fully awaited reload/warmup measurement per unique configuration.
  A run-local cache retains the complete highest valid decode observation;
  revisits are cache hits. Expert-cache slots use a sequential upward-first
  walk, lock direction after a win, and stop on the first loss instead of
  sweeping the whole grid. Promotion requires a strictly higher decode result,
  exact root quality, prefill, stability, RAM and swap gates. Swap accounting keeps cold
  init, warmup and the discarded
  primer in a diagnostic setup window; a fail-closed settling barrier anchors
  the subsequent measured steady window, whose 128 MiB limit alone feeds the
  promotion gate. A root already below the normal 10% RAM floor uses one
  immutable baseline-relative envelope; only memory-neutral/reducing resident
  geometries may then be measured. Candidate configurations must remain
  environment-only; persist the record-holder finalist only after its active-agent
  warmup and final steady swap probe succeed, never an intermediate trial.
  Adoption is a durable transaction whose startup recovery rolls an interrupted
  install/commit back to the full initial snapshot.

Views call this layer; this layer calls `InferenceService`. Keep all published
mutation on the main actor, avoid a second engine instance, and put new behavior
in the extension matching its responsibility rather than growing
`ChatStore.swift`.

An app-managed model under `Application Support/DwarfStar/models` is restored
from its plain persisted path; an external model is restored through the
security-scoped bookmark owned by `ModelPicker`. Do not let bookmark restoration
overwrite a newer managed selection, and do not move catalog/download logic
into this view model.

## Async ownership and tool-result order

Every user turn captures `conversationEpoch`. Stop, New Chat, session activation,
and the next fresh turn advance that epoch as well as cancelling the current
task. Stream events, tool dispatch, sub-agent progress, completion cleanup, and
context/profile callbacks must verify the captured epoch before reading an old
message index or mutating UI state. This second ownership check is required
because cancellation can race with an `AsyncStream` or external-tool await.

For a multi-tool assistant block, keep one result slot per emitted call. Automatic
results, policy denials, duplicate/limit errors, sub-agent results, and manually
entered values fill those original slots. Pass the completed slots to the engine
in call order; grouping automatic results ahead of manual ones changes DSML's
positional call/result association.
