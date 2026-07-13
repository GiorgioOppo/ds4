# Chat View Models

`ChatStore` is the main-actor state owner for the Chat feature. The primary
file declares shared observable state and initialization; focused extensions
separate responsibilities:

- `+ModelLifecycle`: model loading, unloading, and shared-engine exposure.
- `+Generation`: prompt submission, streaming events, cancellation, and errors.
- `+ToolLoop` and `+Agents`: tool execution and sub-agent coordination.
- `+Sessions`: saved-chat lifecycle and history restoration.
- `+Attachments`: file selection and context preparation.
- `+PerformanceSettings`, `+Tuning`, and `+Benchmark`: runtime knobs and metrics.

Views call this layer; this layer calls `InferenceService`. Keep all published
mutation on the main actor, avoid a second engine instance, and put new behavior
in the extension matching its responsibility rather than growing
`ChatStore.swift`.

