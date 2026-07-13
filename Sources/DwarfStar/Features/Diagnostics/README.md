# DwarfStar/Features/Diagnostics

- **`Controllers/DiagnosticsController.swift` / `Views/DiagnosticsView.swift`** dump tokens and
  chat-template rendering through the native tokenizer, without subprocesses.
  This is useful when verifying tokenization, prompt rendering, tool schemas, or
  DSML tool-call formatting. The view also embeds `EngineConsole`, a live view
  of the engine's captured stderr (Metal/kernel diagnostics).

The controller performs inspection and exposes observable output; the view only
renders it. Diagnostics must remain read-only with respect to active inference
and KV state.
