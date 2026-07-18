# Conversation/Backends/DeepSeekV4/DSML

Implementation of the DSML protocol used by the model to declare and emit
tool calls.

## Main files

- [`ToolMarkup.swift`](ToolMarkup.swift): builds DSML delimiters and tags.
- [`ChatRenderer.swift`](ChatRenderer.swift): renders history, tool schemas
  and the generation prompt, with full or compact mode.
- [`ToolCallParser.swift`](ToolCallParser.swift): extracts invocations and
  parameters from the model output.

## Flow

The types in [`Models`](../../../Models/README.md) go into the renderer; the
produced text is tokenized. After generation, the parser reconstructs
`[ToolCall]`; the tool result goes back into the history as
`ChatTurn.toolResult`.

## Modification rules

- Preserve delimiters, `</tool_result>` escaping and stable JSON ordering.
- Treat model input and tool results as untrusted data.
- Measure compact mode both for prefill reduction and for reliability.
- Add round-trip tests when rendering or parsing changes.
