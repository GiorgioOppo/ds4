# Conversation/Models

Value types shared between the interface, the engine and the chat renderer.

## Main files

- [`ConversationModels.swift`](ConversationModels.swift): defines `ToolSpec`,
  the schema of an available function; `ToolCall`, an invocation with JSON
  arguments; `ChatTurn`, a typed sequence of system, user, assistant and tool
  result messages.

The types are `Sendable` and `Equatable`; `ToolSpec` and `ToolCall` are also
`Identifiable`, so they can safely cross UI and concurrent services.

## Flow and rules

These models do not execute tools and know nothing about the backend. They are
consumed by [`DSML`](../DSML/README.md) and by the layers above. Add new cases
to `ChatTurn` only while updating renderer, parser, persistence and
exhaustiveness tests; do not put runtime state or application dependencies
here.
