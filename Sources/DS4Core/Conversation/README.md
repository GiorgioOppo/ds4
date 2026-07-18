# Conversation

Portable conversation types and backend-specific rendering formats.

## Contents

- [`Models/`](Models/README.md): `ToolSpec`, `ToolCall` and `ChatTurn`.
- [`Backends/DeepSeekV4/`](Backends/DeepSeekV4/README.md): DeepSeek V4
  template and DSML tool protocol.
- [`Backends/GLM52/`](Backends/GLM52/README.md): GLM roles, reasoning, flat
  XML tool calls and a compact incremental parser.
- [`Backends/Qwen/`](Backends/Qwen/README.md): documented extension point,
  with no placeholder renderers or parsers.

## Flow and dependencies

The application layer builds a sequence of `ChatTurn`s; the frontend policy
selects DSML for DeepSeek or the native GLM protocol. The rendered text goes
to the tokenizer of the same architecture, and the matching parser
reconstructs any calls. The folder depends only on `DS4Core` types and
Foundation.

## Modification rules

Templates and delimiters are part of each backend's trained protocol: any
variation must be checked against `tokenizer.chat_template`, covered by
tests, and also evaluated for its prefill token cost.
