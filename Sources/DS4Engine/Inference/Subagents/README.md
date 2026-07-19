**English** | [Italiano](README.it.md)

# Inference/Subagents

Runs delegated tasks in an isolated context, preserving the main conversation.

## Component and flow

`InferenceService+Subagents.swift` resolves the role, target, and granted
tools; prepares a dedicated prefix; saves the main KV; runs a bounded number
of inference/tool rounds; and finally restores the original context. File or
project prefixes can be reused through a content-addressed KV cache. Project
content, the question, the role prompt, tool schemas, and tool results are
neutralized before adding the trusted framing of the isolated chat.

## Dependencies

Uses [`Agents`](../../Agents/README.md), [`Projects`](../../Projects/README.md),
[`Tools`](../../Tools/README.md), and [`Persistence/KV`](../../Persistence/KV/README.md).

## Extension

- Always intersect the requested tools with both `subAgentGrantable` and
  `allowedTools`, the trusted scope captured from the parent profile. The role
  and arguments chosen by the model may narrow the scope, never widen it.
- Enforce limits on rounds, tokens, and the text reported in the trace.
- Restore the main KV even on error or cancellation.
- Do not implicitly share content or authorizations between agents.
