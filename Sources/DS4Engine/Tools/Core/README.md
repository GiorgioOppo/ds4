**English** | [Italiano](README.it.md)

# Tools/Core

`ToolRegistry.swift` defines the shared contract and the catalog of built-ins.

## Types and responsibilities

- `BuiltinTool`: a `DS4Core.ToolSpec` plus an execution closure.
- `ToolOutput`: text and metadata returned to the model loop.
- `ToolExecutionPolicy`: explicit allow-list and deny-by-default for each loop.
- `constrainSubAgentTools`: intersects the model's choices with the trusted
  delegation scope of the parent profile; `subagent_run` does not widen
  permissions.
- `ToolRegistry`: the `builtins`, `projectScoped`, and `subAgentGrantable`
  lists, lookup, spec composition, dispatch, and argument helpers.
- `ArithmeticEvaluator`: parser shared with the calculator.

## Flow and dependencies

Depends on Foundation and `DS4Core`. The concrete definitions are registry
extensions in [`Builtins`](../Builtins/README.md); MCP tools are added by
[`MCPManager`](../MCP/README.md).

## Extension

Register each tool exactly once and keep public names stable. If it requires a
project, put it in `projectScoped`; if it is not safe for a sub-agent, exclude
it explicitly. Helpers here must stay generic.

Every execution must pass the same allow-list declared to the model:
`execute(_:policy:)` for built-ins or `executeAuto(_:policy:)` for built-ins
and MCP. A denied call returns `tool_not_allowed`; only a call that is allowed
but unknown returns `nil` and may therefore use the manual flow.
