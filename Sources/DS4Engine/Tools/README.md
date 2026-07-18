# Tools

Implements the model's function calling: contracts, built-in tools, shared
integrations and external MCP servers.

## Structure

- [`Core`](Core/README.md): `BuiltinTool`, `ToolOutput` and `ToolRegistry`.
- [`Builtins`](Builtins/README.md): one tool per file, grouped by domain.
- [`Integrations`](Integrations/README.md): reusable web, git and GitHub clients.
- [`MCP`](MCP/README.md): MCP configuration, protocol, transports and manager.

The security properties and authorized surfaces are summarized in
[`SICUREZZA.md`](SICUREZZA.md).

## Flow

1. The client asks `ToolRegistry` for the enabled `ToolSpec`s.
2. The model emits the tool name and the argument JSON.
3. `executeAuto` resolves built-ins first, then the MCP index.
4. `ToolOutput` returns to the inference loop as the tool result.

Project tools use [`ProjectCache`](../Projects/README.md); the profiles in
[`Agents`](../Agents/README.md) decide which names are declared.

## Extension

A new tool must have a narrow schema, bounded output, readable errors and a
clear per-project/sub-agent policy. Shared logic, or logic with external side
effects, belongs in `Integrations`, not duplicated in the built-in's closure.
