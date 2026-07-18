# Tools/Builtins

Contains the built-in tools. Each file adds a `BuiltinTool` property via
`extension ToolRegistry`.

## Groups

- [`Arithmetic`](Arithmetic/README.md): time and deterministic computation.
- [`Files`](Files/README.md): raw access confined to the project.
- [`Projects`](Projects/README.md): navigation of the `ProjectCache` index.
- [`Web`](Web/README.md): protected search and fetch.
- [`Agents`](Agents/README.md): listing and delegation to sub-agents.
- `Git.swift`: whitelisted local git operations.
- `GitHubClone.swift`: controlled import of a public repository.

## Registration

A new tool is created in its domain folder, added to
`ToolRegistry.builtins` and classified as `projectScoped` when necessary.
Update the profiles in [`../../Agents`](../../Agents/README.md) that must
expose it. `subAgentGrantable` includes only operations allowed in delegated
contexts.

## Rules

Always validate the JSON and return short messages. Do not implement network
clients or process invocations directly in the tool file: put that logic in
[`../Integrations`](../Integrations/README.md). See also
[`../SICUREZZA.md`](../SICUREZZA.md).
