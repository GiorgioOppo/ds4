**English** | [Italiano](README.it.md)

# Tools/Integrations

Contains side-effecting implementations reused by the built-ins.

## Components

- `WebClient.swift`: HTTP(S), SSRF protection, redirects, timeouts, body limit
  and HTML-to-text conversion.
- `GitTool.swift`: whitelisted subset of local git commands.
- `GitHubTool.swift`: controlled download of public repositories as archives,
  extraction and import into `ProjectCache`.

## Flow and dependencies

The built-ins in [`Web`](../Builtins/Web/README.md) and `Git.swift` translate
JSON into calls to these integrations. The integrations return data or errors
and do not format the model's prompt.

## Extension

Centralize network/process validation and shared limits here. Do not expose
arbitrary shell, private hosts or unvalidated arguments. Every new external
integration must support cancellation and produce bounded output.
