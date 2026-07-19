**English** | [Italiano](SICUREZZA.it.md)

# Tool security

Tool arguments are produced by the model and may derive from untrusted
content. They must therefore be treated as hostile input.

## Files and projects

All paths are confined to the imported root, re-resolved before the I/O and
subject to the rules in
[`../Projects/SICUREZZA-PERCORSI.md`](../Projects/SICUREZZA-PERCORSI.md).
Read, list and search outputs have line, result and byte limits.

## Network

`WebClient` accepts only HTTP(S) to public addresses, re-checks every
redirect and enforces timeouts/size limits. `GitHubTool` uses a fixed host
and validates owner, repository and ref. The local git built-in exposes no
network operations.

## Sub-agents

A sub-agent receives the intersection of the requested tools and
`ToolRegistry.subAgentGrantable`. Tools that change the global project or
orchestrate other agents must not be granted implicitly.

## MCP

MCP servers are external code. Names are namespaced and the inverse mapping
is recorded, not inferred from the string. Stdio processes inherit the app
sandbox; HTTP servers require explicit trust in the endpoint.

## Checklist for a new tool

- minimal JSON schema and type/range validation;
- declared authorization and resource boundary;
- cancellation and timeouts for slow operations;
- bounded output and no secrets in logs;
- tests for malformed arguments and hostile paths/URLs.
