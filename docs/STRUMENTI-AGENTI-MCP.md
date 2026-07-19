**English** | [Italiano](STRUMENTI-AGENTI-MCP.it.md)

# Tools, agents, and MCP

The tool system lets the model request structured operations without embedding
application logic in the renderer or the Metal backend. Built-in tools and MCP
tools share the registry and the DSML format exposed to the model.

## Flow of a call

```text
ToolSpec -> ChatRenderer -> DSML prompt -> generated tokens
   -> ToolCallParser -> toolCall event
   -> ChatStore+ToolLoop / DistributedController
   -> ToolRegistry -> execution
   -> structured result -> new tool_result turn -> inference
```

`DS4Core` knows only specs, rendering, and parsing. `InferenceService` emits
the complete call; `ChatStore+ToolLoop` and `DistributedController` orchestrate
the rounds. `DS4Engine` provides the registry and execution. The GUI shows
status and results without reinterpreting the DSML protocol.

## Registry

`ToolRegistry` associates name, description, parameter schema, and handler.
Each agent uses an allow-list to filter the `ToolSpec`s declared in the prompt:
an excluded tool is not presented to the model in the current turn.

This allow-list is an exposure filter, **not** a security boundary and not a
check automatically applied at execution time. The
`ToolRegistry.executeAuto(_:)` API directly resolves any registered built-in or
MCP tool with that name and does not receive the allowed set. A caller that
accepts `ToolCall`s from an untrusted source and requires enforcement must
therefore explicitly check the name against its own policy before invoking
`executeAuto`; registration and allow-listing, on their own, do not authorize
the call.

Built-in categories:

- arithmetic and clock;
- project-restricted files;
- project indexing and editing;
- Git and GitHub import;
- Web fetch and search;
- listing and launching sub-agents.

The tools are grouped under `Sources/DS4Engine/Tools/Builtins`, one per file or
cohesive responsibility.

## Local tool security

Project tools must resolve and normalize paths against the authorized root.
Symlinks, `..`, and absolute paths must not allow escaping the root. Git
operations use a whitelist; the GitHub import must not inherit arbitrary
credentials.

A new tool that modifies files must clearly declare its scope and return
structured errors. Do not use model output as a generic shell command.

## Agents

An agent profile contains:

- id and name;
- system prompt;
- icon/presentation metadata;
- tool allow-list;
- expert-usage profile.

Switching agents changes the application context and the routing profile; it
does not create a second decoder. Profiles are managed under `DS4Engine/Agents`
and presented by the Chat/Tuning features.

## Sub-agents

Sub-agents receive an explicitly delegated question and context. They run an
isolated conversation and return only the final answer to the caller. The KV
cache is content-keyed so compatible prefixes can be reused without
contaminating the main context.

The orchestrator must limit depth, tools, and the amount of content passed to
the sub-agent. Stopping the main chat must propagate to child tasks.

## MCP

`Sources/DS4Engine/Tools/MCP` implements:

- persistable configuration;
- the JSON-RPC protocol;
- stdio transport;
- Streamable HTTP;
- tool discovery and invocation;
- name mapping into the local registry.

Remote tools are exposed as `mcp_<server>_<tool>` to avoid collisions.
Configurations can come from JSON compatible with `mcpServers`.

## stdio transport

The manager launches a child process, sends JSON-RPC messages on stdin, and
reads stdout. Logs or non-protocol text must not be interpreted as a valid
response. Environment and arguments come from the explicit configuration.

## Streamable HTTP

The transport sends JSON-RPC to the configured endpoint, preserves any headers,
and handles responses/streams according to the supported protocol. Tokens and
headers are sensitive data and must not appear in diagnostic logs.

## Adding a built-in tool

1. Create the implementation in the correct category.
2. Define a strict JSON schema and a short description.
3. Validate every parameter before any effect.
4. Register the tool at the central point.
5. Decide which agents may use it.
6. Add tests for success, invalid input, and the security boundary.
7. Update the category README.

## Adding MCP functionality

Keep configuration, transport, and registry mapping separate. Do not introduce
MCP dependencies into `DS4Core` or `DS4Metal`. Cover handshake, tool listing,
invocation, remote errors, timeouts, and cancellation.

## Related documents

- [PIPELINE-INFERENZA.md](PIPELINE-INFERENZA.md)
- [GUI-SERVER-E-API.md](GUI-SERVER-E-API.md)
- [`templates/README.md`](../templates/README.md)
- [`Sources/DS4Engine/Tools/README.md`](../Sources/DS4Engine/Tools/README.md)
