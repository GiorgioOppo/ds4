# DS4Engine/Tools

Function-calling support: built-in tools the model can invoke, plus the project
library they operate on.

- **`ToolRegistry.swift`** defines the registry surface: `builtins`,
  `projectScoped`, `subAgentGrantable`, `execute`, `specs`, plus shared helpers
  for argument parsing, arithmetic tools, and expression evaluation.
- **`Builtins/`** contains **one file per tool** using
  `extension ToolRegistry { static let X = BuiltinTool(...) }`. Adding a tool
  means adding a file here and registering it in `builtins[]`.
- **`MCP/`** contains the MCP (Model Context Protocol) client: config,
  JSON-RPC protocol helpers, stdio/HTTP transports, per-server client, and
  `MCPManager.shared`, which exposes connected servers' tools as namespaced
  `ToolSpec`s (`mcp_<server>_<tool>`) next to the built-ins (see `MCP/README.md`).
- **`ProjectCache.swift`** indexes imported projects and backs the `project_*`
  and `file_*` tools for read/list/search/write/edit/add/line-modify operations.
  It does not touch chat memory.
- **`GitTool.swift`** runs whitelisted local git subcommands in the project root.
  Network operations are intentionally excluded.
- **`Agents.swift`** defines `AgentProfile` (system prompt, tools, and expert
  profile) and `AgentRegistry`, the shared roster read by `agents_list`.
  Default roles: General, Coding, Code (agentic editing), Reviewer (read-only
  code review), Debug (root-cause + minimal fix), Orchestrator (sub-agents),
  Research (web), Math, Writing, LaTeX, Documentation.
