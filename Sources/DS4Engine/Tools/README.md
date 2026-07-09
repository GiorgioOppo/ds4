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
  It does not touch chat memory. Every path is confined to the project root
  even in the presence of symlinks (resolved and re-checked; symlinks are also
  never indexed), and `project_edit` rereads its base from disk so an external
  change (git, the user's editor) is never silently reverted. `project_search`
  reads cold files without inserting them into the content cache, so searching
  a project larger than the cache budget cannot evict the files being read.
- **`WebClient.swift`** is the shared SSRF-guarded HTTP client behind
  `web_search`/`web_fetch`: http/https only, public hosts only, size- and
  time-capped responses (see `Builtins/README.md`).
- **`GitTool.swift`** runs whitelisted local git subcommands in the project root.
  Network operations are intentionally excluded. `stash` — the one allowed
  subcommand that rewrites the working tree — triggers an automatic project
  re-index (`project_reload` covers every other out-of-band change).
- **`GitHubTool.swift`** backs the `github_clone` built-in: downloads a public
  GitHub repository as an HTTPS tarball (host pinned to `codeload.github.com`,
  arguments strictly validated, size-capped), extracts it under
  `Application Support/DwarfStar/github-projects`, imports it into
  `ProjectCache` as the active project, and returns a compact orientation
  summary (tree + documentation files) — so the model explores with the
  `project_*` tools instead of paying prefill for the whole repository.
- **`Agents.swift`** defines `AgentProfile` (system prompt, tools, and expert
  profile) and `AgentRegistry`, the shared roster read by `agents_list`.
  Default roles: General, Coding, Code (agentic editing), Reviewer (read-only
  code review), Debug (root-cause + minimal fix), Orchestrator (sub-agents),
  Research (web), Math, Writing, LaTeX, Documentation.
