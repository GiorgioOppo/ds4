# DS4Engine/Tools/MCP

MCP (Model Context Protocol) client support: the app connects to external MCP
servers and exposes their tools to the model next to the built-ins.

- **`MCPConfig.swift`** — `MCPServerConfig` (name + transport + enabled) and the
  `{"mcpServers": …}` JSON interchange used by Claude Desktop / Cursor / VS
  Code (`command`/`args`/`env` for stdio servers, `url`/`headers` for HTTP
  ones), so configs can be imported and exported verbatim. The format is also
  documented in the root
  [Configuration Reference](../../../../README.md#configuration-reference).
- **`MCPProtocol.swift`** — pure JSON-RPC 2.0 / MCP frame helpers: build
  `initialize` / `tools/list` / `tools/call` requests, classify incoming frames,
  parse tool lists into `MCPToolInfo`, and flatten call results (text content,
  `structuredContent`, `isError`) into the plain text fed back to the model.
- **`MCPTransport.swift`** — the two spec transports behind one protocol:
  `MCPStdioTransport` spawns the server as a child process and speaks
  newline-delimited JSON-RPC on stdin/stdout (stderr tail kept for
  diagnostics); `MCPHTTPTransport` POSTs frames to a Streamable-HTTP endpoint
  (JSON or SSE responses, `Mcp-Session-Id` echoed).
- **`MCPClient.swift`** — one connection: owns the transport, runs the
  initialize handshake, routes responses by request id with per-request
  timeouts (cancelled on settle), answers server `ping`s, surfaces
  disconnects, and honors task cancellation (the user's Stop settles an
  in-flight call immediately instead of waiting out the timeout).
- **`MCPManager.swift`** — process-wide registry (the `AgentRegistry` pattern):
  the app pushes configs in; the manager owns the clients and serves cached,
  lock-protected synchronous snapshots (statuses, namespaced `ToolSpec`s) to
  every consumer — chat, agents, distributed mode. Server tool `read_file` on
  server `fs` becomes `mcp_fs_read_file` (collisions get a numeric suffix);
  the reverse mapping is an explicit index, never name parsing. Change
  handlers fire on every state transition so consumers re-declare tools when
  a server (dis)connects — `ChatStore` re-syncs the engine's declared tools,
  the MCP panel refreshes — instead of polling.

Tool loops never call this layer directly: `ToolRegistry.executeAuto(_:)`
dispatches built-ins first, then MCP, and `ToolRegistry.autoSpecs(enabled:)`
is the one place that composes the declared specs.

Sandbox note: in a sandboxed (App Store) build, stdio children inherit the app
sandbox, so servers needing broad file/network access should run externally
and be reached over HTTP. Dev builds (`swift run`, `make app`) are unsandboxed.
