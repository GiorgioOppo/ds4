**English** | [Italiano](README.it.md)

# Tools/MCP

Connects Model Context Protocol servers and presents their tools alongside
the built-ins.

## Files

- `MCPConfig.swift`: stdio/HTTP configurations and `mcpServers` import/export.
- `MCPProtocol.swift`: JSON-RPC 2.0, initialize, list/call and result parsing.
- `MCPTransport.swift`: stdio processes and Streamable HTTP/JSON/SSE.
- `MCPClient.swift`: actor for handshake, request IDs, timeouts, ping and disconnect.
- `MCPManager.swift`: thread-safe registry of clients, states and namespaced tools.

## Flow

The manager applies the configuration, creates one client per server,
completes `initialize` and keeps a synchronous snapshot of the `ToolSpec`s. A
remote `read_file` tool from the `fs` server is exposed as `mcp_fs_read_file`;
collisions receive a suffix and the reverse mapping stays in the index. The
registry routes the call to the correct client and converts the result to
text.

## Dependencies and lifecycle

Depends on Foundation and `DS4Core`; consumed by [`Core`](../Core/README.md).
Timeouts and cancellation must resolve the pending request immediately.
Change handlers notify consumers when the declarable list changes.

## Extension

Preserve JSON-RPC compatibility, ID-based correlation and notification
handling. Do not derive server/tool by parsing the public name. A new
transport implements `MCPTransport` and must define framing, session,
cancellation and stderr/network diagnostics.

In a sandboxed app, stdio processes inherit the sandbox; for servers that
require broader access, prefer an external process reached via HTTP.
