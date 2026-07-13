# Tool Engine Tests

- `ToolRegistryTests.swift` validates registration, grants, schemas, and
  dispatch.
- `MCPTests.swift` covers MCP messages/configuration with isolated fixtures.
- `GitHubToolTests.swift` validates parsing and safe command construction.

Do not invoke real remote services, mutate the developer's repositories, or
depend on installed credentials. Tool tests should inject execution boundaries
and assert authorization failures as carefully as successful calls.

