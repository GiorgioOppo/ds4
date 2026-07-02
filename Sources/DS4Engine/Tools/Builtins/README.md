# DS4Engine/Tools/Builtins

One file per built-in tool. Each file adds an
`extension ToolRegistry { static let <tool> = BuiltinTool(...) }`; shared helpers
such as `stringArg`, `intArg`, and `binaryTool` live in `../ToolRegistry.swift`.

| Tool | File | Purpose |
|---|---|---|
| `now` | `Clock.swift` | Current date/time in ISO-8601 format. |
| `calculator` | `Calculator.swift` | Evaluates an arithmetic expression. |
| `add`/`subtract`/`multiply` | `Add`/`Subtract`/`Multiply.swift` | Two-operand arithmetic. |
| `project_list`/`read`/`search` | `Project*.swift` | Explores the indexed project. |
| `project_write`/`edit` | `ProjectWrite`/`ProjectEdit.swift` | Writes or edits indexed text files. |
| `file_read`/`lines`/`write`/`add`/`modify` | `File*.swift` | Raw file access, including line-based reads/edits. |
| `git` | `Git.swift` | Local whitelisted git operations. |
| `agents_list` | `AgentsList.swift` | Lists available roles and tools for orchestration. |
| `subagent_search`/`run` | `Subagent*.swift` | Delegates work to isolated sub-agents; `run` is executed by the engine. |

## Adding A Tool

1. Create `Builtins/NewTool.swift` with the `ToolRegistry` extension.
2. Add the tool to `builtins[]`.
3. Add it to `projectScoped` if it requires an active imported project.
4. Add it to `subAgentGrantable` only when isolated sub-agents may safely use it.
