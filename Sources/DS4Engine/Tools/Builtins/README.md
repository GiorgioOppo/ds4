# DS4Engine/Tools/Builtins

One file per built-in tool. Each file adds an
`extension ToolRegistry { static let <tool> = BuiltinTool(...) }`; shared helpers
such as `stringArg`, `intArg`, and `binaryTool` live in `../ToolRegistry.swift`.

| Tool | File | Purpose |
|---|---|---|
| `now` | `Clock.swift` | Current date/time in ISO-8601 format. |
| `calculator` | `Calculator.swift` | Evaluates an arithmetic expression (`+ - * / % ^`, parentheses, `pi`/`e`, functions like `sqrt`, `sin`, `log`). |
| `add`/`subtract`/`multiply` | `Add`/`Subtract`/`Multiply.swift` | Two-operand arithmetic. |
| `web_search` | `WebSearch.swift` | Web search (DuckDuckGo, keyless); returns title/url/snippet. Override the endpoint with `DS4_SEARCH_URL` (must contain `%@`). |
| `web_fetch` | `WebFetch.swift` | Fetches an http(s) URL and returns readable text (HTML stripped); long pages are read in chunks via `offset`. |
| `project_tree` | `ProjectTree.swift` | Whole-project overview: directories with file counts, in one call. |
| `project_find` | `ProjectFind.swift` | Finds files by name/path (`*` wildcard); contents are `project_search`'s job. |
| `project_list`/`read`/`search` | `Project*.swift` | Explores the indexed project (`read` takes optional `lines` up to 400 per call; `search` accepts an optional `path` scope). |
| `project_write`/`edit` | `ProjectWrite`/`ProjectEdit.swift` | Writes or edits indexed text files. |
| `file_read`/`lines`/`write`/`add`/`modify` | `File*.swift` | Raw file access, including line-based reads/edits. |
| `file_delete` | `FileDelete.swift` | Deletes one file inside the project root (never directories). |
| `git` | `Git.swift` | Local whitelisted git operations. |
| `agents_list` | `AgentsList.swift` | Lists available roles and tools for orchestration. |
| `subagent_search`/`run` | `Subagent*.swift` | Delegates work to isolated sub-agents; `run` is executed by the engine. |

`DS4_SEARCH_URL` and the other environment variables are documented in the root
[Configuration Reference](../../../../README.md#configuration-reference).

## Adding A Tool

1. Create `Builtins/NewTool.swift` with the `ToolRegistry` extension.
2. Add the tool to `builtins[]`.
3. Add it to `projectScoped` if it requires an active imported project.
4. `subAgentGrantable` is derived automatically (every built-in except the
   orchestration tools); nothing to do unless the tool must be kept from
   sub-agents — in that case extend the exclusion list there.
5. Reference it from the relevant `AgentProfile.defaults` tool lists (and
   prompts) in `../Agents.swift` so the right roles actually expose it.

## Web Tools & Safety

`web_search`/`web_fetch` are the only tools that reach the network. Because they
run on model-emitted arguments (attacker-influençable via prompt injection from
an imported project or a fetched page), every request goes through the SSRF
guard in `../WebClient.swift`: **http/https only**, and the host must resolve to
a **public** address — loopback, private (10/8, 172.16/12, 192.168/16, CGNAT),
link-local, and IPv6 ULA/loopback are refused, and each redirect hop is
re-validated. Responses are capped at 3 MB / 20 s. The app needs the
`network.client` entitlement (already present). Note: macOS ATS may block plain
`http://` pages; HTTPS always works.
