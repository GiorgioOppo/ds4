# Native tools

The `DeepSeekTools` target ships a code-agent toolbox the model can
call directly — no MCP round-trip. Inspired by opencode's
`packages/opencode/src/tool/*`; mapping below.

## Catalogue

| Tool | Category | Status | What it does |
|---|---|---|---|
| `read` | readOnly | ✅ | Read a UTF-8 file, line-numbered output, with optional offset/limit. |
| `write` | mutating | ✅ | Create or overwrite a file (atomic). |
| `edit` | mutating | ✅ | Exact-match string replacement; non-unique matches refused unless `replaceAll`. |
| `glob` | readOnly | ✅ | Walk the agent root, match by glob, sort by recent mtime. |
| `grep` | readOnly | ✅ | NSRegularExpression search across files matched by an inner glob. |
| `shell` | dangerous | ✅ | Subprocess via `/bin/zsh -c`, combined output, 32 KB cap, watchdog timeout, optional `sandbox-exec` wrap. |
| `apply_patch` | mutating | ✅ | Minimal unified-diff applier (create / delete / in-place, reverse-order hunks, exact context). |
| `webfetch` | network | ✅ | URLSession GET; HTML → plain text unless `raw=true`. 1 MB cap. |
| `websearch` | network | ⚠️ | DuckDuckGo lite scraper as default backend — fragile by design. Replace with a real API for serious use. |
| `repo_clone` | dangerous | ✅ | Shallow `git clone` via subprocess. Uses user's local git auth. |
| `repo_overview` | readOnly | ✅ | Tree (depth-capped) + extension histogram + content of conventional manifests. |
| `lsp` | readOnly | 🚧 stub | Registers a schema; `run` throws `notImplemented`. Pending: spawn `sourcekit-lsp`, JSON-RPC framing, definition/hover/references/diagnostics. |
| `plan` | planning | ✅ | Read/replace high-level plan note. |
| `task` | planning | ✅ | Manage active task list (list/set/update). |
| `todo` | planning | ✅ | Cross-task TODO bag (list/add/check/uncheck). |

Legend: ✅ shipped · ⚠️ shipped but brittle · 🚧 stub registered, not
yet functional.

## Categories and modes

The registry filters tools by `ToolCategory` according to the active
`AgentMode`:

| Mode | readOnly | planning | mutating | dangerous | network |
|---|---|---|---|---|---|
| **build** | allowed | allowed | allowed (consent) | allowed (consent) | allowed (consent) |
| **plan** | allowed | allowed | **denied** | **denied** | allowed (consent) |

"Consent" means the registry asks the `PermissionDelegate` the first
time per session per `(tool, category)`. The GUI delegate renders
`PermissionPromptView`; the CLI default (`AutoPermissionDelegate`)
auto-allows everything except `.dangerous` unless explicitly opted in.

## Adding a new tool

1. Drop a new file in `Sources/DeepSeekTools/Tools/`.
2. Implement `struct YourTool: Tool` with a `schema: ToolSchema`
   (use `SchemaBuilder` helpers) and a `run(input:context:)`.
3. Add the type to `DefaultTools.standard(...)` so it ships with the
   default registry.
4. Add a smoke test under `Tests/DeepSeekToolsTests/`.
5. If the tool needs new persistence, add a path in
   `PersistencePaths` and a store under `DeepSeekUI/State/`.

## Integration points in DeepSeekUI

- `NativeToolHost` owns the singleton `ToolRegistry` + `PlanStore`,
  bridges the permission gate to a SwiftUI sheet
  (`PermissionPromptView`), and exposes
  `dispatch(name:input:mode:rootDirectory:)` for the inference loop.
- `PermissionStore` persists the user's "always allow / always deny /
  ask" defaults across launches.
- `Settings → Tools` is the read-only inventory; `Settings →
  Permissions` is the rule editor.
- `AgentEditSheet` now carries `agentMode` (Plan / Build) alongside
  the thinking-mode picker.
- `SlashCommandLibrary` provides `/tools`, `/permissions`, `/mode`
  and friends as in-line commands the composer intercepts.

## Future work — see `TODO.md`

- Wire the registry into `InferenceService` (it is the one piece
  that closes the loop with the model — currently scaffolded; the
  remote and local dispatch paths each need a small adapter).
- Implement the `lsp` tool with a real `sourcekit-lsp` client.
- Replace the `websearch` scraper with a configurable provider
  (Tavily / Brave / Serper / Bing).
- Hook `HTTPRecorder` into the OpenRouter session so the tool calls
  it makes can be replayed in tests.
- Tune the `Sandbox.defaultProfile` for normal developer workflows.
