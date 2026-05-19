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

## Unix toolbox (opt-in)

A 50-tool Unix-style toolbox lives under
`Sources/DeepSeekTools/Tools/Unix/`. Off by default — enable by
passing `includeUnixTools: true` to `DefaultTools.standard(...)`.
Each tool has a fixed binary (or pure-Swift implementation) and
typed parameters — there is **no** general shell/bash escape
hatch.

Shared infrastructure:

- `UnixBinary` (in `_UnixBinary.swift`) — safe subprocess runner
  that sets `terminationHandler` before `run()`, drains pipes
  concurrently, escalates SIGTERM → SIGKILL on timeout, and uses a
  monotonic deadline.
- `UnixWalker` (in `_UnixWalker.swift`) — filesystem walker with
  symlinks NOT followed by default and cycle detection when they
  are.

| Family | Tools |
|---|---|
| Files (10, readOnly) | `ls`, `head`, `tail`, `wc`, `stat`, `du`, `basename`, `dirname`, `find`, `which` |
| Text (10, readOnly) | `sort`, `uniq`, `cut`, `tr`, `paste`, `comm`, `xxd`, `md5`, `sha1`, `sha256` |
| Hash/encode (1, readOnly) | `base64` |
| Text via binary (3, readOnly) | `sed` (no `-i`), `awk` (DSL caveat), `file` |
| Mutating (7) | `touch`, `mkdir`, `cp`, `mv`, `rm` (requires `confirm` for recursive/dir), `ln`, `chmod` |
| Archive (5, mutating) | `tar` (list/extract), `gzip`, `gunzip`, `zip`, `unzip` |
| System (6, readOnly) | `uname`, `date`, `env` (redacts secrets), `hostname`, `whoami`, `id` |
| Process (3) | `ps` (readOnly), `lsof` (readOnly), `kill` (**dangerous**, PID validation) |
| JSON (1, readOnly) | `jq` (probes Homebrew / MacPorts paths) |
| Git (4, readOnly) | `git_status`, `git_log`, `git_diff`, `git_blame` |

Caveats worth knowing:

- `find` is for **exact-name / mtime / size / type** filtering only.
  For glob patterns, use the existing `glob` tool — they're orthogonal
  on purpose so the model picks one cleanly.
- `awk` exposes the full awk DSL, which can read files outside the
  agent root via `getline < "/path"`. The category is `.readOnly`
  because awk has no in-place mutation analog, but for stricter
  sandboxing wrap with `sandbox-exec`.
- `env` redacts variables matching common secret patterns
  (`*_TOKEN`, `*_KEY`, `*_SECRET`, `*_PASSWORD`, `AWS_*`, `OPENAI_*`,
  `ANTHROPIC_*`, `GH_TOKEN`, `GITHUB_TOKEN`) by default.
- `rm` refuses to delete the agent root and requires `confirm: true`
  for any directory or recursive removal.
- `ln` refuses link targets that resolve outside the agent root, to
  prevent the link from widening the sandbox.
- `kill` is `.dangerous` (not `.mutating`): blocked in plan mode,
  refuses PID 1, PID 0, the agent's own PID, and any PID not owned
  by the current UID.
- `jq` is not part of base macOS — install via Homebrew, MacPorts,
  or Nix. The tool probes `/opt/homebrew/bin/jq`, `/usr/local/bin/jq`,
  `/opt/local/bin/jq`; on miss, returns a `not_found` error with a
  `brew install jq` hint.
- `git_log` defaults to `--oneline -n 20` to stay within the 32 KB
  output cap; the model can override `n` for more.

## Xcode / Apple-platform toolbox (opt-in)

A 30-tool toolbox for macOS / iOS / iPadOS / visionOS / watchOS / tvOS
development lives under `Sources/DeepSeekTools/Tools/Xcode/`. Off by
default — enable by passing `includeXcodeTools: true` to
`DefaultTools.standard(...)`. Every Xcode-toolchain command goes
through `/usr/bin/xcrun` (`_Xcrun.swift`) so it picks up the active
Xcode chosen by `xcode-select -p`.

| Family | Tools |
|---|---|
| Build (8) | `xcodebuild_list`, `xcodebuild_build`, `xcodebuild_test`, `xcodebuild_clean`, `xcodebuild_archive`, `xcodebuild_showsdks`, `xcodebuild_showdestinations`, `xcodebuild_exportarchive` |
| Swift PM (3) | `swift_build`, `swift_test`, `swift_package` (resolve / update / init / describe / clean) |
| Simulator (8, mutating) | `simctl_list` (readOnly), `simctl_boot`, `simctl_shutdown`, `simctl_install`, `simctl_launch`, `simctl_uninstall`, `simctl_screenshot`, `simctl_erase` |
| Real device (2) | `devicectl_list` (readOnly), `devicectl_install` (**dangerous** — physical hardware) |
| Signing (3, readOnly) | `codesign_verify`, `codesign_display`, `security_find_identity` |
| Mach-O inspect (2, readOnly) | `otool_info` (header / loadcommands / libraries / symbols / archs), `lipo_info` |
| Plist / version / results (4) | `plutil_print`, `plutil_lint`, `agvtool_version` (mutating), `xcresulttool_get` (parses `.xcresult`) |

Notes:

- `xcodebuild_test` and `xcodebuild_archive` default to a 600-900 s
  timeout (build operations on real projects take minutes). The model
  can override via `timeoutSeconds`.
- All input paths (workspace, project, archive, app bundles,
  exportOptions.plist, …) must resolve inside the agent root. The
  default `derivedDataPath` for `xcodebuild_build`/`test` is `build/`
  so derived data lives inside the repo by default.
- `xcodebuild_build` exposes `noCodesign=true` which sets
  `CODE_SIGNING_ALLOWED=NO` etc. — needed for simulator-only or CI
  builds without provisioning profiles.
- `devicectl_install` is `.dangerous`, not `.mutating`: physical-device
  installs are harder to recover from than simulator state changes.
- `xcresulttool_get` uses the Xcode 16+ `test-results summary/tests`
  invocations for `kind=summary|tests` and falls back to the legacy
  `--legacy object` form when `kind=object` is requested.
- `agvtool_version` operations like `next-build` and `set-marketing`
  write Info.plist files; `read-*` operations are pure reads but the
  category is `.mutating` because the same tool can be used to write.

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
