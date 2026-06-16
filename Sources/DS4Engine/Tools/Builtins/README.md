# DS4Engine/Tools/Builtins

Un file per tool integrato. Ogni file è una `extension ToolRegistry { static let <tool> = BuiltinTool(...) }`; gli helper (`stringArg`, `intArg`, `binaryTool`, …) vivono in `../ToolRegistry.swift`.

| Tool | File | Cosa fa |
|---|---|---|
| `now` | `Clock.swift` | data/ora ISO-8601 |
| `calculator` | `Calculator.swift` | valuta un'espressione aritmetica |
| `add`/`subtract`/`multiply` | `Add`/`Subtract`/`Multiply.swift` | aritmetica a 2 operandi |
| `project_list`/`read`/`search` | `Project*.swift` | esplora il progetto (indice) |
| `project_write`/`edit` | `ProjectWrite`/`ProjectEdit.swift` | scrive/edita file di testo indicizzati |
| `file_read`/`lines`/`write`/`add`/`modify` | `File*.swift` | accesso grezzo ai file (anche per riga) |
| `git` | `Git.swift` | git locale (whitelist) |
| `agents_list` | `AgentsList.swift` | elenca ruoli e tool (per l'orchestratore) |
| `subagent_search`/`run` | `Subagent*.swift` | delega a un sub-agent isolato (run gestito dall'engine) |

**Aggiungere un tool**: crea `Builtins/NuovoTool.swift` con la `extension`, poi aggiungi il nome a `builtins[]` (e a `projectScoped` se richiede un progetto).
